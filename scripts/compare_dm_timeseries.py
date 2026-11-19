"""Overlay the GP-reconstructed DM(t) from an SPNA run (plot_gp_timeseries.py's
gp_timeseries.npz) against the epoch-wise DM(t) the correction pipeline
measured directly from LOFAR (DMts.npy, from dm_correction_pipeline.py's
epochwise_dm step) -- a visual check of whether the DM(t) correction already
absorbed the structure the epoch-wise fit found, or whether the GP on the
corrected data still tracks it.

Both sides are plotted as their own GP: the epoch-wise points are refit with
the identical Matern-3/2 GP used by dm_correction_pipeline.py's fit_gp() (not
imported from there, to keep this script importable without libstempo/tempo2
-- the fitting logic is small enough to duplicate, and must stay byte-for-
-byte identical to the pipeline's own, including random_state=0, or the two
would silently diverge) rather than plotted as raw scattered points -- the
earlier version plotted raw epoch-wise points with their quoted error bars,
and a handful of huge-error outliers (not clipped by the pipeline's own
sigma-clip, which only drops points that are outliers relative to their OWN
error) blew up the y-range on every panel. Comparing two smooth curves is
also the more apples-to-apples question: "did the post-correction DM GP
still find the same structure the pre-correction GP smoothing found", not
"does a single post-correction GP band happen to bracket noisy raw points".

Local script, no discovery needed (pure numpy/matplotlib/sklearn): gp_timeseries.npz
must already exist (pull it from the cluster after running
scripts/run_gp_timeseries.sh there), and DMts.npy only ever existed locally
-- it was never uploaded to the cluster.

Units: the GP band is a delay in microseconds at --fref (1400 MHz by
default, matching plot_gp_timeseries.py's default); converted here to DM
(pc/cm^3) via the standard dispersion relation so both curves are directly
comparable. The epoch-wise curve is linearly detrended (its DM-fit par has
DM1=DM2=0, so it mixes secular drift with the stochastic variation; a
1st-order detrend is a simple, defensible way to put it on the same "just
the variation" footing as the GP curve, not an exact match to whatever
DM1/DM2 the corrected run's timing model actually fit) -- the SAME detrend
coefficient (fit from the raw epoch-wise points) is subtracted from the
refit GP's own mean curve, so both stay on that footing together.

Colors match case_style in compare_posteriors.py: the post-correction GP
uses that dataset's own color (so it matches its appearance in every
corner-plot comparison elsewhere in the summary); the epoch-wise GP is
black, since every hue in that palette already stands for a dataset.

Usage:
    python scripts/compare_dm_timeseries.py --pulsar J1744-1134 \
        --dataset feathers_combined_corrected --model dmrn
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_posteriors import case_style  # noqa: E402

DAY = 86400.0
K_DM = 4.148808e3   # standard dispersion constant: delay[s] = K_DM * DM[pc/cm^3] / freq[MHz]^2
PRE_COLOR = "#000000"  # neutral: in the summary's palette every hue already means a dataset


def _fit_epochwise_gp(mjd, dm_val, dm_err, max_dm_err=1.0, sigma_clip=5.0):
    """Byte-for-byte copy of dm_correction_pipeline.py's fit_gp()/_fit_one_gp()
    -- duplicated (not imported) so this script stays import-able without
    libstempo/tempo2; see module docstring. Keep in sync if the pipeline's
    GP ever changes."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    mask = np.isfinite(dm_val) & (dm_err < max_dm_err)
    mjd, dm_val, dm_err = mjd[mask], dm_val[mask], dm_err[mask]
    order = np.argsort(mjd)
    mjd, dm_val, dm_err = mjd[order], dm_val[order], dm_err[order]

    t0 = mjd.min()
    t = (mjd - t0).reshape(-1, 1)
    y_mean = dm_val.mean()
    y = dm_val - y_mean

    def _fit_one(t, y, dm_err):
        kernel = ConstantKernel(1e-4, (1e-12, 10)) * Matern(length_scale=100.0, length_scale_bounds=(1.0, 1e5), nu=1.5) \
                 + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-1))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=dm_err**2, n_restarts_optimizer=10,
                                       normalize_y=False, random_state=0)
        gp.fit(t, y)
        return gp

    gp = _fit_one(t, y, dm_err)
    pred_mean, pred_std = gp.predict(t, return_std=True)
    z = (y - pred_mean) / np.sqrt(dm_err**2 + pred_std**2)
    keep = np.abs(z) < sigma_clip
    if (~keep).sum():
        t, y, dm_err = t[keep], y[keep], dm_err[keep]
        gp = _fit_one(t, y, dm_err)

    def predict(mjd_query):
        tq = (np.asarray(mjd_query) - t0).reshape(-1, 1)
        mean, std = gp.predict(tq, return_std=True)
        return mean + y_mean, std

    return predict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pulsar", required=True)
    ap.add_argument("--dataset", default="feathers_combined_corrected")
    ap.add_argument("--model", default="dmrn")
    ap.add_argument("--samples_dir_base", default="plot_output",
                     help="base of the pulled-down plot_output tree (default: plot_output)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    npz_path = os.path.join(args.samples_dir_base, args.dataset, args.pulsar,
                             args.model, "gp_timeseries.npz")
    dmts_path = os.path.join(args.pulsar, "dm_correction_out", "matched_correction", "DMts.npy")
    if not os.path.exists(npz_path):
        raise SystemExit(f"not found: {npz_path} -- run scripts/run_gp_timeseries.sh "
                          f"on the cluster and git pull first")
    if not os.path.exists(dmts_path):
        raise SystemExit(f"not found: {dmts_path} -- epoch-wise DM time series "
                          f"(local-only, never pushed)")

    gp = np.load(npz_path)
    if "dmgp_q50" not in gp:
        raise SystemExit(f"{npz_path} has no dmgp band -- is --model/--dataset a dmrn run?")
    fref = float(gp["fref"])
    t_gp = gp["t_mjd"]
    q16_us, q50_us, q84_us = gp["dmgp_q16"], gp["dmgp_q50"], gp["dmgp_q84"]

    # delay [us] <-> DM [pc/cm^3] at fref (positive conversion factor, so
    # percentile order is preserved); to_delay_us is to_dm's inverse, used
    # for the secondary (right-hand) axis on each panel below
    to_dm = lambda delay_us: (delay_us * 1e-6) * fref ** 2 / K_DM
    to_delay_us = lambda dm: dm * K_DM / fref ** 2 * 1e6
    q16_dm, q50_dm, q84_dm = to_dm(q16_us), to_dm(q50_us), to_dm(q84_us)

    dmts = np.load(dmts_path)
    t_ew, dm_ew, dmerr_ew = dmts[:, 0], dmts[:, 1], dmts[:, 2]
    ok = np.isfinite(dm_ew)
    t_ew, dm_ew, dmerr_ew = t_ew[ok], dm_ew[ok], dmerr_ew[ok]

    # linear detrend (see module docstring -- DM1/DM2 were forced to 0 in
    # the epoch-wise fit, so this curve mixes secular drift with variation);
    # the coefficient is fit from the raw points, then applied to the refit
    # GP's own mean curve below, so both stay on the same footing
    coeffs = np.polyfit(t_ew, dm_ew, 1)

    # refit the epoch-wise GP (same kernel/settings as the correction
    # pipeline's own fit_gp -- see _fit_epochwise_gp docstring) and evaluate
    # it on the post-correction GP's own time grid, so both curves share an
    # x-axis; this also makes the earlier full-range/zoomed-range split
    # unnecessary -- two smooth GP curves don't have the raw-outlier
    # blow-up problem a scatter of epoch-wise points with huge error bars did
    ew_predict = _fit_epochwise_gp(t_ew, dm_ew, dmerr_ew)
    ew_mean, ew_std = ew_predict(t_gp)
    ew_mean_detrended = ew_mean - np.polyval(coeffs, t_gp)

    post_color, post_label, _ = case_style((args.dataset, args.model), 0)
    gp_label = f"post-correction GP ({post_label})"
    ew_label = "epoch-wise GP (LOFAR), linearly detrended"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(t_gp, q16_dm, q84_dm, alpha=0.3, color=post_color, label=gp_label)
    ax.plot(t_gp, q50_dm, lw=1.3, color=post_color)
    ax.fill_between(t_gp, ew_mean_detrended - ew_std, ew_mean_detrended + ew_std,
                     alpha=0.3, color=PRE_COLOR, label=ew_label)
    ax.plot(t_gp, ew_mean_detrended, lw=1.3, color=PRE_COLOR)
    ax.axhline(0.0, lw=0.5, color="k", alpha=0.4)
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$\Delta$DM [pc cm$^{-3}$]")
    secax = ax.secondary_yaxis("right", functions=(to_delay_us, to_dm))
    secax.set_ylabel(rf"delay at {fref:.0f} MHz [$\mu$s]")
    ax.set_title(f"{args.pulsar}: DM(t) -- post-correction GP vs the original "
                 f"epoch-wise GP it was corrected with")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path = args.out or os.path.join("plot_output", "compare",
                                         f"{args.pulsar}_dm_timeseries_ew_vs_gp.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  post-correction GP ({args.dataset}:{args.model}): {len(t_gp)} points, fref={fref:.0f} MHz, "
          f"median DM range [{q50_dm.min():.2e}, {q50_dm.max():.2e}] pc/cm^3")
    print(f"  epoch-wise GP: refit from {len(t_ew)} points (of {len(ok)} total, {ok.sum()} finite), "
          f"detrend slope={coeffs[0]:.3e} pc/cm^3/day")


if __name__ == "__main__":
    main()
