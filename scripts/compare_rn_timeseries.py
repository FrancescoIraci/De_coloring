"""Red-noise time series of several SPNA runs of one pulsar (e.g. lofaronly
uncorrected vs corrected), from plot_gp_timeseries.py's gp_timeseries.npz.

Top row: the red-noise reconstructions (median + 16-84% band) overlaid.
One row per run below: that run's own timing residuals -- raw TOAs in grey,
1-day epoch averages in black -- with its red-noise band on top. The
residuals come from the npz itself (saved by plot_gp_timeseries.py), i.e.
exactly the ones the reconstruction was conditioned on.

By default the bands have the timing-model projection removed draw by draw
(rednoise_tmfree_*): a steep red noise puts most of its power in
quadratic-like modes that F0/F1 absorb, so the full reconstruction can be
hundreds of microseconds while the residuals stay flat. --raw shows it.

Local, no discovery needed:
    python scripts/compare_rn_timeseries.py --pulsar J1012+5307 \
        --runs feathers_lofaronly_uncorrected:rnonly feathers_lofaronly_corrected:rnonly
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# white outline: keeps the red-noise line readable on top of the residual points
HALO = [pe.withStroke(linewidth=3.5, foreground="white")]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_posteriors import case_style  # noqa: E402


def epoch_average(t, y, err, gap=1.0):
    """Inverse-variance weighted mean per epoch (TOAs within `gap` days of
    the epoch's first TOA); t must be sorted."""
    edges, t0 = [0], t[0]
    for i in range(1, len(t)):
        if t[i] - t0 > gap:
            edges.append(i)
            t0 = t[i]
    edges.append(len(t))
    tm, ym, em = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        w = 1.0 / err[a:b] ** 2
        tm.append(np.average(t[a:b], weights=w))
        ym.append(np.average(y[a:b], weights=w))
        em.append(1.0 / np.sqrt(w.sum()))
    return np.array(tm), np.array(ym), np.array(em)


def load(base, dataset, model, pulsar, raw):
    """{t_mjd, res_us, err_us, q16, q50, q84} for one run's red noise; by
    default the timing-model-free bands (what the residuals can see)."""
    path = os.path.join(base, dataset, pulsar, model, "gp_timeseries.npz")
    d = np.load(path)
    key = "rednoise" if raw else "rednoise_tmfree"
    if "res_us" not in d.files or f"{key}_q50" not in d.files:
        raise SystemExit(f"{path} lacks res_us or {key}_q* -- rerun plot_gp_timeseries.py "
                         "(newer versions save both)")
    out = {k: d[k] for k in ("t_mjd", "res_us", "err_us")}
    out.update({q: d[f"{key}_{q}"] for q in ("q16", "q50", "q84")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pulsar", required=True)
    ap.add_argument("--runs", required=True, nargs="+", metavar="DATASET:MODEL")
    ap.add_argument("--plot_output_base", default="plot_output")
    ap.add_argument("--raw", action="store_true",
                    help="plot the full reconstruction, including the part the "
                         "timing model absorbs (default: timing-model-free)")
    ap.add_argument("--out", default=None,
                    help="default: plot_output/compare/<pulsar>_rn_timeseries.pdf")
    args = ap.parse_args()

    runs = [tuple(s.split(":", 1)) for s in args.runs]
    data = {rk: load(args.plot_output_base, *rk, args.pulsar, args.raw) for rk in runs}
    style = {rk: case_style(rk, i) for i, rk in enumerate(runs)}

    fig, axes = plt.subplots(1 + len(runs), 1, figsize=(11, 2.8 * (1 + len(runs))),
                             sharex=True)
    ax0 = axes[0]
    for rk in runs:
        d, (color, label, ls) = data[rk], style[rk]
        ax0.fill_between(d["t_mjd"], d["q16"], d["q84"], color=color, alpha=0.25)
        ax0.plot(d["t_mjd"], d["q50"], color=color, ls=ls, lw=1.4, label=label)
    ax0.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax0.set_ylabel(r"red noise [$\mu$s]")
    ax0.legend(fontsize=9, loc="upper left", handlelength=3)

    for ax, rk in zip(axes[1:], runs):
        d, (color, label, ls) = data[rk], style[rk]
        t, res, err = d["t_mjd"], d["res_us"], d["err_us"]
        te, re, ee = epoch_average(t, res, err)
        ax.plot(t, res, ".", ms=1.5, color="0.65", alpha=0.5, zorder=1,
                label="residuals (TOAs)")
        ax.errorbar(te, re, yerr=ee, fmt="o", ms=2.5, lw=0.6, color="k", zorder=2,
                    label="residuals (1-day epoch averages)")
        ax.fill_between(t, d["q16"], d["q84"], color=color, alpha=0.35, zorder=3)
        ax.plot(t, d["q50"], color=color, ls=ls, lw=1.8, zorder=4, path_effects=HALO,
                label=f"red noise ({label})")
        # epoch averages set the scale: single-channel TOAs scatter far wider
        # and would flatten everything else
        lo = min(np.percentile(re, 0.5), np.percentile(d["q16"], 0.5))
        hi = max(np.percentile(re, 99.5), np.percentile(d["q84"], 99.5))
        pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylabel(r"residual [$\mu$s]")
        ax.set_title(label, fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right", ncol=3)
    axes[-1].set_xlabel("MJD")
    what = "full, incl. timing-model-absorbed part" if args.raw else "timing-model part removed"
    fig.suptitle(f"{args.pulsar}: red-noise reconstruction ({what})", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or os.path.join("plot_output", "compare", f"{args.pulsar}_rn_timeseries.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
