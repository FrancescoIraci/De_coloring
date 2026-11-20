"""Overlay posteriors from several spna_sampling.py runs of the SAME pulsar,
as corner plots, for direct comparison -- e.g. "does the DM-GP amplitude
collapse once the data is DM(t)-corrected?" or "does red noise change
between the high-freq-only and the combined dataset?".

Reuses plot_posteriors.py's chain-loading (same h5 layout, same cold-chain-
is-every-ntemps-th-row fix) -- no discovery/jexplore needed, runs anywhere
the raw epochs_*.h5 files are (so: on the cluster, not after an rsync that
leaves those behind).

A "run" is a (dataset, model) pair, i.e. samples/<dataset>/<pulsar>/<model>/,
given as DATASET:MODEL or DATASET:MODEL:BURN_EPOCHS -- the per-run burn
overrides --burn_epochs, since runs of different datasets/models generally
need different burn-in.

Splitting logic: only the stochastic, non-white-noise parameters are
considered (red noise, DM GP, chromatic, SW GP, n_earth -- same convention
as plot_posteriors.py's own corner.pdf; white noise is excluded by default,
there are too many per-backend params for a readable corner and they are
not the point of this comparison). Each such parameter is assigned to the
exact SET of given runs that has it (e.g. "rednoise_log10_A" -> {rnonly,
dmrn} if both model it; "dmgp_log10_A" -> {dmrn} only, since rnonly has no
DM GP at all). One corner page is produced per distinct run-set found,
overlaying exactly the runs in that set -- so parameter sets that don't
match across runs automatically end up on their own page instead of being
forced into a shared corner that doesn't exist for all of them. With
--joint, everything goes in one corner instead (see plot_joint).

Usage
-----
    # highfreq_corrected, with vs without a DM GP
    python scripts/compare_posteriors.py --pulsar J1744-1134 \
        --runs feathers_highfreq_corrected:rnonly:4 feathers_highfreq_corrected:dmrn:4

    # one RN+DM corner, three station sets + an rnonly run
    python scripts/compare_posteriors.py --pulsar J1012+5307 --joint \
        --runs feathers_combined_corrected:dmrn:12 feathers_highfreq_corrected:dmrn:4 \
               feathers_lofaronly_corrected:dmrn:7 feathers_combined_corrected:rnonly:19

    # include white-noise groups too (one corner per shared backend)
    python scripts/compare_posteriors.py --pulsar J1744-1134 --runs ... --include_wn
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

try:
    import corner as corner_module
except ImportError:
    corner_module = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_posteriors import load_prior_layout, load_chains, group_of  # noqa: E402

WN_SUFFIXES = {"efac", "log10_tnequad", "log10_t2equad", "log10_ecorr"}

# Okabe-Ito palette (colorblind-safe). Color = dataset, line style = model,
# fixed across every plot: the same data always has the same color, and
# whether a DM GP is in the model reads from the line style. combined and
# highfreq keep corrected/uncorrected in the same hue family (dark/light).
# Labels spell out "DM+RN" / "RN only" instead of "dmrn"/"rnonly": the
# adjacent r-n in those render as a near-"m" in matplotlib's default font.
DATASET_STYLE = {
    "feathers_combined_corrected":    ("#D55E00", "combined, corrected"),
    "feathers_combined_uncorrected":  ("#E69F00", "combined, uncorrected"),
    "feathers_highfreq_corrected":    ("#0072B2", "highfreq, corrected"),
    "feathers_highfreq_uncorrected":  ("#56B4E9", "highfreq, uncorrected"),
    "feathers_lofaronly_corrected":   ("#009E73", "lofaronly, corrected"),
    "feathers_lofaronly_uncorrected": ("#CC79A7", "lofaronly, uncorrected"),
    # MJD-cut variants: Okabe-Ito has no hue left, so darker tones from Paul
    # Tol's (also colorblind-safe) muted set, each in its parent dataset's family
    "feathers_combined_corrected_mjd57000-59300":  ("#882255", "combined, corrected, MJD 57000-59300"),
    "feathers_highfreq_corrected_mjd57000-59300":  ("#332288", "highfreq, corrected, MJD 57000-59300"),
    "feathers_lofaronly_corrected_mjd57000-59300": ("#117733", "lofaronly, corrected, MJD 57000-59300"),
    "feathers_combined_uncorrected_mjd57000-59300":  ("#999933", "combined, uncorrected, MJD 57000-59300"),
    "feathers_highfreq_uncorrected_mjd57000-59300":  ("#88CCEE", "highfreq, uncorrected, MJD 57000-59300"),
    "feathers_lofaronly_uncorrected_mjd57000-59300": ("#CC6677", "lofaronly, uncorrected, MJD 57000-59300"),
    "feathers_lofaronly_corrected_mjd57000-60000": ("#AA4499", "lofaronly, corrected, MJD 57000-60000"),
}
MODEL_STYLE = {"dmrn": ("-", "DM+RN"), "rnonly": ("--", "RN only")}
FALLBACK_COLORS = ["#000000", "#7F7F7F", "#F0E442"]

LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2.0))  # 1 and 2 sigma
LINEWIDTH = 1.6
GROUP_PRIORITY = {"rednoise": 0, "dmgp": 1, "chromatic": 2, "swgp": 3, "sw": 3}


def case_style(run_key, fallback_idx):
    """(color, label, linestyle) for a (dataset, model) run_key."""
    dataset, model = run_key
    if dataset in DATASET_STYLE:
        color, ds_label = DATASET_STYLE[dataset]
    else:
        print(f"  [warn] {dataset} has no fixed color in DATASET_STYLE -- using a fallback")
        color = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
        ds_label = dataset.replace("feathers_", "")
    ls, model_label = MODEL_STYLE.get(model, (":", model))
    return color, f"{ds_label} — {model_label}", ls


def hist2d_kwargs(color, ls):
    """Contour lines only (no density shading): overlaid runs stay readable
    and the dashed/solid model distinction stays visible."""
    return dict(color=color, plot_datapoints=False, plot_density=False,
                no_fill_contours=True, levels=LEVELS,
                contour_kwargs=dict(linestyles=ls, linewidths=LINEWIDTH))


def corner_kwargs(color, ls):
    return dict(hist2d_kwargs(color, ls), show_titles=False,
                hist_kwargs=dict(density=True, linestyle=ls, linewidth=LINEWIDTH))


def legend_handles(run_keys, style):
    return [Line2D([0], [0], color=style[rk][0], ls=style[rk][2], lw=2.5, label=style[rk][1])
            for rk in run_keys]


def load_run(samples_dir_base, dataset, model, pulsar, burn_epochs, burn_iters, thin):
    """One run's physical-space samples, keyed by short parameter name.
    Returns {short_name: 1d_array} or None if the run isn't there."""
    run_dir = os.path.join(samples_dir_base, dataset, pulsar, model)
    prior_path = os.path.join(run_dir, "prior_dict.json")
    if not os.path.exists(prior_path):
        print(f"  [skip] {dataset}:{model} -- no prior_dict.json in {run_dir}")
        return None

    flat_names, short_names, lows, highs, _ = load_prior_layout(prior_path, pulsar)
    try:
        p, _ = load_chains(run_dir, burn_epochs, burn_iters)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [skip] {dataset}:{model} -- {e}")
        return None

    nwalker, ndim, niters = p.shape
    physical = lows[None, :, None] + p * (highs - lows)[None, :, None]
    flat = physical.transpose(1, 0, 2).reshape(ndim, -1)  # (ndim, nwalker*niters)
    if thin > 1:
        flat = flat[:, ::thin]
    return {short: flat[i] for i, short in enumerate(short_names)}


def corner_figure(loaded, names, style, title, bins=20):
    """One corner over `names`, overlaying every run in `loaded`. A run with
    every name goes in as a normal corner overlay; a run with only some of
    them (e.g. rnonly, no dmgp_*) is drawn only in the panels whose
    parameters it actually has, the rest of its row/column stays empty."""
    K = len(names)
    ranges = []
    for n in names:
        vals = np.concatenate([d[n] for d in loaded.values() if n in d])
        lo, hi = np.percentile(vals, [0.5, 99.5])
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    full = [rk for rk, d in loaded.items() if all(n in d for n in names)]
    if not full:
        raise SystemExit("need at least one run that has every plotted parameter")
    partial = [rk for rk in loaded if rk not in full]

    fig = None
    for rk in full:
        color, _, ls = style[rk]
        samples = np.vstack([loaded[rk][n] for n in names]).T
        fig = corner_module.corner(samples, fig=fig, labels=names, range=ranges, bins=bins,
                                   **corner_kwargs(color, ls))
    axes = np.array(fig.axes).reshape(K, K)

    for rk in partial:
        color, _, ls = style[rk]
        idx = [i for i, n in enumerate(names) if n in loaded[rk]]
        for i in idx:
            axes[i, i].hist(loaded[rk][names[i]], bins=bins, range=ranges[i], density=True,
                            histtype="step", color=color, linestyle=ls, linewidth=LINEWIDTH)
            for j in idx:
                if j < i:
                    corner_module.hist2d(loaded[rk][names[j]], loaded[rk][names[i]],
                                         ax=axes[i, j], bins=bins, range=[ranges[j], ranges[i]],
                                         **hist2d_kwargs(color, ls))

    # corner resets each diagonal's ylim to the latest overlaid run's peak,
    # clipping taller ones drawn earlier -- rescale to the tallest of all
    for i, n in enumerate(names):
        peak = max(np.histogram(d[n], bins=bins, range=ranges[i], density=True)[0].max()
                   for d in loaded.values() if n in d)
        axes[i, i].set_ylim(0, 1.1 * peak)

    # reserve headroom above the panel grid so the title and legend each get
    # their own row instead of overlapping -- bbox_inches="tight" on save
    # crops the unused margin back out
    fig.subplots_adjust(top=0.82)
    fig.legend(handles=legend_handles(full + partial, style), loc="upper right",
               bbox_to_anchor=(0.98, 0.89), frameon=True, fontsize=11, handlelength=3)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)
    return fig


def parse_run(spec, default_burn):
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise SystemExit(f"bad --runs entry {spec!r}: expected DATASET:MODEL[:BURN_EPOCHS]")
    burn = int(parts[2]) if len(parts) == 3 else default_burn
    return (parts[0], parts[1]), burn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples_dir_base", default="samples")
    ap.add_argument("--pulsar", required=True)
    ap.add_argument("--runs", required=True, nargs="+", metavar="DATASET:MODEL[:BURN]",
                     help="two or more runs to overlay; an optional third field sets "
                          "that run's own burn-in epochs")
    ap.add_argument("--burn_epochs", type=int, default=15,
                     help="burn-in for runs given without their own :BURN")
    ap.add_argument("--burn_iters", type=int, default=0)
    ap.add_argument("--thin", type=int, default=20,
                     help="keep every Nth sample (corner plots get slow/cluttered "
                          "at full chain length; default 20)")
    ap.add_argument("--include_wn", action="store_true",
                     help="also make a corner page per shared white-noise backend "
                          "(default: only red noise / DM / chromatic / SW / n_earth)")
    ap.add_argument("--joint", action="store_true",
                     help="one corner over all non-WN parameters together (e.g. RN+DM, "
                          "4x4) instead of one page per group; runs lacking some of "
                          "them (rnonly) are drawn only in their own panels")
    ap.add_argument("--out", default=None,
                     help="output PDF (default: plot_output/compare/<pulsar>.pdf, "
                          "or <pulsar>_joint.pdf with --joint)")
    args = ap.parse_args()

    if corner_module is None:
        raise SystemExit("the 'corner' package is required: pip install --user corner")

    runs = [parse_run(spec, args.burn_epochs) for spec in args.runs]
    if len(runs) < 2:
        raise SystemExit("need at least 2 --runs to compare")

    print(f"pulsar : {args.pulsar}")
    loaded = {}
    for (dataset, model), burn in runs:
        data = load_run(args.samples_dir_base, dataset, model, args.pulsar,
                         burn, args.burn_iters, args.thin)
        if data is not None:
            loaded[(dataset, model)] = data
            print(f"  [ok]   {dataset}:{model} (burn {burn}) -- {len(data)} parameters, "
                  f"{next(iter(data.values())).size} samples (thinned)")

    if len(loaded) < 2:
        raise SystemExit(f"need at least 2 loadable runs to compare, got {len(loaded)}")

    # name -> set of run_keys that have it, restricted to non-WN groups
    # unless --include_wn
    name_runs = {}
    for run_key, data in loaded.items():
        for short in data:
            grp, suf = group_of(short)
            if not args.include_wn and suf in WN_SUFFIXES:
                continue
            name_runs.setdefault(short, set()).add(run_key)

    if not name_runs:
        raise SystemExit("no comparable (non-WN) parameters found at all -- "
                          "pass --include_wn to compare white noise too")

    style = {rk: case_style(rk, i) for i, rk in enumerate(loaded)}

    if args.joint:
        names = sorted(name_runs, key=lambda s: (GROUP_PRIORITY.get(group_of(s)[0], 9), s))
        out_path = args.out or os.path.join("plot_output", "compare", f"{args.pulsar}_joint.pdf")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig = corner_figure(loaded, names, style, args.pulsar)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {out_path} ({', '.join(names)})")
        return

    # partition parameter names by their exact run-set, keep only run-sets
    # with >=1 run that actually has data (always true here) -- group the
    # names within one run-set by their group_of() label, one corner page
    # per (run_set, group)
    by_runset_group = {}
    for short, run_set in name_runs.items():
        grp, _ = group_of(short)
        key = (frozenset(run_set), grp)
        by_runset_group.setdefault(key, []).append(short)

    def sort_key(item):
        (run_set, grp), _ = item
        return (len(loaded) - len(run_set), GROUP_PRIORITY.get(grp, 9), grp)

    pages = sorted(by_runset_group.items(), key=sort_key)

    out_path = args.out or os.path.join("plot_output", "compare", f"{args.pulsar}.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with PdfPages(out_path) as pdf:
        for (run_set, grp), names in pages:
            names = sorted(names)
            run_keys = [rk for rk in loaded if rk in run_set]  # stable order
            n_only = "" if len(run_set) == len(loaded) else \
                f" (only: {', '.join(style[rk][1] for rk in run_keys)})"
            fig = corner_figure({rk: loaded[rk] for rk in run_keys}, names, style,
                                f"{args.pulsar}: {grp}{n_only}")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\nwrote {out_path} ({len(pages)} page(s))")
    for (run_set, grp), names in pages:
        print(f"  - {grp} ({', '.join(sorted(names))})  "
              f"runs: {[style[rk][1] for rk in loaded if rk in run_set]}")


if __name__ == "__main__":
    main()
