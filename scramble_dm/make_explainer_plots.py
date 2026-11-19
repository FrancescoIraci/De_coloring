"""
Explanatory plots for the frequency-scrambling test (scramb_dm_claude.py) --
not the test's own output, just visual intuition for what it's doing.

Two figures per pulsar:
  {psr}_residuals_vs_freq.png   -- raw residuals against observing frequency
                                    (log-x), the direct diagnostic a chromatic
                                    (nu^-idx) signal would show up in.
  {psr}_scramble_demo.png       -- TOA time vs frequency, true assignment next
                                    to one example random permutation: the
                                    actual operation scramb_dm.py performs
                                    niter times per pulsar.

Local only, discovery-local env:
    /opt/anaconda3/envs/discovery-local/bin/python3 scramble_dm/make_explainer_plots.py \
        --feathers 'feathers_combined_corrected/*' --psrs J1012+5307 --outdir scramble_dm/out/explainer
"""
import os
import argparse
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_residuals_vs_freq(psr, outpath, seed=0):
    freqs = np.asarray(psr.freqs)
    res_us = np.asarray(psr.residuals) * 1e6
    backends = np.asarray(psr.backend_flags)

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(seed)
    for b in np.unique(backends):
        sel = backends == b
        color = rng.random(3) * 0.75  # muted, but every backend distinguishable
        ax.scatter(freqs[sel], res_us[sel], s=6, alpha=0.5, color=color, linewidths=0)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("observing frequency [MHz]")
    ax.set_ylabel(r"residual [$\mu$s]")
    ax.set_title(f"{psr.name}: residuals vs. observing frequency\n"
                 f"({len(np.unique(backends))} backends, {len(freqs)} TOAs, colors = backend)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_scramble_demo(psr, outpath, seed=0, n_examples=16):
    """'Before -> after' for a small, explicit set of example TOAs: each
    one's TRUE observing frequency (left) connected to the frequency it
    gets handed in ONE example scramble (right). One TOA per backend (the
    real test has thousands of TOAs; this keeps it small enough to read),
    and -- to make the example fully self-contained and literally
    checkable -- the permutation shown is drawn only from these N TOAs'
    own frequencies, not the whole dataset's (the real scramb_dm() permutes
    across every TOA in the fit at once; the mechanism is identical, just
    pooling far more TOAs than the handful drawn here).
    """
    freqs = np.asarray(psr.freqs)
    backends = np.asarray(psr.backend_flags)

    # one representative TOA per backend, backends sorted by frequency so
    # the plot's left column reads as a frequency-ordered ladder
    uniq_backends = sorted(np.unique(backends), key=lambda b: np.median(freqs[backends == b]))
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.where(backends == b)[0]) for b in uniq_backends]
    if len(idx) > n_examples:
        keep = np.linspace(0, len(idx) - 1, n_examples).round().astype(int)
        idx = [idx[i] for i in keep]
    idx = np.array(idx)

    ex_freqs, ex_backends = freqs[idx], backends[idx]
    ex_freqs_scrambled = rng.permutation(ex_freqs)

    fig, ax = plt.subplots(figsize=(8, 7))
    rng_c = np.random.default_rng(0)
    colors = {b: rng_c.random(3) * 0.8 for b in ex_backends}
    for f_true, f_scr, b in zip(ex_freqs, ex_freqs_scrambled, ex_backends):
        c = colors[b]
        ax.plot([0, 1], [f_true, f_scr], color=c, lw=1.3, alpha=0.8, marker="o", ms=6, label=b)

    ax.set_yscale("log")
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["true\n(this TOA's real frequency)", "scrambled\n(what it gets handed)"])
    ax.set_ylabel("observing frequency [MHz]")
    ax.set_title(f"{psr.name}: one example scramble, {len(idx)} example TOAs "
                 f"(one per backend)\neach line = one TOA, keeping its own time/residual "
                 f"but a reassigned frequency", fontsize=11)
    # a text label next to every point gets unreadable where several backends
    # cluster at similar frequencies (common for this project's data) -- a
    # side legend (same colors) scales to crowded clusters instead
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              frameon=False, title="backend", title_fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feathers", required=True, help="glob, e.g. 'feathers_combined_corrected/*'")
    ap.add_argument("--psrs", nargs="+", help="only these pulsars (default: all matched)")
    ap.add_argument("--outdir", default="scramble_dm/out/explainer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=16)
    args = ap.parse_args()

    import discovery as ds

    os.makedirs(args.outdir, exist_ok=True)
    for path in sorted(glob.glob(args.feathers)):
        psr = ds.Pulsar.read_feather(path)
        if args.psrs and psr.name not in args.psrs:
            continue
        print(f"{psr.name} ({len(psr.toas)} TOAs)")
        plot_residuals_vs_freq(psr, os.path.join(args.outdir, f"{psr.name}_residuals_vs_freq.png"),
                                seed=args.seed)
        plot_scramble_demo(psr, os.path.join(args.outdir, f"{psr.name}_scramble_demo.png"),
                            seed=args.seed, n_examples=args.n_examples)


if __name__ == "__main__":
    main()
