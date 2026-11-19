"""
Epoch-Wise Total-DM Fitting
===========================
Recover the *total* DM time series by fitting the dispersion measure
independently on each single-epoch observation of a multi-frequency
pulsar timing dataset.

Differences w.r.t. epochwise_dm.py
----------------------------------
1. Epoch splitting is keyed on the **observation filename** (column 1 of
   the .tim line), not on a frequency heuristic. Every TOA produced from
   the same archive belongs to the same epoch, no matter how many
   frequency channels survived outlier trimming. This makes it impossible
   to leak TOAs across observation boundaries. Byte-identical duplicate
   TOA lines are dropped.

2. Before each per-epoch fit, **all** timing parameters are frozen and
   only DM is freed, so the single-epoch fit is well conditioned even
   when the global .par has F0/F1/astrometry/binary/JUMPs flagged for
   fitting. (tempo2 always fits an implicit constant phase offset, which
   removes the DM/absolute-phase degeneracy.)

3. DM1 and DM2 are **zeroed** so that the per-epoch DM absorbs the full
   dispersion: the result is the *total* DM(t) time series.

4. Each epoch is fitted in an **isolated subprocess**. tempo2 occasionally
   segfaults at the C level on individual epochs; isolating the fit means
   such an epoch is recorded as NaN and the run continues instead of the
   whole process dying. A per-epoch timeout also guards against hangs.

Usage
-----
    python epochwise_dm_total.py \
        -n J1640+2224 \
        -d J1640+2224 \
        -p nodoubles_postfit.par \
        -t lofar/J1640+2224_ToAs_no_doubles.tim \
        -o J1640+2224/dm_correction_out/epochwise \
        [--plot] \
        [--keep-epochs] \
        [--timeout 120]
"""

import os
import sys
import argparse
import subprocess
from collections import OrderedDict

import numpy as np
import libstempo as LT


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Recover the total DM time series with epoch-wise DM fitting"
    )
    p.add_argument("-n", "--name",    default="J1640+2224", metavar="NAME",
                   help="Pulsar name")
    p.add_argument("-d", "--dirpath", required=True, metavar="DIR",
                   help="Folder containing the par and tim files")
    p.add_argument("-p", "--parfile", required=True, metavar="FILE",
                   help="Timing parameter (.par) file, relative to dirpath")
    p.add_argument("-t", "--timfile", required=True, metavar="FILE",
                   help="TOA (.tim) file, relative to dirpath")
    p.add_argument("-o", "--outdir",  required=True, metavar="DIR",
                   help="Output directory for per-epoch .tim files and results")
    p.add_argument("--min-freqs", type=int, default=2, metavar="N",
                   help="Minimum distinct frequencies needed to fit an epoch "
                        "(default: 2)")
    p.add_argument("--timeout", type=float, default=120.0, metavar="SEC",
                   help="Per-epoch fit timeout in seconds (default: 120)")
    p.add_argument("--plot", action="store_true",
                   help="Save a DM time-series plot to <outdir>/DM.pdf")
    p.add_argument("--keep-epochs", action="store_true",
                   help="Keep per-epoch .tim files in <outdir> after fitting")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Epoch splitting (by observation filename)
# ---------------------------------------------------------------------------

def split_epochs(tim_path: str, outdir: str) -> int:
    """
    Split a .tim file into one file per observation, grouping TOA lines by
    the archive filename in column 1. Epochs are emitted in order of first
    appearance. Byte-identical duplicate TOA lines are dropped. Returns the
    number of epoch files written.
    """
    with open(tim_path) as f:
        lines = f.readlines()

    header_prefixes = ("FORMAT", "MODE", "C ", "#", "INCLUDE", "SKIP",
                       "NOSKIP", "TIME", "EFAC", "EQUAD")

    groups = OrderedDict()   # filename -> list of TOA lines
    seen_toas = set()        # drop byte-identical duplicate TOA lines
    n_dup = 0
    for line in lines:
        s = line.strip()
        if not s or s.startswith(header_prefixes):
            continue
        cols = s.split()
        if len(cols) < 4:
            continue
        if s in seen_toas:            # exact duplicate TOA -> skip
            n_dup += 1
            continue
        seen_toas.add(s)
        filename = cols[0]            # archive path uniquely tags the epoch
        groups.setdefault(filename, []).append(line)

    if n_dup:
        print(f"[INFO] {n_dup} duplicate TOA line(s) dropped")

    for idx, toa_block in enumerate(groups.values()):
        out_path = os.path.join(outdir, f"{idx}.tim")
        with open(out_path, "w") as out:
            out.write("FORMAT 1\nMODE 1\n")
            out.writelines(toa_block)

    n_epochs = len(groups)
    print(f"[INFO] {n_epochs} epoch files written to {outdir}/")
    return n_epochs


# ---------------------------------------------------------------------------
# Single-epoch worker (runs in an isolated subprocess)
# ---------------------------------------------------------------------------

def write_dmfit_par(src: str, dst: str):
    """
    Write the .par used for the per-epoch DM fit. The full par is kept;
    only the following lines are modified:

      * DM1, DM2, NE_SW  ->  "<NAME> 0 0"  (value 0, fit flag off)
      * every other parameter keeps its value but its fit flag is set to 0
      * DM is left untouched (it already carries fit flag 1)

    NE_SW must be zeroed here (on disk, before loading): tempo2 applies the
    solar-wind delay at load time, so changing psr['NE_SW'].val afterwards
    has no effect on the fit.
    """
    out = []
    for raw in open(src):
        s = raw.rstrip("\n")
        t = s.split()
        if not t or s.lstrip().startswith(("#", "C ")):
            out.append(s)
            continue
        name = t[0]
        if name in ("DM1", "DM2", "NE_SW"):     # zero value, fit flag off
            out.append(f"{name:<14s} 0 0")
        elif name == "DM":                      # leave as is (already fit=1)
            out.append(s)
        elif name == "JUMP":                    # freeze JUMPs (flag at index 4)
            if len(t) >= 5 and t[4] in ("0", "1"):
                t[4] = "0"
            out.append(" ".join(t))
        else:                                   # turn off any other fit flag
            if len(t) >= 3 and t[2] in ("0", "1"):
                t[2] = "0"
            out.append(" ".join(t))
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")


def _free_only_dm(psr):
    """Safety net: ensure nothing but DM is fitted (the par already enforces
    this, but guard against any par-format edge case)."""
    for par in psr.pars(which="all"):
        try:
            psr[par].fit = False
        except Exception:
            pass
    psr["DM"].fit = True


def fit_one_epoch():
    """
    Worker entry point. Loads one epoch, fits DM, and prints a single
    result line to stdout:

        RESULT <mjd> <dm> <dm_err> <ntoa>     fit succeeded
        SKIP   <mjd> <ntoa> <nfreq>           too few distinct frequencies
        FAIL   <reason>                       python-level failure

    A C-level segfault simply kills this process (nonzero exit code); the
    parent treats that as a failed epoch.
    """
    wp = argparse.ArgumentParser()
    wp.add_argument("--fit-one", nargs=2, metavar=("TIM", "PAR"))
    wp.add_argument("--min-freqs", type=int, default=2)
    a = wp.parse_args()
    tim_path, par_path = a.fit_one

    try:
        psr = LT.tempopulsar(par_path, tim_path)
    except Exception as exc:
        print(f"FAIL load: {exc}")
        return

    toas  = psr.toas()
    mjd   = float(np.mean(toas))
    ntoa  = len(toas)
    nfreq = len(np.unique(np.round(psr.freqs, 3)))

    if nfreq < a.min_freqs:
        print(f"SKIP {mjd:.8f} {ntoa} {nfreq}")
        return

    try:
        _free_only_dm(psr)
        psr.fit()
        print(f"RESULT {mjd:.8f} {psr['DM'].val:.10f} "
              f"{psr['DM'].err:.6e} {ntoa}")
    except Exception as exc:
        print(f"FAIL fit: {exc}")


# ---------------------------------------------------------------------------
# DM fitting (parent: drives one isolated subprocess per epoch)
# ---------------------------------------------------------------------------

def fit_dm_per_epoch(par_path: str, outdir: str, n_epochs: int,
                     min_freqs: int, timeout: float) -> np.ndarray:
    """
    Fit DM on each per-epoch .tim file in an isolated subprocess.

    Returns
    -------
    dm_ts : ndarray, shape (n_epochs, 4)
        Columns: [MJD_mean_TOA, DM_fit, DM_err, n_TOA]
    """
    mjds    = np.full(n_epochs, np.nan)
    dm_vals = np.full(n_epochs, np.nan)
    dm_errs = np.full(n_epochs, np.nan)
    ntoa    = np.zeros(n_epochs)

    self_path = os.path.abspath(__file__)

    for obs in range(n_epochs):
        tim_path = os.path.join(outdir, f"{obs}.tim")
        cmd = [sys.executable, self_path, "--fit-one", tim_path, par_path,
               "--min-freqs", str(min_freqs)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[WARNING] Epoch {obs}: timed out after {timeout:.0f}s.")
            continue

        out = ""
        for ln in res.stdout.splitlines():
            if ln.startswith(("RESULT", "SKIP", "FAIL")):
                out = ln
                break

        if res.returncode != 0 and not out.startswith(("SKIP", "RESULT")):
            print(f"[WARNING] Epoch {obs}: worker crashed "
                  f"(exit {res.returncode}, likely tempo2 segfault).")
            continue

        tok = out.split()
        if out.startswith("RESULT"):
            mjds[obs]    = float(tok[1])
            dm_vals[obs] = float(tok[2])
            dm_errs[obs] = float(tok[3])
            ntoa[obs]    = int(tok[4])
            print(f"  Epoch {obs:>4d}/{n_epochs-1}  MJD={mjds[obs]:.2f}  "
                  f"nTOA={int(ntoa[obs]):>3d}  "
                  f"DM={dm_vals[obs]:.6f} +/- {dm_errs[obs]:.2e}")
        elif out.startswith("SKIP"):
            mjds[obs] = float(tok[1])
            ntoa[obs] = int(tok[2])
            print(f"  Epoch {obs:>4d}/{n_epochs-1}  MJD={mjds[obs]:.2f}  "
                  f"only {tok[3]} freq(s) -> skipped")
        else:
            reason = out[5:].strip() if out else "no output"
            print(f"[WARNING] Epoch {obs}: fit failed ({reason}).")

    return np.column_stack((mjds, dm_vals, dm_errs, ntoa))


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def save_dm_plot(dm_ts: np.ndarray, outdir: str):
    import matplotlib.pyplot as plt

    mjds, dm, dm_err = dm_ts[:, 0], dm_ts[:, 1], dm_ts[:, 2]
    mask = np.isfinite(dm)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(mjds[mask], dm[mask], yerr=dm_err[mask],
                fmt=".", color="darkblue", ecolor="steelblue",
                alpha=0.8, capsize=2)
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"DM  [pc cm$^{-3}$]")
    ax.set_title("Total DM time series (epoch-wise fit)")
    fig.tight_layout()
    out_path = os.path.join(outdir, "DM.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Worker mode: fit a single epoch and exit (called as a subprocess).
    if "--fit-one" in sys.argv:
        fit_one_epoch()
        return

    args = parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    par_path = os.path.join(args.dirpath, args.parfile)
    tim_path = os.path.join(args.dirpath, args.timfile)

    psr  = LT.tempopulsar(par_path, tim_path)
    print(f"[INFO] Pulsar : {args.name}")
    print(f"[INFO] TOAs   : {len(psr.toas())}")

    # Build the par used for the per-epoch fit (DM1=DM2=NE_SW=0, only DM free)
    # and save it alongside the results.
    stem, ext = os.path.splitext(os.path.basename(par_path))
    fit_par = os.path.join(outdir, f"{stem}.dmfit{ext}")
    write_dmfit_par(par_path, fit_par)
    print(f"[INFO] DM-fit par written -> {fit_par} (DM1=DM2=NE_SW=0, only DM free)")

    n_epochs = split_epochs(tim_path, outdir)

    print("\n[INFO] Starting epoch-wise total-DM fitting ...\n")
    dm_ts = fit_dm_per_epoch(fit_par, outdir, n_epochs,
                             args.min_freqs, args.timeout)

    n_ok = int(np.isfinite(dm_ts[:, 1]).sum())
    print(f"\n[INFO] {n_ok}/{n_epochs} epochs fitted successfully")

    npy_path = os.path.join(outdir, "DMts.npy")
    np.save(npy_path, dm_ts)
    print(f"[INFO] DM time series saved -> {npy_path} "
          f"(columns: MJD, DM, DM_err, nTOA)")

    if args.plot:
        save_dm_plot(dm_ts, outdir)

    if not args.keep_epochs:
        for obs in range(n_epochs):
            ep_path = os.path.join(outdir, f"{obs}.tim")
            if os.path.exists(ep_path):
                os.remove(ep_path)
        print("[INFO] Per-epoch .tim files removed (use --keep-epochs to retain)")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
