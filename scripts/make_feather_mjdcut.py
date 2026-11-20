#!/usr/bin/env python3
"""
Build a feather from an existing dataset restricted to an MJD window, with
the timing model refit on the cut data -- e.g. to test how much the edges
of the time span drive a noise-analysis result.

Same steps make_feathers_matched.py applies to every other dataset, on an
already-built par+tim pair instead of the raw data: filter each leaf tim to
[MJD_LO, MJD_HI], drop JUMPs left with no TOAs, refit every fit=1 parameter
(refit_par.py), build the feather (make_feathers.py --build-one) -- twice in
independent subprocesses, accepted only when both agree, against the
tempo2/libstempo instability documented there. The DM(t) correction is NOT
redone: the cut TOAs keep exactly the correction they had, so the only
difference from the parent dataset is which TOAs are in it.

Local, igwn-py38-feathers env (e.g. the common window of LOFAR and the
high-frequency stations other than GMRT, for J1012+5307):
    for d in lofaronly_corrected highfreq_corrected combined_corrected; do
        conda run -n igwn-py38-feathers python scripts/make_feather_mjdcut.py \\
            --dataset $d --mjd 57000 59300
    done
"""
import argparse
import os
import subprocess
import sys

import dm_correction_pipeline as dmcp
from make_feathers_matched import build_matched_tim

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def build_checked(parfile, timfile, outpath, max_attempts=4):
    cmd = [sys.executable, os.path.join(HERE, "make_feathers.py"), "--build-one",
           parfile, timfile, outpath]
    prev = None
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[mjdcut] build attempt {attempt} crashed, retrying ...", flush=True)
            prev = None
            continue
        std = dmcp._read_feather_residual_std_us(outpath)
        if prev is not None and abs(std - prev) < 1e-3 * max(std, prev, 1e-9):
            return std
        print(f"[mjdcut] attempt {attempt}: residual std {std:.3f} us, confirming ...", flush=True)
        prev = std
    raise RuntimeError(f"{outpath}: no two consistent builds in {max_attempts} attempts")


def freeze_redundant_jump(par_path, tim_path):
    """If every TOA carries a flag some JUMP applies to (e.g. LOFAR-only data,
    where all stations are JUMPed relative to the high-frequency reference
    they had in the combined par), JUMPs + Offset are degenerate and tempo2's
    fit can fail. Fix (fit flag 0) the JUMP covering the most TOAs, so that
    station becomes the reference. Returns its flag/value, or None."""
    lines = open(par_path).read().split("\n")
    jumps = {}  # line index -> (flag, value)
    for i, line in enumerate(lines):
        t = line.split()
        if len(t) >= 5 and t[0] == "JUMP" and t[1].startswith("-") and t[4] == "1":
            jumps[i] = (t[1], t[2])
    if not jumps:
        return None
    counts, uncovered = {i: 0 for i in jumps}, 0
    for leaf in dmcp._find_leaf_tim_paths(tim_path):
        for line in open(leaf):
            t = line.split()
            if len(t) < 5 or t[0] in ("C", "#", "FORMAT", "MODE", "INCLUDE") or line.startswith(("C ", "#")):
                continue
            pairs = set(zip(t, t[1:]))
            hit = [i for i, fv in jumps.items() if fv in pairs]
            for i in hit:
                counts[i] += 1
            uncovered += not hit
    if uncovered:
        return None
    i = max(counts, key=counts.get)
    t = lines[i].split()
    t[4] = "0"
    lines[i] = " ".join(t)
    open(par_path, "w").write("\n".join(lines))
    return " ".join(jumps[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pulsar", default="J1012+5307")
    ap.add_argument("--dataset", default="lofaronly_corrected",
                    choices=["lofaronly_corrected", "highfreq_corrected", "combined_corrected",
                             "lofaronly_uncorrected", "highfreq_uncorrected", "combined_uncorrected"],
                    help="which matched dataset to cut (its par/tim under dm_correction_out/)")
    ap.add_argument("--mjd", nargs=2, type=float, default=[57000.0, 60000.0])
    ap.add_argument("--maxobs", type=int, default=30000)
    args = ap.parse_args()

    lo, hi = args.mjd
    tag = f"mjd{lo:.0f}-{hi:.0f}"
    work = os.path.join(REPO, args.pulsar, "dm_correction_out")
    sources = {  # same par/tim pairs make_feathers_matched.py builds each dataset from
        "lofaronly_corrected": ("nodoubles_postfit.forfeather.lofaronly.clean.par",
                                f"matched_correction/{args.pulsar}_lofaronly_corrected.tim"),
        "highfreq_corrected": ("nodoubles_postfit.forfeather.highfreq.clean.par",
                               f"matched_correction/{args.pulsar}_highfreq_corrected.tim"),
        "combined_corrected": ("nodoubles_postfit.forfeather.combined.clean.par",
                               f"matched_correction/{args.pulsar}_dmcorrected.tim"),
        "lofaronly_uncorrected": ("nodoubles_postfit.lofaronly_uncorrected.clean.par",
                                  f"matched_correction/{args.pulsar}_lofaronly_original.tim"),
        "highfreq_uncorrected": ("nodoubles_postfit.highfreq_uncorrected.clean.par",
                                 f"matched_correction/{args.pulsar}_highfreq_original.tim"),
        "combined_uncorrected": ("nodoubles_postfit.matched_full.refit.par",
                                 "nodoubles_postfit_matched_full.tim"),
    }
    par_name, tim_name = sources[args.dataset]
    par, tim = os.path.join(work, par_name), os.path.join(work, tim_name)

    cut_tim = os.path.join(work, f"{args.pulsar}_{args.dataset}_{tag}.tim")
    kept, dropped = build_matched_tim(dmcp._find_leaf_tim_paths(tim), lo, hi,
                                      os.path.join(work, "matched_tims", f"{args.dataset}_{tag}"),
                                      cut_tim)
    print(f"[mjdcut] {args.pulsar} {args.dataset}: kept {kept} TOAs, dropped {dropped} "
          f"outside MJD {lo:.0f}-{hi:.0f}", flush=True)

    stem = os.path.join(work, os.path.splitext(par_name)[0] + f".{tag}")
    clean_par = stem + ".par"
    n_jumps = dmcp.strip_orphaned_jumps(par, cut_tim, clean_par)
    if n_jumps:
        print(f"[mjdcut] dropped {n_jumps} orphaned JUMP(s)", flush=True)
    frozen = freeze_redundant_jump(clean_par, cut_tim)
    if frozen:
        print(f"[mjdcut] every TOA is JUMPed: fixed JUMP {frozen} (reference station)", flush=True)
    refit_par = stem + ".refit.par"
    subprocess.run([sys.executable, os.path.join(HERE, "refit_par.py"),
                    clean_par, cut_tim, refit_par, str(args.maxobs)], check=True)

    out = os.path.join(REPO, f"feathers_{args.dataset}_{tag}", args.pulsar)
    std = build_checked(refit_par, cut_tim, out)
    print(f"[mjdcut] done -> {out} (residual std {std:.3f} us)", flush=True)


if __name__ == "__main__":
    main()
