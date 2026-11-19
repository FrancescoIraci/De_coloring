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

Default: J1012+5307 lofaronly corrected, MJD 57000-60000. Local,
igwn-py38-feathers env:
    python scripts/make_feather_mjdcut.py
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pulsar", default="J1012+5307")
    ap.add_argument("--dataset", default="lofaronly_corrected",
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
    refit_par = stem + ".refit.par"
    subprocess.run([sys.executable, os.path.join(HERE, "refit_par.py"),
                    clean_par, cut_tim, refit_par, str(args.maxobs)], check=True)

    out = os.path.join(REPO, f"feathers_{args.dataset}_{tag}", args.pulsar)
    std = build_checked(refit_par, cut_tim, out)
    print(f"[mjdcut] done -> {out} (residual std {std:.3f} us)", flush=True)


if __name__ == "__main__":
    main()
