#!/usr/bin/env python3
"""
Build ONE discovery-compatible feather from a par+tim pair, in this process.
Called as an isolated subprocess by make_feathers_matched.py and
make_feather_mjdcut.py (--build-one), never for several pulsars at once.

Runs inside an environment with enterprise + libstempo (e.g. igwn-py38-feathers).
No discovery/jax import needed here: enterprise.pulsar.Pulsar already ships a
native Tempo2Pulsar.to_feather() (-> FeatherPulsar.save_feather), and its output
schema (columns/vector_columns/tensor_columns/metadata) is a strict superset of
discovery.Pulsar's (extra "telescope" column, extra "_pdist" metadata key) --
discovery.Pulsar.read_feather only ever reads the keys IT knows about, so it
reads these files with no changes needed on either side.

One Pulsar() construction per process
--------------------------------------
tempo2/libstempo's C extension keeps a fair amount of global/static state, and
building more than one Tempo2Pulsar in the same Python process is fragile:
across repeated runs with no code changes, this showed up as a hard segfault
on the SECOND Pulsar() construction one run, and as silently corrupted
residuals (a wrong but plausible-looking std, no crash, no error) on another
-- reproduced with byte-identical input. Callers therefore spawn one
subprocess per feather, and accept a build only when two independent builds
agree (see make_feathers_matched.py's spawn_build_checked).

No noisedict is baked in by default (white noise is left to the sampler);
pass --noisedict a.json to store fixed values instead.

Usage
-----
    conda run -n igwn-py38-feathers python make_feathers.py --build-one PAR TIM OUTPATH
"""
import argparse
import json
import os

import numpy as np


def build_pulsar(parfile, timfile, ephem=None, clk=None):
    """Load a tempo2 pulsar through enterprise. ephem/clk=None -> respect the
    par file."""
    from enterprise.pulsar import Pulsar

    kwargs = dict(timing_package="tempo2", drop_t2pulsar=True, planets=True)
    if ephem:
        kwargs["ephem"] = ephem
    if clk:
        kwargs["clk"] = clk
    return Pulsar(parfile, timfile, **kwargs)


def build_one(parfile, timfile, outpath, noisedict, ephem, clk):
    print(f"[make_feathers]   {os.path.basename(outpath)} <- "
          f"{os.path.basename(parfile)} + {os.path.basename(timfile)} ...", flush=True)
    psr = build_pulsar(parfile, timfile, ephem, clk)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    psr.to_feather(outpath, noisedict=noisedict)
    print(f"[make_feathers]   -> {outpath}  "
          f"({len(psr.toas)} TOAs, Mmat {psr.Mmat.shape}, "
          f"res std={np.std(psr.residuals * 1e6):.3f} us)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-one", nargs=3, required=True, metavar=("PARFILE", "TIMFILE", "OUTPATH"))
    ap.add_argument("--ephem", default=None, help="ephemeris override (default: par's EPHEM)")
    ap.add_argument("--clk", default=None, help="clock chain override (default: par's CLK)")
    ap.add_argument("--noisedict", default=None, help="JSON noise dict to store in the feather")
    args = ap.parse_args()

    noisedict = None
    if args.noisedict:
        with open(args.noisedict) as fh:
            noisedict = json.load(fh)
    build_one(*args.build_one, noisedict, args.ephem, args.clk)


if __name__ == "__main__":
    main()
