#!/usr/bin/env python3
"""
Refit every fit=1 parameter of a par+tim pair via tempo2's classical
iterative GLS fit (libstempo), saving the result to a new par file.

Used as the "get a proper linearization point for THIS specific dataset"
step before building a feather from a Tspan/station-restricted tim: the
par's stored values only need to be close enough to the true local optimum
for the noise analysis's own Mmat-based marginalization to be valid (it
refits everything fit=1 again anyway, jointly with the noise parameters) --
but starting from a value fit on a very different dataset (a much longer
baseline, or a different station mix) is an avoidable source of residual
linearization error, and refitting first removes it for free.

One tempopulsar() construction per process on purpose: building more than
one Tempo2Pulsar in the same process has proven fragile elsewhere in this
project (see make_feathers.py's docstring) -- a segfault once, silently
corrupted residuals with no error another time, both with byte-identical
input. Not needed here since this script only ever builds one, but keeping
it a separate process from whatever calls it (as make_feathers_matched.py
does, via subprocess.run) preserves that isolation.

Usage: python refit_par.py PARFILE TIMFILE OUT_PARFILE [MAXOBS]
"""
import sys

import libstempo as LT

parfile, timfile, outpar = sys.argv[1:4]
maxobs = int(sys.argv[4]) if len(sys.argv) > 4 else 40000

psr = LT.tempopulsar(parfile, timfile, maxobs=maxobs)
n_free = len(psr.pars(which="fit"))
psr.fit(iters=3)
psr.savepar(outpar)
print(f"[refit_par] {outpar} <- refit of {parfile} against {timfile} "
      f"({n_free} free parameters, {len(psr.toas())} TOAs)", flush=True)
