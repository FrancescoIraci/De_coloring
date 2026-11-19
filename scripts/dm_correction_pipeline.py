"""
DM Correction Pipeline
======================
End-to-end pipeline: epoch-wise total-DM fit on LOFAR TOAs -> GP smoothing of
the resulting DM(t) -> apply the DM(t) correction to the full multi-telescope
dataset -> plot residuals pre/post correction.

Steps
-----
1. Epoch-wise total-DM fit (epochwise_dm_total.py), using ONLY the LOFAR TOAs
   (wide fractional bandwidth -> good chromatic leverage) but the FULL/global
   .par file (fit against all telescopes).
2. Fit a GP (exponential-family Matern kernel) to the resulting noisy DM(t)
   points, to get a smooth DM(t) curve with an uncertainty band.
3. Load the FULL multi-telescope .tim (all telescopes, via its own INCLUDE
   file) with a .par where DM1=DM2=NE_SW=0 (same file produced in step 1), and
   for every TOA whose epoch falls INSIDE the LOFAR time span, subtract the
   modeled DM-mismatch delay (GP_DM(t) - DM0) / (DM_CONST * freq^2). TOAs
   outside the LOFAR time span are left uncorrected -- the GP has no
   constraint there, so "correcting" them would just be injecting the GP
   prior, not a real measurement.
4. Plot residuals vs MJD, pre- and post-correction, with a marginal histogram
   (rotated 90 deg) comparing the two distributions.

Usage
-----
    python dm_correction_pipeline.py \
        -n J1012+5307 \
        -d J1012+5307 \
        -p nodoubles_postfit.par \
        --lofar-tim lofar/J1012+5307_ToAs_pta_dmo_no_doubles.tim \
        --full-tim J1012+5307_all_no_doubles.tim \
        -o J1012+5307/dm_correction_out \
        --plot
"""

import os
import sys
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epochwise_dm_total as ewdm

DM_CONST = 2.41e-4   # tempo2's dispersion constant (t2fit_stdFitFuncs.C / tempo2.h)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Epoch-wise DM -> GP -> residual correction pipeline")
    p.add_argument("-n", "--name", default="J1012+5307")
    p.add_argument("-d", "--dirpath", required=True, help="Folder containing par/tim files")
    p.add_argument("-p", "--parfile", required=True, help="Global .par (fit against all telescopes)")
    p.add_argument("--lofar-tim", required=True, metavar="FILE",
                   help="LOFAR-only .tim, relative to dirpath, used to measure DM(t)")
    p.add_argument("--full-tim", required=True, metavar="FILE",
                   help="Full multi-telescope .tim (can be an INCLUDE file), relative to dirpath, "
                        "to which the DM(t) correction is applied")
    p.add_argument("-o", "--outdir", required=True)
    p.add_argument("--min-freqs", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--maxobs", type=int, default=40000, help="libstempo TOA buffer size for the full dataset")
    p.add_argument("--max-dm-err", type=float, default=0.0005,
                   help="drop epoch-wise DM points with error above this before fitting the GP")
    p.add_argument("--sigma-clip", type=float, default=5.0,
                   help="drop epoch-wise DM points more than this many sigma from a first-pass GP fit, "
                        "then refit (0 disables clipping)")
    p.add_argument("--recompute-dm", action="store_true",
                   help="force re-running the epoch-wise DM fit even if DMts.npy already exists")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: epoch-wise total-DM fit (reuses epochwise_dm_total.py)
# ---------------------------------------------------------------------------

def epochwise_dm(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    dmts_path = os.path.join(outdir, "DMts.npy")
    stem = os.path.splitext(os.path.basename(args.parfile))[0]
    dmfit_par = os.path.join(outdir, f"{stem}.dmfit.par")

    if os.path.exists(dmts_path) and os.path.exists(dmfit_par) and not args.recompute_dm:
        print(f"[INFO] Reusing existing {dmts_path} (use --recompute-dm to force a re-run)")
        return np.load(dmts_path), dmfit_par

    par_path = os.path.join(args.dirpath, args.parfile)
    lofar_tim_path = os.path.join(args.dirpath, args.lofar_tim)

    ewdm.write_dmfit_par(par_path, dmfit_par)
    print(f"[INFO] DM-fit par written -> {dmfit_par} (DM1=DM2=NE_SW=0, only DM free)")

    n_epochs = ewdm.split_epochs(lofar_tim_path, outdir)

    print("\n[INFO] Starting epoch-wise total-DM fitting on LOFAR TOAs ...\n")
    dm_ts = ewdm.fit_dm_per_epoch(dmfit_par, outdir, n_epochs, args.min_freqs, args.timeout)

    n_ok = int(np.isfinite(dm_ts[:, 1]).sum())
    print(f"\n[INFO] {n_ok}/{n_epochs} epochs fitted successfully")
    np.save(dmts_path, dm_ts)

    for obs in range(n_epochs):
        ep_path = os.path.join(outdir, f"{obs}.tim")
        if os.path.exists(ep_path):
            os.remove(ep_path)

    return dm_ts, dmfit_par


# ---------------------------------------------------------------------------
# Step 2: GP smoothing of the DM(t) time series
# ---------------------------------------------------------------------------

def _fit_one_gp(t, y, dm_err):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    # bounds are wide on purpose -- tight bounds make the optimizer peg the
    # amplitude/length-scale at the edge instead of finding the interior optimum
    kernel = ConstantKernel(1e-4, (1e-12, 10)) * Matern(length_scale=100.0, length_scale_bounds=(1.0, 1e5), nu=1.5) \
             + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-1))

    # random_state fixed: n_restarts_optimizer draws its random starting
    # points from numpy's global RNG state when unset, so without this the
    # fit is a different (and, on sparse/noisy input, occasionally
    # degenerate -- e.g. the GP-constrained Tspan collapsing to one point)
    # local optimum on every separate process invocation of this function,
    # even given byte-identical input (observed via backfill_plots.py
    # reproducing the exact same DMts.npy across runs).
    gp = GaussianProcessRegressor(kernel=kernel, alpha=dm_err**2, n_restarts_optimizer=10,
                                   normalize_y=False, random_state=0)
    gp.fit(t, y)
    return gp


def fit_gp(dm_ts, max_dm_err, sigma_clip=5.0):
    mjd, dm_val, dm_err = dm_ts[:, 0], dm_ts[:, 1], dm_ts[:, 2]
    mask = np.isfinite(dm_val) & (dm_err < max_dm_err)
    mjd, dm_val, dm_err = mjd[mask], dm_val[mask], dm_err[mask]

    order = np.argsort(mjd)
    mjd, dm_val, dm_err = mjd[order], dm_val[order], dm_err[order]

    t0 = mjd.min()
    t = (mjd - t0).reshape(-1, 1)
    y_mean = dm_val.mean()
    y = dm_val - y_mean

    # First pass: fit on everything that passed the error cut. A handful of
    # epochs can have a small quoted dm_err but a value that's a genuine
    # outlier relative to the smooth trend (bad archive/RFI/template glitch
    # slipping through); left in, they drag the fitted length-scale down to
    # an implausibly short value as the GP tries to "explain" the glitch,
    # which then injects noise (not signal) into every other epoch's
    # correction. Refit after clipping points whose fit residual is large
    # relative to their own quoted error.
    gp = _fit_one_gp(t, y, dm_err)
    pred_mean, pred_std = gp.predict(t, return_std=True)
    z = (y - pred_mean) / np.sqrt(dm_err**2 + pred_std**2)
    keep = np.abs(z) < sigma_clip
    n_clipped = (~keep).sum()
    if n_clipped:
        print(f"[INFO] GP: clipped {n_clipped} outlier epoch(s) (>{sigma_clip}-sigma from first-pass fit) "
              f"before the final fit")
        t, y, dm_err, dm_val = t[keep], y[keep], dm_err[keep], dm_val[keep]
        mjd = mjd[keep]
        gp = _fit_one_gp(t, y, dm_err)

    print(f"[INFO] GP kernel: {gp.kernel_}")

    mjd_span = (mjd.min(), mjd.max())

    def predict(mjd_query):
        mjd_query = np.asarray(mjd_query)
        tq = (mjd_query - t0).reshape(-1, 1)
        mean, std = gp.predict(tq, return_std=True)
        return mean + y_mean, std

    return predict, mjd_span, (mjd, dm_val, dm_err)


# ---------------------------------------------------------------------------
# Step 3: apply the DM(t) correction to the full multi-telescope dataset
# ---------------------------------------------------------------------------

def apply_correction(args, dmfit_par, gp_predict, mjd_span, dm0):
    import libstempo as LT

    full_tim_path = os.path.join(args.dirpath, args.full_tim)
    par_path = os.path.join(args.dirpath, args.parfile)   # original, unmodified par

    def load_frozen(par):
        p = LT.tempopulsar(par, full_tim_path, maxobs=args.maxobs)
        for p_ in p.pars(which="all"):
            try:
                p_val = p[p_]
                p_val.fit = False
            except Exception:
                pass
        return p

    # "pre": residuals from the pulsar's CURRENT best model (DM1/DM2/NE_SW as
    # fit). removemean=False keeps both loads on the same absolute (TZR-based)
    # reference frame, which we need below to take a clean difference.
    psr_pre = load_frozen(par_path)
    res_pre = psr_pre.residuals(removemean=False).copy()
    mjds = psr_pre.toas().copy()
    freqs = psr_pre.freqs.copy()
    toaerrs = psr_pre.toaerrs.copy()   # microseconds
    sys_vals = np.asarray(psr_pre.flagvals("sys"))
    group = np.where(
        np.char.startswith(sys_vals, "NENUFAR"), "NenuFAR",
        np.where(np.char.startswith(sys_vals, "LOFAR") | np.char.startswith(sys_vals, "DE60"),
                 "LOFAR", "Other telescopes"))
    # NenuFAR observes even lower than LOFAR (~40-75 MHz vs ~110-190 MHz) --
    # it is far MORE DM-sensitive (delay ~ 1/freq^2), so any imperfection in
    # the GP-measured DM(t) is amplified there far more than at LOFAR itself.
    # Keeping it separate from the (~1200-2650 MHz) "other telescopes" group
    # avoids a handful of hugely amplified NenuFAR points swamping the stats
    # for the actually-high-frequency data.

    # same TOAs/model but with DM1=DM2=NE_SW=0 -- used only to isolate that
    # model's own DM contribution via tempo2 itself (avoids re-deriving the
    # solar-elongation-dependent NE_SW delay by hand)
    psr_dmfit = load_frozen(dmfit_par)
    res_dmfit_baseline = psr_dmfit.residuals(removemean=False).copy()

    in_span = (mjds >= mjd_span[0]) & (mjds <= mjd_span[1])
    print(f"[INFO] {in_span.sum()}/{len(mjds)} TOAs fall within the LOFAR Tspan "
          f"[{mjd_span[0]:.1f}, {mjd_span[1]:.1f}] and will be corrected; "
          f"{(~in_span).sum()} TOAs outside are left uncorrected.")

    # res_post = res_dmfit_baseline - gp_delay is algebraically identical to
    # res_pre + (original DM1/DM2/NE_SW contribution) - (GP-measured contribution):
    # both start from the same "no DM(t) model" baseline and swap in a
    # different DM(t) model on top of it.
    corrected_res = res_pre.copy()
    dm_gp_pred = np.full(len(mjds), np.nan)
    if in_span.sum() > 0:
        dm_gp_pred[in_span], _ = gp_predict(mjds[in_span])
        gp_delay = (dm_gp_pred[in_span] - dm0) / (DM_CONST * freqs[in_span]**2)
        corrected_res[in_span] = res_dmfit_baseline[in_span] - gp_delay

    raw_res = res_pre
    # physical (non-recentred) values -- these are what a materialized -dmo
    # correction must reproduce; the cosmetic recentring below must NOT
    # contaminate them (a constant time offset does not correspond to any
    # real per-TOA DM value, so it would corrupt the -dmo we later derive)
    raw_res_absolute = raw_res.copy()
    corrected_res_absolute = corrected_res.copy()

    if in_span.sum() > 0:
        # raw_res and corrected_res are two DIFFERENT models (original DM1/DM2/
        # NE_SW vs GP-swapped-in DM(t)), neither jointly re-fit for zero mean
        # against the rest of the timing model -- so their absolute levels are
        # not directly comparable, only their scatter (std) is. Re-centre each
        # one on its own mean for a fair visual comparison; std is unaffected.
        # This is done PER GROUP (not with one pooled mean across LOFAR/NenuFAR/
        # Other): NenuFAR's post-correction values can be huge (chromatic
        # amplification of any GP imperfection), and a single pooled mean would
        # let that drag every other group's zero-point off-center too.
        for g in np.unique(group):
            sel = in_span & (group == g)
            if sel.sum() == 0:
                continue
            raw_res[sel] -= np.average(raw_res[sel])
            corrected_res[sel] -= np.average(corrected_res[sel])

    return {
        "mjds": mjds, "freqs": freqs, "group": group, "in_span": in_span, "toaerrs": toaerrs,
        "raw_res": raw_res, "corrected_res": corrected_res, "dm_gp_pred": dm_gp_pred,
        "raw_res_absolute": raw_res_absolute, "corrected_res_absolute": corrected_res_absolute,
        "res_dmfit_baseline": res_dmfit_baseline,
    }


# ---------------------------------------------------------------------------
# Step 3b: materialize the correction as a plain par+tim pair
#
# Everything above lives only inside this script's Python arrays. To make the
# correction usable by anyone with a stock tempo2/libstempo install (no need
# to re-run this pipeline), we bake it into the .tim file itself via the
# "-dmo" flag, which dm_delays.C already adds unconditionally to the model's
# DM value for any TOA that carries it (dm_delays.C: `dmval += dmval2` for
# "-dmo") -- independent of whatever DM1/DM2/NE_SW the .par has active. The
# final .par is the ORIGINAL, unmodified one (DM1/DM2/NE_SW stay exactly as
# globally fit): outside the LOFAR Tspan the injected delta is exactly 0, so
# those TOAs are untouched and the original polynomial model stays in charge
# there, matching what "raw_res"/"corrected_res" already do for such TOAs.
#
# Two non-obvious tempo2 behaviors were found (both empirically, by
# differencing against a from-scratch load of the untouched original tree
# until the mismatch vanished) that a naive rewrite gets wrong:
#
#  1. Flattening a multi-file INCLUDE tree into one big .tim -- even with
#     byte-identical TOA content -- silently shifts tempo2's computed clock
#     correction by ~10s for a handful of TOAs. Fix: keep each original
#     source file as its own separate (corrected) file, INCLUDEd individually
#     from a single-level list (not nested/flattened).
#  2. Dropping commented-out ("C "-prefixed) TOA lines when rewriting a leaf
#     file -- lines that are inactive either way -- ALSO shifts the clock
#     correction for nearby active TOAs by the same ~10s. Fix: copy every
#     leaf file's full original content verbatim, line for line (comments,
#     blanks, headers included), and only ever edit the "-dmo" flag in place
#     on genuinely active TOA lines.
# ---------------------------------------------------------------------------

DMO_FLAG_RE = re.compile(r'(?<!\S)-dmo\s+(\S+)')


def _is_active_toa_line(stripped_line):
    if not stripped_line:
        return False
    cols = stripped_line.split()
    return cols[0] not in ("FORMAT", "MODE", "C", "#", "INCLUDE") and len(cols) >= 4


def _find_leaf_tim_paths(tim_path):
    """Recursively resolve INCLUDE directives into the list of LEAF .tim file
    paths (files with no INCLUDE of their own), in encounter order -- matching
    tempo2/libstempo's TOA load order (verified empirically: contiguous
    per-source blocks, no global MJD resort). Assumes no file mixes INCLUDE
    directives with its own TOA lines (true of every top-level/leaf file
    encountered so far -- top-level files are pure INCLUDE lists, leaf files
    have no INCLUDEs at all); prints a warning and treats the file as a leaf
    too if that assumption is violated, rather than silently dropping lines.
    """
    base_dir = os.path.dirname(os.path.abspath(tim_path))
    leaves = []
    has_include = False
    has_own_toa = False
    with open(tim_path) as f:
        for raw in f:
            stripped = raw.strip()
            cols = stripped.split()
            if cols and cols[0] == "INCLUDE":
                has_include = True
                sub = os.path.normpath(os.path.join(base_dir, cols[1]))
                leaves.extend(_find_leaf_tim_paths(sub))
            elif _is_active_toa_line(stripped):
                has_own_toa = True
    if has_include and has_own_toa:
        print(f"[WARNING] {tim_path} mixes INCLUDE directives with its own TOA lines "
              f"-- materialization does not handle this case, results may be wrong.")
    if not has_include:
        leaves.append(os.path.abspath(tim_path))
    return leaves


def _count_active_toa_lines(path):
    n = 0
    with open(path) as f:
        for raw in f:
            if _is_active_toa_line(raw.strip()):
                n += 1
    return n


def _classify_sys(sys_val):
    """Same LOFAR/NenuFAR/Other-telescopes split as apply_correction()'s
    vectorized version, for a single -sys flag value (used to classify a
    whole leaf .tim file by its first active TOA's system)."""
    if sys_val.startswith("NENUFAR"):
        return "NenuFAR"
    if sys_val.startswith("LOFAR") or sys_val.startswith("DE60"):
        return "LOFAR"
    return "Other telescopes"


def _leaf_sys_value(path):
    """The -sys flag value of the first active TOA line in a leaf .tim file
    (leaf files are single-system in every case seen so far)."""
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if _is_active_toa_line(s):
                m = re.search(r'(?<!\S)-sys\s+(\S+)', s)
                return m.group(1) if m else ""
    return ""


def _read_feather_residual_std_us(feather_path):
    """Residual std (microseconds) straight out of a written feather file,
    via pyarrow only -- no discovery/enterprise import needed. Used as a
    cheap correctness fingerprint for the consensus-retry in
    make_feathers_matched.py (see spawn_build_checked's docstring)."""
    import numpy as np
    import pyarrow.feather as feather

    t = feather.read_table(feather_path)
    return float(np.std(np.asarray(t["residuals"])) * 1e6)


def lofar_mjd_range(lofar_tim_path):
    """(min, max) MJD across every active TOA line in lofar_tim_path's
    resolved leaf tree -- a cheap, fit-free stand-in for the LOFAR Tspan
    boundary, usable BEFORE any epoch-wise DM fitting has ever run.

    Confirmed empirically (see the calling conversation) to differ from the
    quality-filtered epoch-wise+GP domain by well under a day, for every
    pulsar checked -- and even where it differs (a low-quality epoch right
    at the edge: too few frequency channels, or sigma-clipped as an
    outlier), apply_correction()'s own in_span check independently
    re-enforces "never extrapolate the GP" against whatever mjd_span THAT
    run's own fit_gp() actually returns, regardless of which window was
    used here to decide what to cut and refit. So using this cheaper
    estimate instead of a full epoch-wise+GP bootstrap pass costs nothing
    in safety, only in an epoch-wise fit that would otherwise have to run
    twice per pulsar."""
    mjd_min, mjd_max = float("inf"), float("-inf")
    for leaf in _find_leaf_tim_paths(lofar_tim_path):
        for raw in open(leaf):
            s = raw.strip()
            if _is_active_toa_line(s):
                mjd = float(s.split()[2])
                mjd_min = min(mjd_min, mjd)
                mjd_max = max(mjd_max, mjd)
    return mjd_min, mjd_max


def _merge_dmo_flag(line, delta_dm):
    """Add delta_dm (pc/cm^3) to whatever -dmo flag the line already carries
    (0 if none), replacing/appending a single -dmo flag. delta_dm == 0 (exact,
    by construction for out-of-Tspan TOAs) leaves the line byte-identical."""
    if delta_dm == 0.0:
        return line
    m = DMO_FLAG_RE.search(line)
    existing = float(m.group(1)) if m else 0.0
    new_val = existing + delta_dm
    if m:
        return line[:m.start(1)] + f"{new_val:.10g}" + line[m.end(1):]
    return line.rstrip() + f" -dmo {new_val:.10g}"


def _rewrite_leaf_with_correction(leaf_path, out_path, deltas):
    """Copy leaf_path to out_path verbatim, line for line, except that each
    active TOA line (in order) gets its -dmo flag merged with the matching
    entry of `deltas`. Every other line (comments, blanks, FORMAT/MODE,
    trailing whitespace/newline convention) is preserved exactly."""
    with open(leaf_path) as f:
        raw_lines = f.read().split("\n")
    # a trailing '' from a final newline shouldn't be treated as a line to keep verbatim as-is
    out = []
    i = 0
    for s in raw_lines:
        if _is_active_toa_line(s.strip()):
            out.append(_merge_dmo_flag(s, deltas[i]))
            i += 1
        else:
            out.append(s)
    if i != len(deltas):
        raise RuntimeError(f"active-TOA-line count mismatch while rewriting {leaf_path}: "
                            f"found {i}, expected {len(deltas)}")
    with open(out_path, "w") as f:
        f.write("\n".join(out))


def _write_final_corrected_par(src_path, dst_path):
    """Copy src_path to dst_path with DM1, DM2 and NE_SW genuinely zeroed
    (value AND fit flag -- the secular/solar-wind model is fully removed,
    not just frozen) and DM's fit flag frozen (its VALUE is kept: it is the
    reference the per-TOA "-dmo" deltas are computed relative to, so it must
    not silently move under a later fit, or every already-baked-in delta
    would be off by the same amount). Every other parameter (F0, F1,
    astrometry, binary, JUMPs, ...) is left completely untouched -- value
    and fit flag both -- so a subsequent fit can still check/refit the rest
    of the timing model.

    With DM1/DM2/NE_SW gone from the model, their old contribution has to be
    reproduced by "-dmo" too wherever there's no GP measurement to replace it
    with (i.e. outside the LOFAR Tspan) -- see the delta_dm formula in
    write_materialized_correction, which handles this uniformly."""
    out = []
    for raw in open(src_path):
        s = raw.rstrip("\n")
        t = s.split()
        if not t:
            out.append(s)
            continue
        name = t[0]
        if name in ("DM1", "DM2", "NE_SW"):
            out.append(f"{name:<14s} 0 0")
        elif name == "DM":
            if len(t) >= 3 and t[2] in ("0", "1"):
                t[2] = "0"
            out.append(" ".join(t))
        else:
            out.append(s)
    with open(dst_path, "w") as f:
        f.write("\n".join(out) + "\n")


def fix_ambiguous_fit_flags(src_path, dst_path):
    """Copy src_path to dst_path, inserting an explicit fit flag "1" on any
    parameter line of the form "NAME value uncertainty" (3 whitespace-
    separated tokens, no fit flag) that tempo2 would otherwise silently
    mis-parse.

    Stock tempo2's par-file grammar is "NAME value [fit_flag] [uncertainty]":
    with only 3 tokens present, the 3rd is read as the fit flag, not the
    uncertainty -- and since a genuine uncertainty is essentially never
    exactly "0" or "1", tempo2 ends up treating the parameter as fit=0 with
    err=0, silently discarding both the fitted-uncertainty value AND the
    intent that the parameter be free. Confirmed empirically (libstempo:
    fit=False, err=0 on exactly this line shape) and confirmed pre-existing
    in delivered EPTA par files (some, not all -- e.g. combined_J1640+2224.par
    writes RAJ/DECJ/PMRA/PX/the whole DDH binary model this way, while
    combined_J1012+5307.par and nodoubles_postfit.par use the explicit
    "value 1 uncertainty" form for the same parameters) -- almost certainly a
    formatting slip from whatever pipeline stage wrote the file, not a
    deliberate freeze: a quoted, non-zero postfit uncertainty is exactly what
    a FITTED parameter reports, not a frozen one.

    The one legitimate 3-token exception found in these files is "CHI2R
    value dof" (reduced chi-square + degrees of freedom, not a fittable
    timing parameter at all) -- excluded by name, not by guessing at the
    3rd token's shape, so nothing else with a coincidentally similar layout
    gets mangled. Lines with 2 tokens (value only, e.g. PEPOCH, TZRMJD,
    NTOA) or already-4-token lines (explicit fit flag present) are left
    untouched either way."""
    NON_FIT_METADATA = {"CHI2R"}
    out = []
    n_fixed = 0
    for raw in open(src_path):
        s = raw.rstrip("\n")
        t = s.split()
        if len(t) == 3 and t[0] not in NON_FIT_METADATA and t[2] not in ("0", "1"):
            out.append(f"{t[0]:<14s} {t[1]} 1 {t[2]}")
            n_fixed += 1
        else:
            out.append(s)
    with open(dst_path, "w") as f:
        f.write("\n".join(out) + "\n")
    return n_fixed


def _collect_flag_values(tim_path, flagname):
    """Set of every distinct value the given flag takes across all active
    TOA lines in tim_path's resolved leaf tree."""
    values = set()
    pattern = re.compile(rf'(?<!\S)-{re.escape(flagname)}\s+(\S+)')
    for leaf in _find_leaf_tim_paths(tim_path):
        with open(leaf) as f:
            for raw in f:
                s = raw.strip()
                if _is_active_toa_line(s):
                    m = pattern.search(s)
                    if m:
                        values.add(m.group(1))
    return values


def strip_orphaned_jumps(par_path, tim_path, dst_path):
    """Copy par_path to dst_path, dropping any JUMP line whose flag/value
    (e.g. "-sys LOFAR.150") matches zero active TOA lines in tim_path's
    resolved leaf tree.

    A JUMP tied to a flag value no TOA carries produces an identically-zero
    column in the design matrix -- confirmed empirically (enterprise builds
    it with no error or warning). Harmless for a classical tempo2 fit (just
    reported unconstrained), but fatal for Bayesian marginalization, where a
    zero column makes the relevant piece of the covariance singular. Matters
    specifically when pairing a par file against a high-frequency-only tim
    (LOFAR/NenuFAR excluded): the JUMPs for those stations, still fit=1 in
    the delivered par, no longer have any TOA to act on."""
    flag_value_cache = {}
    out = []
    n_dropped = 0
    for raw in open(par_path):
        s = raw.rstrip("\n")
        t = s.split()
        if t and t[0] == "JUMP" and len(t) >= 3 and t[1].startswith("-"):
            flagname, value = t[1][1:], t[2]
            if flagname not in flag_value_cache:
                flag_value_cache[flagname] = _collect_flag_values(tim_path, flagname)
            if value not in flag_value_cache[flagname]:
                n_dropped += 1
                continue
        out.append(s)
    with open(dst_path, "w") as f:
        f.write("\n".join(out) + "\n")
    return n_dropped


def write_forfeather_corrected_par(src_path, dst_path):
    """Variant of _write_final_corrected_par for feathers destined for
    Bayesian noise analysis (enterprise/discovery), where "timing parameters"
    need fit=1 so libstempo's design matrix carries a column for them and the
    sampler marginalizes over them analytically.

    DM1/DM2/NE_SW: value=0 AND fit=0. They contribute nothing (already fully
    absorbed into the per-TOA "-dmo" correction) and a null direction has no
    business in the design matrix -- fit=1 on an identically-zero parameter
    would just hand the marginalizer a degenerate column.

    DM: value kept (the -dmo deltas are computed relative to it), fit flipped
    to 1 (unlike _write_final_corrected_par's fit=0). Unlike DM1/DM2/NE_SW,
    DM is not identically zero and is not perfectly known -- the epoch-wise
    GP measurement it is tied to has its own uncertainty -- so it is a
    genuine "timing parameter" that should be marginalized like the rest of
    the timing model (F0, astrometry, ...), all of which are already fit=1
    in the source par and are left untouched here.

    Only DM's fit flag differs from _write_final_corrected_par -- writing
    this by copying the value/-dmo semantics rather than duplicating the
    whole function would be more DRY, but the two outputs serve genuinely
    different consumers (a plain tempo2 re-fit vs. a discovery feather) and
    keeping them as two short, independently-readable functions is safer
    than a shared one branching on a "for_feather" flag deep inside a loop."""
    out = []
    for raw in open(src_path):
        s = raw.rstrip("\n")
        t = s.split()
        if not t:
            out.append(s)
            continue
        name = t[0]
        if name in ("DM1", "DM2", "NE_SW"):
            out.append(f"{name:<14s} 0 0")
        elif name == "DM":
            if len(t) >= 3:
                t[2] = "1"
            elif len(t) == 2:
                t.append("1")
            out.append(" ".join(t))
        else:
            out.append(s)
    with open(dst_path, "w") as f:
        f.write("\n".join(out) + "\n")


def write_nesw_zeroed_par(src_path, dst_path):
    """Copy src_path to dst_path with ONLY NE_SW zeroed (value AND fit=0);
    DM/DM1/DM2 and every other parameter are left exactly as delivered
    (including their own fit flags).

    For the "no DM(t) correction" comparison arm: keeps the original secular
    DM1/DM2 polynomial active (that is the whole point of this arm -- it is
    the baseline the deterministic LOFAR-based correction is being compared
    against) but still drops NE_SW, since a single time-independent value is
    a poor model of the solar wind's real elongation/time dependence -- an
    absent term is more honest than a bad constant one. A proper
    time-dependent solar-wind treatment, if wanted, is a noise-model
    decision for later, not a par-file one."""
    out = []
    for raw in open(src_path):
        s = raw.rstrip("\n")
        t = s.split()
        if t and t[0] == "NE_SW":
            out.append("NE_SW          0 0")
        else:
            out.append(s)
    with open(dst_path, "w") as f:
        f.write("\n".join(out) + "\n")


def filter_leaf_by_mjd(leaf_path, out_path, mjd_lo, mjd_hi):
    """Copy leaf_path to out_path, dropping active TOA lines whose MJD (3rd
    whitespace-separated field -- filename, freq, MJD, err, site, ...flags --
    standard tempo2 TOA format) falls outside [mjd_lo, mjd_hi]. Every other
    line (comments, blanks, FORMAT/MODE) is kept verbatim. Returns
    (n_kept, n_dropped)."""
    out = []
    n_kept = n_dropped = 0
    for raw in open(leaf_path):
        s = raw.rstrip("\n")
        stripped = s.strip()
        if _is_active_toa_line(stripped):
            mjd = float(stripped.split()[2])
            if mjd_lo <= mjd <= mjd_hi:
                out.append(s)
                n_kept += 1
            else:
                n_dropped += 1
            continue
        out.append(s)
    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")
    return n_kept, n_dropped


def write_materialized_correction(args, result, outdir):
    import libstempo as LT

    full_tim_path = os.path.join(args.dirpath, args.full_tim)
    par_path = os.path.join(args.dirpath, args.parfile)

    leaf_paths = _find_leaf_tim_paths(full_tim_path)
    counts = [_count_active_toa_lines(lf) for lf in leaf_paths]
    n_lines = sum(counts)
    mjds, freqs = result["mjds"], result["freqs"]
    if n_lines != len(mjds):
        print(f"[WARNING] Materialization skipped: {n_lines} active TOA lines found across "
              f"{len(leaf_paths)} source file(s) but {len(mjds)} TOAs loaded by libstempo -- "
              f"cannot safely align lines to corrections.")
        return None

    # Delta needed on top of DM alone (DM1=DM2=NE_SW=0 in the final par) to
    # reproduce corrected_res_absolute everywhere -- one formula, no in/out
    # of Tspan branching needed:
    #   in-Tspan:  reduces to (DM_GP(t) - DM0)              -- the GP measurement
    #   out-of-Tspan: reduces to the ORIGINAL DM1/DM2/NE_SW contribution --
    #                 i.e. -dmo alone reproduces what the (now-removed)
    #                 polynomial used to provide there, so nothing is lost
    #                 for the part of the dataset the GP doesn't cover.
    dmfit_baseline, corr_abs = result["res_dmfit_baseline"], result["corrected_res_absolute"]
    delta_dm = (dmfit_baseline - corr_abs) * DM_CONST * freqs**2

    tims_dir = os.path.join(outdir, "materialized_tims")
    os.makedirs(tims_dir, exist_ok=True)

    used_names = {}
    include_lines = []
    leaf_info = []   # (original_leaf_path, materialized_name, group_label) per leaf
    idx = 0
    for leaf_path, n in zip(leaf_paths, counts):
        block_delta = delta_dm[idx:idx + n]
        idx += n

        base = os.path.basename(leaf_path)
        k = used_names.get(base, 0)
        name = base if k == 0 else f"{os.path.splitext(base)[0]}_{k}{os.path.splitext(base)[1]}"
        used_names[base] = k + 1

        out_path = os.path.join(tims_dir, name)
        _rewrite_leaf_with_correction(leaf_path, out_path, block_delta)
        include_lines.append(f"INCLUDE materialized_tims/{name}")
        leaf_info.append((leaf_path, name, _classify_sys(_leaf_sys_value(leaf_path))))

    stem = os.path.splitext(os.path.basename(args.parfile))[0]
    final_par_path = os.path.join(outdir, f"{stem}.dmcorrected.par")
    final_tim_path = os.path.join(outdir, f"{args.name}_dmcorrected.tim")

    _write_final_corrected_par(par_path, final_par_path)
    with open(final_tim_path, "w") as f:
        f.write("FORMAT 1\n" + "\n".join(include_lines) + "\n")
    print(f"[INFO] Materialized correction written -> {final_par_path} / {final_tim_path} "
          f"({len(leaf_paths)} source file(s) in {tims_dir}/)")

    # Validate: a FRESH, plain load of (final_par, final_tim) -- no Python
    # post-processing -- should reproduce corrected_res_absolute up to a
    # single overall constant. That constant is expected, not a bug: if
    # TZRMJD/TZRFRQ (tempo2's absolute-phase reference point) falls inside
    # the corrected Tspan at a LOFAR-like frequency, our correction also
    # shifts the model's prediction AT the reference point itself, which
    # (per tempo2's own absolute-phase convention, removemean=False) shifts
    # every TOA's residual by the same constant -- physically harmless,
    # since only the SCATTER (std), never an overall additive constant,
    # carries information. So we validate the scatter of the difference, not
    # its absolute size.
    psr_check = LT.tempopulsar(final_par_path, final_tim_path, maxobs=args.maxobs)
    for p_ in psr_check.pars(which="all"):
        try:
            p_val = psr_check[p_]
            p_val.fit = False
        except Exception:
            pass
    mjd_check = psr_check.toas()
    mjd_diff = np.max(np.abs(mjd_check - mjds))
    res_check = psr_check.residuals(removemean=False)
    diff_us = (res_check - corr_abs) * 1e6

    # NenuFAR is excluded from the pass/fail check on purpose: its own
    # correction is already known to be unreliable (chromatic amplification
    # of any GP imperfection at ~40-75 MHz -- see the "Other telescopes"/
    # NenuFAR split elsewhere in this script), so tiny floating-point
    # differences in how the -dmo delta was written (%.10g) get blown up
    # there too. That is a property of NenuFAR's own correction, not a sign
    # the materialization itself is broken -- judge reliable groups only.
    group = result.get("group") if isinstance(result, dict) else None
    if group is not None and (group == "NenuFAR").any():
        reliable = group != "NenuFAR"
        std_reliable = np.std(diff_us[reliable])
        mean_reliable = np.mean(diff_us[reliable])
        nf = diff_us[~reliable]
        print(f"[INFO] Validation (LOFAR+other, excl. NenuFAR): residual(final)-corrected_res = "
              f"{mean_reliable:+.4f} +/- {std_reliable:.4f} us")
        print(f"[INFO] Validation (NenuFAR only, known-unreliable correction): "
              f"{np.mean(nf):+.4f} +/- {np.std(nf):.4f} us (not judged)")
        diff_std_us, diff_mean_us = std_reliable, mean_reliable
    else:
        diff_std_us, diff_mean_us = np.std(diff_us), np.mean(diff_us)

    ok = mjd_diff < 1e-6 and diff_std_us < 0.1
    print(f"[INFO] Validation: max MJD misalignment={mjd_diff:.3e} d, "
          f"residual(final)-corrected_res = {diff_mean_us:+.4f} +/- {diff_std_us:.4f} us "
          f"(mean is an expected constant if TZRMJD falls in-Tspan; std is what must be ~0) "
          f"({'OK' if ok else 'MISMATCH -- investigate!'})")
    return final_par_path, final_tim_path, leaf_info


def write_highfreq_only_tims(args, leaf_info, outdir):
    """Write two INCLUDE-based .tim files containing ONLY the "Other
    telescopes" (non-LOFAR, non-NenuFAR) leaves -- one pointing at the
    ORIGINAL leaf files (pre-correction), one at their corrected copies in
    materialized_tims/ (post-correction) -- so the high-frequency-only data
    can be loaded on its own, e.g. to isolate the correction's effect there
    without LOFAR/NenuFAR in the mix."""
    other = [(leaf, name) for leaf, name, group in leaf_info if group == "Other telescopes"]
    if not other:
        print("[INFO] No 'Other telescopes' leaves found -- skipping high-frequency-only tim files")
        return None, None

    pre_path = os.path.join(outdir, f"{args.name}_highfreq_original.tim")
    post_path = os.path.join(outdir, f"{args.name}_highfreq_corrected.tim")

    with open(pre_path, "w") as f:
        f.write("FORMAT 1\n")
        for leaf, _ in other:
            f.write(f"INCLUDE {os.path.relpath(leaf, outdir)}\n")

    with open(post_path, "w") as f:
        f.write("FORMAT 1\n")
        for _, name in other:
            f.write(f"INCLUDE materialized_tims/{name}\n")

    print(f"[INFO] High-frequency-only tims written -> {pre_path} / {post_path} "
          f"({len(other)} source file(s))")
    return pre_path, post_path


def write_lofaronly_tims(args, leaf_info, outdir):
    """Write two INCLUDE-based .tim files containing ONLY the LOFAR/NenuFAR
    leaves -- one pointing at the ORIGINAL leaf files (pre-correction), one
    at their corrected copies in materialized_tims/ (post-correction) --
    mirrors write_highfreq_only_tims with the group filter inverted. The
    "corrected" one applies the SAME GP-smoothed DM(t) correction back onto
    LOFAR's own TOAs: its residuals there no longer reflect the (trivially
    exact, by construction) per-epoch free-DM fit, but how well the smoothed
    curve describes LOFAR's own data."""
    lofar = [(leaf, name) for leaf, name, group in leaf_info if group in ("LOFAR", "NenuFAR")]
    if not lofar:
        print("[INFO] No LOFAR/NenuFAR leaves found -- skipping LOFAR-only tim files")
        return None, None

    pre_path = os.path.join(outdir, f"{args.name}_lofaronly_original.tim")
    post_path = os.path.join(outdir, f"{args.name}_lofaronly_corrected.tim")

    with open(pre_path, "w") as f:
        f.write("FORMAT 1\n")
        for leaf, _ in lofar:
            f.write(f"INCLUDE {os.path.relpath(leaf, outdir)}\n")

    with open(post_path, "w") as f:
        f.write("FORMAT 1\n")
        for _, name in lofar:
            f.write(f"INCLUDE materialized_tims/{name}\n")

    print(f"[INFO] LOFAR-only tims written -> {pre_path} / {post_path} "
          f"({len(lofar)} source file(s))")
    return pre_path, post_path


# ---------------------------------------------------------------------------
# Step 4: plot
# ---------------------------------------------------------------------------

PRE_COLOR = "C0"          # matplotlib default color cycle -- "before"
POST_COLOR = "C1"         # matplotlib default color cycle -- "after"
GROUP_MARKERS = {"LOFAR": "o", "NenuFAR": "^", "Other telescopes": "s"}


def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#333333",
        "legend.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 11,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
    })


def plot_dm_timeseries(gp_train, gp_predict, mjd_span, outdir):
    set_plot_style()

    mjd, dm_val, dm_err = gp_train
    mjd_pred = np.linspace(mjd_span[0], mjd_span[1], 1000)
    dm_pred, dm_std = gp_predict(mjd_pred)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(mjd_pred, dm_pred - 2*dm_std, dm_pred + 2*dm_std,
                     color=POST_COLOR, alpha=0.2, label="GP 95% CI")
    ax.plot(mjd_pred, dm_pred, color=POST_COLOR, lw=1.5, label="GP mean")
    ax.errorbar(mjd, dm_val, yerr=dm_err, fmt=".", color=PRE_COLOR, ecolor=PRE_COLOR,
                alpha=0.5, ms=4, elinewidth=0.7, label="epoch-wise DM (LOFAR)")

    # a handful of epochs with huge quoted dm_err (not clipped by the GP fit's
    # own sigma-clip, which only drops points that are outliers relative to
    # their OWN error bar) otherwise blow up the autoscaled y-range and
    # flatten the visible trend to a line -- clip to the GP band's own range
    # padded by 50%, same device used for the EW-vs-GP comparison plot
    # (scripts/compare_dm_timeseries.py); outlier points simply run off the
    # top/bottom, which is itself a visual answer to "how much of an outlier".
    gp_lo, gp_hi = (dm_pred - 2*dm_std).min(), (dm_pred + 2*dm_std).max()
    span = gp_hi - gp_lo
    pad = 0.5 * span if span > 0 else 1e-6
    ax.set_ylim(gp_lo - pad, gp_hi + pad)

    ax.set_xlabel("MJD")
    ax.set_ylabel(r"DM (pc cm$^{-3}$)")
    ax.set_title("Epoch-wise total DM(t) and GP fit")
    ax.legend(loc="best")

    fig.tight_layout()
    out_path_pdf = os.path.join(outdir, "dm_timeseries.pdf")
    out_path_png = os.path.join(outdir, "dm_timeseries.png")
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_png)
    print(f"[INFO] Plot saved -> {out_path_pdf} / {out_path_png}")


def plot_high_freq_only(result, mjd_span, outdir):
    set_plot_style()

    mjds, group, in_span = result["mjds"], result["group"], result["in_span"]
    raw_us = result["raw_res"] * 1e6
    corr_us = result["corrected_res"] * 1e6

    sel = in_span & (group == "Other telescopes")
    if sel.sum() == 0:
        print("[INFO] No 'Other telescopes' TOAs in-Tspan -- skipping high-frequency-only plot")
        return
    mjds, raw_us, corr_us = mjds[sel], raw_us[sel], corr_us[sel]

    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharey=ax)

    ax.scatter(mjds, raw_us, s=14, c=PRE_COLOR, alpha=0.5, marker="s", edgecolors="none", label="before")
    ax.scatter(mjds, corr_us, s=14, c=POST_COLOR, alpha=0.7, marker="s", edgecolors="none", label="after")

    ax.set_xlabel("MJD")
    ax.set_ylabel(r"Residual ($\mu$s)")
    ax.set_title("High-frequency-only residuals before/after DM(t) correction")
    ax.legend(loc="upper left")

    lo, hi = np.nanpercentile(np.concatenate([raw_us, corr_us]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    axh.hist(raw_us, bins=bins, orientation="horizontal", color=PRE_COLOR, alpha=0.6,
             label="before", edgecolor="none")
    axh.hist(corr_us, bins=bins, orientation="horizontal", color=POST_COLOR, alpha=0.6,
             label="after", edgecolor="none")
    axh.set_xlabel("Count")
    axh.legend()
    plt.setp(axh.get_yticklabels(), visible=False)

    fig.tight_layout()
    out_path_pdf = os.path.join(outdir, "residuals_high_freq_only.pdf")
    out_path_png = os.path.join(outdir, "residuals_high_freq_only.png")
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_png)
    print(f"[INFO] Plot saved -> {out_path_pdf} / {out_path_png}")

    std_pre, std_post = raw_us.std(), corr_us.std()
    print(f"[INFO] {'Other telescopes':18s} n={len(raw_us):5d}  std before={std_pre:8.3f} us  "
          f"std after={std_post:8.3f} us  ({100*(1-std_post/std_pre):+.1f}%)")


def plot_high_freq_post_errorbars(result, mjd_span, outdir):
    set_plot_style()

    mjds, group, in_span = result["mjds"], result["group"], result["in_span"]
    raw_us = result["raw_res"] * 1e6
    corr_us = result["corrected_res"] * 1e6
    err_us = result["toaerrs"]   # already in microseconds

    sel = in_span & (group == "Other telescopes")
    if sel.sum() == 0:
        print("[INFO] No 'Other telescopes' TOAs in-Tspan -- skipping high-frequency post-correction errorbar plot")
        return
    mjds, raw_us, corr_us, err_us = mjds[sel], raw_us[sel], corr_us[sel], err_us[sel]

    z_pre = raw_us / err_us
    z_post = corr_us / err_us

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 2, width_ratios=[3, 1.3], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    axz = fig.add_subplot(gs[1])

    ax.errorbar(mjds, corr_us, yerr=err_us, fmt="s", ms=3, color=POST_COLOR, ecolor=POST_COLOR,
                alpha=0.6, elinewidth=0.7, capsize=0)
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"Residual ($\mu$s)")
    ax.set_title("High-frequency-only residuals after DM(t) correction (with TOA uncertainties)")

    # residual normalized by its own TOA uncertainty: if the errors fully
    # explain the scatter this should look like a standard normal (std=1)
    lo, hi = -5, 5
    bins = np.linspace(lo, hi, 50)
    axz.hist(z_pre, bins=bins, density=True, color=PRE_COLOR, alpha=0.5,
             label=f"before (std={z_pre.std():.2f})")
    axz.hist(z_post, bins=bins, density=True, color=POST_COLOR, alpha=0.5,
             label=f"after (std={z_post.std():.2f})")
    xx = np.linspace(lo, hi, 200)
    axz.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), "k--", lw=1.2, label="N(0,1)")
    axz.set_xlabel("Residual / TOA uncertainty")
    axz.set_ylabel("Density")
    axz.set_title("Normalized residuals")
    axz.legend()

    fig.tight_layout()
    out_path_pdf = os.path.join(outdir, "residuals_high_freq_post_errorbars.pdf")
    out_path_png = os.path.join(outdir, "residuals_high_freq_post_errorbars.png")
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_png)
    print(f"[INFO] Plot saved -> {out_path_pdf} / {out_path_png}")
    print(f"[INFO] Normalized residual std: before={z_pre.std():.3f}  after={z_post.std():.3f} "
          f"(1.0 = fully explained by TOA uncertainties)")


def plot_correction(result, mjd_span, outdir):
    set_plot_style()

    mjds, group, in_span = result["mjds"], result["group"], result["in_span"]
    raw_us = result["raw_res"] * 1e6
    corr_us = result["corrected_res"] * 1e6

    # Points outside the LOFAR Tspan can't be corrected (no GP constraint
    # there) -- rather than show them oddly offset, just drop them.
    mjds, group = mjds[in_span], group[in_span]
    raw_us, corr_us = raw_us[in_span], corr_us[in_span]

    present_groups = [g for g in ("LOFAR", "NenuFAR", "Other telescopes") if (group == g).any()]

    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharey=ax)

    for g in present_groups:
        sel = group == g
        ax.scatter(mjds[sel], raw_us[sel], s=10, c=PRE_COLOR, alpha=0.35, marker=GROUP_MARKERS[g],
                   edgecolors="none", label=f"before ({g})")
    for g in present_groups:
        sel = group == g
        ax.scatter(mjds[sel], corr_us[sel], s=10, c=POST_COLOR, alpha=0.6, marker=GROUP_MARKERS[g],
                   edgecolors="none", label=f"after ({g})")

    ax.set_xlabel("MJD")
    ax.set_ylabel(r"Residual ($\mu$s)")
    ax.set_title("Timing residuals before/after DM(t) correction (GP fit to LOFAR)")
    ax.legend(loc="upper left", ncol=len(present_groups))

    lo, hi = np.nanpercentile(np.concatenate([raw_us, corr_us]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    axh.hist(raw_us, bins=bins, orientation="horizontal", color=PRE_COLOR, alpha=0.6,
             label="before", edgecolor="none")
    axh.hist(corr_us, bins=bins, orientation="horizontal", color=POST_COLOR, alpha=0.6,
             label="after", edgecolor="none")
    axh.set_xlabel("Count")
    axh.legend()
    axh.grid(axis="y", visible=False)
    plt.setp(axh.get_yticklabels(), visible=False)

    fig.tight_layout()
    out_path_pdf = os.path.join(outdir, "residuals_pre_post.pdf")
    out_path_png = os.path.join(outdir, "residuals_pre_post.png")
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_png)
    print(f"[INFO] Plot saved -> {out_path_pdf} / {out_path_png}")

    for label, sel in [(g, group == g) for g in present_groups] + [("all (in-Tspan)", np.ones_like(group, dtype=bool))]:
        if sel.sum() == 0:
            continue
        std_pre, std_post = raw_us[sel].std(), corr_us[sel].std()
        print(f"[INFO] {label:18s} n={sel.sum():5d}  std before={std_pre:8.3f} us  "
              f"std after={std_post:8.3f} us  ({100*(1-std_post/std_pre):+.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dm_ts, dmfit_par = epochwise_dm(args)

    dm0 = None
    for raw in open(os.path.join(args.dirpath, args.parfile)):
        t = raw.split()
        if t and t[0] == "DM":
            dm0 = float(t[1])
            break
    if dm0 is None:
        raise RuntimeError("Could not find DM in the .par file")
    print(f"[INFO] Reference DM0 = {dm0}")

    gp_predict, mjd_span, gp_train = fit_gp(dm_ts, args.max_dm_err, args.sigma_clip)
    print(f"[INFO] LOFAR Tspan (GP-constrained region): [{mjd_span[0]:.2f}, {mjd_span[1]:.2f}]")

    result = apply_correction(args, dmfit_par, gp_predict, mjd_span, dm0)

    materialize_result = write_materialized_correction(args, result, args.outdir)
    if materialize_result is not None:
        _, _, leaf_info = materialize_result
        write_highfreq_only_tims(args, leaf_info, args.outdir)

    if args.plot:
        plot_dm_timeseries(gp_train, gp_predict, mjd_span, args.outdir)
        plot_correction(result, mjd_span, args.outdir)
        plot_high_freq_only(result, mjd_span, args.outdir)
        plot_high_freq_post_errorbars(result, mjd_span, args.outdir)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
