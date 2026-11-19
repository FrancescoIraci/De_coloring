#!/usr/bin/env python3
"""
Build the Tspan-matched feather sets for J1012+5307, J1640+2224,
J1744-1134, in one pass, one epoch-wise DM fit per pulsar.

Every dataset below is restricted to the SAME Tspan (raw LOFAR coverage,
~2012-12-19/20 to 2023-05-30 for these three -- the start/end of the whole
LOFAR-EPTA monitoring campaign and of this data release, not a per-pulsar
coincidence). Names below encode both axes that matter -- station set and
correction status -- explicitly, so which-is-which never has to be
remembered:

  feathers_highfreq_corrected/<NAME>    -- high-frequency stations only
                                            (GMRT included, LOFAR/NenuFAR
                                            excluded), DM(t)-corrected.
  feathers_lofaronly_corrected/<NAME>   -- LOFAR/NenuFAR only, DM(t)-
                                            corrected: the SAME smoothed
                                            GP curve applied back onto
                                            LOFAR's own TOAs, so its
                                            residuals here reflect how well
                                            the smoothed curve describes
                                            LOFAR's own data, not the
                                            (trivially exact, by
                                            construction) per-epoch free-DM
                                            fit anymore.
  feathers_combined_corrected/<NAME>    -- ALL stations together (LOFAR +
                                            high-freq), DM(t)-corrected.
                                            = highfreq_corrected union
                                            lofaronly_corrected, as one
                                            dataset/one Mmat.
  feathers_combined_uncorrected/<NAME>  -- ALL stations together, Tspan-cut
                                            but NOT DM(t)-corrected
                                            (NE_SW -> 0, DM1/DM2 left free):
                                            the original reference, kept as
                                            a secondary check, not part of
                                            the main comparison.

The main comparison pair is highfreq_corrected vs combined_corrected (same
Tspan, same correction, only the station set differs) and, separately,
highfreq_corrected vs lofaronly_corrected (splits the combined dataset back
into its two halves). combined_uncorrected is a secondary reference only.

Pipeline, per pulsar (see the calling conversation for the reasoning behind
each step)
----------------------------------------------------------------------
1. Tspan boundary = raw min/max MJD of the LOFAR tim (lofar_mjd_range --
   cheap, no fitting). Deliberately NOT the quality-filtered epoch-wise+GP
   domain: that would need epoch-wise fitting to already exist, which is
   exactly the redundant second pass this version removes. The two are
   confirmed empirically to differ by well under a day here, and
   apply_correction()'s own in_span check (step 6) independently
   re-enforces "never extrapolate the GP" against whatever mjd_span THIS
   run's fit_gp() actually returns, regardless of this boundary's exact
   precision -- so nothing unsafe follows from the cheaper estimate.
2. Cut every station's leaves (not just LOFAR) to that boundary.
3. Fix ambiguous fit-flag formatting (fix_ambiguous_fit_flags -- a no-op
   except for combined_J1640+2224.par), zero NE_SW, drop now-orphaned
   JUMPs, refit (every fit=1 parameter, jointly) on this Tspan-cut,
   all-station dataset. This IS feathers_combined_uncorrected's final par
   -- nothing further needed for that side.
4. Epoch-wise DM fit on the (uncut -- its own coverage already matches the
   Tspan) LOFAR tim, referenced against the par from step 3 -- NOT the
   original full-baseline par -- so the frozen non-DM parameters are a
   locally-accurate description of the LOFAR years (confirmed elsewhere:
   PM/PX/DM1/DM2 shift several sigma of their full-baseline uncertainty
   when refit on this shorter window).
5. GP over the resulting DM(t).
6. Apply the correction to every station's Tspan-cut leaves (in_span check
   against THIS run's own GP domain -- see step 1).
7. Materialize via "-dmo", split into the high-freq-only and LOFAR-only
   leaf subsets, build the for-feather par (DM1/DM2/NE_SW=0, DM fit=1)
   from step 3's par, drop newly-orphaned JUMPs per subset.
8. Build all four feathers.

One pulsar per process: epoch-wise fitting alone builds a tempopulsar per
LOFAR epoch (1500+ for J1012) in one process, which is fine on its own
(dm_correction_pipeline.py's own original pipeline always did exactly
this), but chaining several PULSARS worth of that back to back in the same
process has segfaulted here -- once, after the first pulsar had already
completed correctly and written all its output to disk, crashing partway
into the second: cumulative C-level state across thousands of tempopulsar
constructions, not a per-pulsar problem. --process-one runs exactly one
pulsar's whole pipeline and exits; main() spawns one such subprocess per
pulsar. The final feather build is its own further-isolated subprocess too
(make_feathers.py --build-one, for the reasons in that script's docstring).

Epoch-wise fitting is cached (DMts.npy + dmfit_par under dm_correction_out/
matched_correction/) and reused across re-runs unless --recompute-dm is
passed -- it is the expensive step (~1h+ for these three pulsars combined),
and nothing upstream of it changes just because a new downstream dataset
(e.g. lofaronly) is being added.

Usage
-----
    conda run -n igwn-py38-feathers python make_feathers_matched.py
    conda run -n igwn-py38-feathers python make_feathers_matched.py --recompute-dm
"""
import argparse
import os
import subprocess
import sys

import dm_correction_pipeline as dmcp

HERE = os.path.dirname(os.path.abspath(__file__))
MAKE_FEATHERS = os.path.join(HERE, "make_feathers.py")

PULSARS = [
    # The three pulsars of this study, from the deduplicated reference data
    # (EPTA no_duplicates/ delivery: J1012+5307 alone had 4014 duplicate
    # LOFAR TOAs, ~21% of its LOFAR data, in the earlier delivery).
    dict(
        name="J1012+5307", dirpath="J1012+5307", orig_par="nodoubles_postfit.par",
        full_tim="J1012+5307_all_no_doubles.tim", lofar_tim="lofar/J1012+5307_ToAs_pta_dmo_no_doubles.tim",
        maxobs=30000, min_freqs=2, timeout=120.0, max_dm_err=1.0, sigma_clip=5.0,
    ),
    dict(
        name="J1640+2224", dirpath="J1640+2224", orig_par="nodoubles_postfit.par",
        full_tim="J1640+2224_all_no_doubles.tim", lofar_tim="lofar/J1640+2224_ToAs_no_doubles.tim",
        maxobs=15000, min_freqs=2, timeout=120.0, max_dm_err=1.0, sigma_clip=5.0,
    ),
    dict(
        name="J1744-1134", dirpath="J1744-1134", orig_par="nodoubles_postfit.par",
        full_tim="J1744-1134_all_no_doubles.tim", lofar_tim="lofar/J1744-1134_ToAs_pta_dmo_no_doubles.tim",
        maxobs=10000, min_freqs=2, timeout=120.0, max_dm_err=1.0, sigma_clip=5.0,
    ),
]

# output directory name <-> what it holds, kept in one place so process_one
# and main() can't drift apart
DIR_HIGHFREQ   = "feathers_highfreq_corrected"
DIR_HIGHFREQ_U = "feathers_highfreq_uncorrected"
DIR_LOFARONLY  = "feathers_lofaronly_corrected"
DIR_LOFARONLY_U = "feathers_lofaronly_uncorrected"
DIR_COMBINED_C = "feathers_combined_corrected"
DIR_COMBINED_U = "feathers_combined_uncorrected"


def build_matched_tim(leaf_paths, mjd_lo, mjd_hi, tims_dir, top_path):
    """Filter every leaf in leaf_paths down to [mjd_lo, mjd_hi], write the
    filtered copies under tims_dir, and write a top-level INCLUDE list
    (top_path) pointing at them. Returns (n_kept, n_dropped) totals."""
    os.makedirs(tims_dir, exist_ok=True)
    include_lines = []
    used_names = {}
    total_kept = total_dropped = 0
    for leaf in leaf_paths:
        base = os.path.basename(leaf)
        k = used_names.get(base, 0)
        name = base if k == 0 else f"{os.path.splitext(base)[0]}_{k}{os.path.splitext(base)[1]}"
        used_names[base] = k + 1

        out_path = os.path.join(tims_dir, name)
        n_kept, n_dropped = dmcp.filter_leaf_by_mjd(leaf, out_path, mjd_lo, mjd_hi)
        total_kept += n_kept
        total_dropped += n_dropped
        if n_kept > 0:
            include_lines.append(f"INCLUDE {os.path.relpath(out_path, os.path.dirname(top_path))}")

    with open(top_path, "w") as f:
        f.write("FORMAT 1\n" + "\n".join(include_lines) + "\n")
    return total_kept, total_dropped


def process_one(cfg, outdir_base, noisedict, ephem, clk, recompute_dm):
    d = os.path.join(HERE, cfg["dirpath"])
    raw_par = os.path.join(d, cfg["orig_par"])
    full_tim = os.path.join(d, cfg["full_tim"])
    lofar_tim = os.path.join(d, cfg["lofar_tim"])
    stem = os.path.join(d, "dm_correction_out", os.path.splitext(cfg["orig_par"])[0])
    os.makedirs(os.path.dirname(stem), exist_ok=True)

    # --- step 1-3: Tspan boundary, cut, refit -> combined_uncorrected's par ---
    mjd_lo, mjd_hi = dmcp.lofar_mjd_range(lofar_tim)
    print(f"[matched] {cfg['name']}: Tspan (raw LOFAR coverage) = [{mjd_lo:.2f}, {mjd_hi:.2f}]",
          flush=True)

    fitflags_par = stem + ".fitflags_fixed.par"
    n_fixed = dmcp.fix_ambiguous_fit_flags(raw_par, fitflags_par)
    if n_fixed:
        print(f"[matched] {cfg['name']}: fixed {n_fixed} ambiguous fit-flag line(s)", flush=True)

    full_leaves = dmcp._find_leaf_tim_paths(full_tim)
    full_matched_tim = stem + "_matched_full.tim"
    n_kept, n_dropped = build_matched_tim(
        full_leaves, mjd_lo, mjd_hi,
        os.path.join(d, "dm_correction_out", "matched_tims", "full_uncorrected"), full_matched_tim)
    print(f"[matched] {cfg['name']} full: kept {n_kept}, dropped {n_dropped} (outside Tspan)",
          flush=True)

    nesw_par = stem + ".nesw_zeroed.par"
    dmcp.write_nesw_zeroed_par(fitflags_par, nesw_par)
    full_par = stem + ".matched_full.par"
    dmcp.strip_orphaned_jumps(nesw_par, full_matched_tim, full_par)

    full_par_refit = stem + ".matched_full.refit.par"
    subprocess.run([sys.executable, os.path.join(HERE, "refit_par.py"),
                     full_par, full_matched_tim, full_par_refit, str(cfg["maxobs"])], check=True)

    dm0 = None
    for raw in open(full_par_refit):
        t = raw.split()
        if t and t[0] == "DM":
            dm0 = float(t[1])
            break
    if dm0 is None:
        raise RuntimeError(f"could not find DM in {full_par_refit}")
    print(f"[matched] {cfg['name']}: refit DM0 = {dm0}", flush=True)

    # --- step 4-6: epoch-wise DM, GP, correction -- referenced against the
    # refit par, ONCE (not against the original full-baseline par first) ---
    ew_outdir = os.path.join(d, "dm_correction_out", "matched_correction")
    ew_args = argparse.Namespace(
        name=cfg["name"], dirpath=d, parfile=os.path.relpath(full_par_refit, d),
        lofar_tim=cfg["lofar_tim"], full_tim=os.path.relpath(full_matched_tim, d),
        outdir=ew_outdir, min_freqs=cfg["min_freqs"], timeout=cfg["timeout"],
        maxobs=cfg["maxobs"], max_dm_err=cfg["max_dm_err"], sigma_clip=cfg["sigma_clip"],
        recompute_dm=recompute_dm,
    )
    print(f"[matched] {cfg['name']}: epoch-wise DM fit referenced against the refit par ...",
          flush=True)
    dm_ts, dmfit_par = dmcp.epochwise_dm(ew_args)
    gp_predict, mjd_span, gp_train = dmcp.fit_gp(dm_ts, cfg["max_dm_err"], cfg["sigma_clip"])
    print(f"[matched] {cfg['name']}: GP-constrained Tspan = [{mjd_span[0]:.2f}, {mjd_span[1]:.2f}]",
          flush=True)

    result = dmcp.apply_correction(ew_args, dmfit_par, gp_predict, mjd_span, dm0)

    # diagnostic plots -- same functions/filenames as dm_correction_pipeline.py's
    # own --plot CLI flag (dm_timeseries, residuals_pre_post, residuals_high_freq_*),
    # just called directly here since process_one() already has everything
    # they need in hand. Saved next to the par/tim working files, one level
    # up from matched_correction/ (same place the original 3 pulsars' plots
    # already lived, and where summary/*.tex expects to find them).
    plots_dir = os.path.join(d, "dm_correction_out")
    dmcp.plot_dm_timeseries(gp_train, gp_predict, mjd_span, plots_dir)
    dmcp.plot_correction(result, mjd_span, plots_dir)
    dmcp.plot_high_freq_only(result, mjd_span, plots_dir)
    dmcp.plot_high_freq_post_errorbars(result, mjd_span, plots_dir)

    materialize_result = dmcp.write_materialized_correction(ew_args, result, ew_outdir)
    if materialize_result is None:
        raise RuntimeError(f"{cfg['name']}: materialization failed (TOA count mismatch)")
    _, combined_corrected_tim, leaf_info = materialize_result
    # combined_corrected_tim already includes EVERY station (LOFAR included)
    # with the "-dmo" correction applied -- write_materialized_correction's
    # own output, before the two splits below narrow it down. Its paired par
    # (materialize_result[0]) freezes DM's fit flag (DM fit=0, for a plain
    # tempo2 re-fit); for a feather we want the SAME forfeather par as every
    # other corrected side (DM fit=1), built once below and reused for all.

    hf_orig_tim, hf_corr_tim = dmcp.write_highfreq_only_tims(ew_args, leaf_info, ew_outdir)
    if hf_corr_tim is None:
        raise RuntimeError(f"{cfg['name']}: no high-freq leaves found")

    lofar_orig_tim, lofar_corr_tim = dmcp.write_lofaronly_tims(ew_args, leaf_info, ew_outdir)
    if lofar_corr_tim is None:
        raise RuntimeError(f"{cfg['name']}: no LOFAR leaves found")

    # --- step 7: for-feather par (DM1/DM2/NE_SW=0, DM fit=1), one shared
    # base, each paired with its own tim and its own orphaned-JUMP check
    # (a station could in principle vanish from one tim and not another,
    # though for combined_corrected_tim -- no station excluded -- none are
    # expected to) ---
    forfeather_par = stem + ".forfeather.par"
    dmcp.write_forfeather_corrected_par(full_par_refit, forfeather_par)

    def cleaned_par(suffix, tim, label):
        out_par = stem + f".forfeather.{suffix}.clean.par"
        n_dropped_jumps = dmcp.strip_orphaned_jumps(forfeather_par, tim, out_par)
        if n_dropped_jumps:
            print(f"[matched] {cfg['name']}: dropped {n_dropped_jumps} orphaned JUMP(s) ({label})",
                  flush=True)
        return out_par

    forfeather_par_combined = cleaned_par("combined", combined_corrected_tim, "combined corrected")
    forfeather_par_highfreq = cleaned_par("highfreq", hf_corr_tim, "highfreq corrected")
    forfeather_par_lofaronly = cleaned_par("lofaronly", lofar_corr_tim, "lofaronly corrected")

    # lofaronly_uncorrected: same LOFAR/NenuFAR station restriction as
    # above, but no DM(t) correction -- pairs with full_par_refit (the SAME
    # uncorrected par combined_uncorrected uses below), not forfeather_par
    # (that one's DM1/DM2/NE_SW=0 zeroing is only needed when the GP
    # correction has been baked into the TOAs via -dmo, which lofar_orig_tim
    # -- the pre-correction leaves -- never had).
    lofaronly_uncorrected_par = stem + ".lofaronly_uncorrected.clean.par"
    n_dropped_jumps = dmcp.strip_orphaned_jumps(full_par_refit, lofar_orig_tim, lofaronly_uncorrected_par)
    if n_dropped_jumps:
        print(f"[matched] {cfg['name']}: dropped {n_dropped_jumps} orphaned JUMP(s) (lofaronly uncorrected)",
              flush=True)

    # highfreq_uncorrected: same idea, mirrored for the high-frequency split
    highfreq_uncorrected_par = stem + ".highfreq_uncorrected.clean.par"
    n_dropped_jumps = dmcp.strip_orphaned_jumps(full_par_refit, hf_orig_tim, highfreq_uncorrected_par)
    if n_dropped_jumps:
        print(f"[matched] {cfg['name']}: dropped {n_dropped_jumps} orphaned JUMP(s) (highfreq uncorrected)",
              flush=True)

    # --- step 8: build all four feathers (each its own subprocess, with a
    # consensus-retry: see spawn_build_checked's docstring) ---
    def spawn_build_checked(parfile, timfile, outpath, max_attempts=4):
        """Build a feather via an isolated subprocess, retrying if the
        result looks corrupted. tempo2/libstempo has, elsewhere in this
        project, both segfaulted and -- rarer, and more dangerous because
        it raises no error -- silently returned a wrong-but-plausible
        residual array from a single, properly isolated Pulsar()
        construction, non-deterministically, on byte-identical input. A
        lone successful-looking build is therefore not proof of
        correctness. Two independent builds (fresh subprocess each) are
        accepted once they agree with each other on the residual std to
        within 0.1% -- a lone corrupted build is not expected to
        reproduce the same wrong value twice in a row."""
        cmd_base = [sys.executable, MAKE_FEATHERS, "--build-one", parfile, timfile, outpath]
        if noisedict:
            cmd_base += ["--noisedict", noisedict]
        if ephem:
            cmd_base += ["--ephem", ephem]
        if clk:
            cmd_base += ["--clk", clk]

        prev_std = None
        for attempt in range(1, max_attempts + 1):
            try:
                subprocess.run(cmd_base, check=True)
            except subprocess.CalledProcessError:
                print(f"[matched]   {os.path.basename(outpath)}: build attempt {attempt} "
                      f"crashed, retrying ...", flush=True)
                prev_std = None
                continue
            std = dmcp._read_feather_residual_std_us(outpath)
            if prev_std is not None and abs(std - prev_std) < 1e-3 * max(std, prev_std, 1e-9):
                return std
            if attempt < max_attempts:
                print(f"[matched]   {os.path.basename(outpath)}: attempt {attempt} std={std:.3f} us, "
                      f"confirming with another independent build ...", flush=True)
            prev_std = std
        raise RuntimeError(f"{outpath}: {max_attempts} build attempts never produced two "
                            f"consistent results (tempo2/libstempo instability) -- last std={prev_std}")

    print(f"[matched] {cfg['name']}: building feathers (isolated subprocess each, with "
          f"consensus-retry) ...", flush=True)
    spawn_build_checked(forfeather_par_highfreq, hf_corr_tim,
                         os.path.join(outdir_base, DIR_HIGHFREQ, cfg["name"]))
    spawn_build_checked(highfreq_uncorrected_par, hf_orig_tim,
                         os.path.join(outdir_base, DIR_HIGHFREQ_U, cfg["name"]))
    spawn_build_checked(forfeather_par_lofaronly, lofar_corr_tim,
                         os.path.join(outdir_base, DIR_LOFARONLY, cfg["name"]))
    spawn_build_checked(lofaronly_uncorrected_par, lofar_orig_tim,
                         os.path.join(outdir_base, DIR_LOFARONLY_U, cfg["name"]))
    spawn_build_checked(forfeather_par_combined, combined_corrected_tim,
                         os.path.join(outdir_base, DIR_COMBINED_C, cfg["name"]))
    spawn_build_checked(full_par_refit, full_matched_tim,
                         os.path.join(outdir_base, DIR_COMBINED_U, cfg["name"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--ephem", default=None)
    ap.add_argument("--clk", default=None)
    ap.add_argument("--noisedict", default=None)
    ap.add_argument("--recompute-dm", action="store_true",
                     help="force re-running epoch-wise DM fitting even if a cached "
                          "DMts.npy already exists (default: reuse the cache)")
    ap.add_argument("--process-one", default=None, help=argparse.SUPPRESS)  # internal
    args = ap.parse_args()

    if args.process_one:
        cfg = next(c for c in PULSARS if c["name"] == args.process_one)
        process_one(cfg, args.outdir, args.noisedict, args.ephem, args.clk, args.recompute_dm)
        return

    for cfg in PULSARS:
        cmd = [sys.executable, os.path.abspath(__file__), "--process-one", cfg["name"],
               "--outdir", args.outdir]
        if args.noisedict:
            cmd += ["--noisedict", args.noisedict]
        if args.ephem:
            cmd += ["--ephem", args.ephem]
        if args.clk:
            cmd += ["--clk", args.clk]
        if args.recompute_dm:
            cmd += ["--recompute-dm"]
        # The whole --process-one run (epoch-wise fit through materialization)
        # is itself many tempopulsar constructions in one process -- fine on
        # its own in general, but has segfaulted for at least one pulsar here
        # (cumulative C-level state, same family as spawn_build_checked's).
        # Retry the whole thing from scratch rather than leaving a pulsar
        # half-done; each attempt is fully independent (epoch-wise fitting
        # is deterministic given the same inputs, so re-running costs time,
        # not correctness) -- and cheap on a retry if the cache from a
        # mostly-successful earlier attempt is still there.
        for attempt in range(1, 4):
            try:
                subprocess.run(cmd, check=True)
                break
            except subprocess.CalledProcessError:
                print(f"[matched] {cfg['name']}: --process-one crashed on attempt {attempt}, "
                      f"retrying from scratch ...")
        else:
            raise RuntimeError(f"{cfg['name']}: --process-one crashed on all attempts")
        print()

    print(f"[matched] done ->")
    for name in (DIR_HIGHFREQ, DIR_HIGHFREQ_U, DIR_LOFARONLY, DIR_LOFARONLY_U, DIR_COMBINED_C, DIR_COMBINED_U):
        print(f"  {os.path.join(args.outdir, name)}/")


if __name__ == "__main__":
    main()
