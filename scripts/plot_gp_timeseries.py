"""Implied-GP time series per pulsar: one figure, shared time axis, one row
per Gaussian process (red noise, DM, chromatic, solar wind) plus the timing
residuals as the bottom row.

For each posterior draw, the GP Fourier coefficients are drawn from their
conditional distribution given the residuals and that draw's hyperparameters
(discovery's PulsarLikelihood.sample_conditional); basis @ coefficients gives
the GP delay at the observing epochs, summarised as median and 16-84% band.

Conventions (after run_spna_all.py in the eptadr3-noiseanalysis pipeline):
- red noise: achromatic basis, delay in microseconds;
- DM / chromatic GPs: plotted on the plain Fourier basis, i.e. the delay at
  the reference frequency (1400 MHz);
- solar wind GP: plotted on its solar basis (actual per-TOA geometry);
- the deterministic solar-wind delay (n_earth) is NOT subtracted: the
  conditional likelihood is rebuilt without delay terms (discovery has no
  conditional with delays), which is the same approximation used there.

Unlike plot_posteriors.py, this script NEEDS discovery (and the feathers):
run it in the discovery-local env on the Mac or inside the container.

Usage:
  python plot_gp_timeseries.py --samples_dir samples/check --model dr3_v1 \
      --feathers_path /path/to/feathers [--pulsar J0406+3039] [--ndraws 100]
"""

import os

# This script is CPU-friendly (one Woodbury solve per draw); default to CPU
# and keep the legacy JAX_PLATFORM_NAME in sync — the GPU container sets it
# to 'gpu', which otherwise conflicts with JAX_PLATFORMS=cpu on nodes
# without a GPU.
_plat = os.environ.get('JAX_PLATFORMS', 'cpu')
os.environ['JAX_PLATFORMS'] = _plat
os.environ['JAX_PLATFORM_NAME'] = _plat.split(',')[0]

import re
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model_builder as mb
from model_builder import ds, ds_sig
from plot_posteriors import load_prior_layout, load_chains

import jax.random as jr

DAY = 86400.0
YR = 365.25 * DAY
DEFAULT_FREF = 1400.0    # reference frequency for the chromatic rows (DM, SW)


def draws_to_params(flat, row, constants):
    """One flat posterior sample -> discovery params dict (vectors reassembled)."""
    params = dict(constants)
    vectors = {}
    for name, val in zip(flat, row):
        m = re.match(r'(.*)\[(\d+)\]$', name)
        if m:
            vectors.setdefault(m.group(1), {})[int(m.group(2))] = float(val)
        else:
            params[name] = float(val)
    for base, comps in vectors.items():
        arr = np.array([comps[i] for i in range(len(comps))])
        params[f'{base}({len(arr)})'] = arr
    return params


def display_basis(psr, name, gp, tspan, fref):
    """Basis used for PLOTTING (not the sampling one): see module docstring."""
    ncomp = int(gp['components'])
    kind = gp.get('basis', 'achromatic')
    if kind == 'solar':
        # rescale the per-TOA solar basis to the reference frequency, so the
        # SW row is directly comparable with the DM / red-noise rows
        fmat = np.asarray(ds_sig.dmfourierbasis_solar(psr, ncomp, T=tspan)[2])
        fmat = fmat * (np.asarray(psr.freqs) / fref)[:, None] ** 2
        return fmat, name
    if kind in ('dm', 'chromatic'):
        # plain fourier basis = signal at the GP's own fref; rescale to the
        # requested display frequency
        gp_fref = gp.get('fref', DEFAULT_FREF)
        fmat = np.asarray(ds_sig.fourierbasis(psr, ncomp, T=tspan)[2])
        fmat = fmat * (gp_fref / fref) ** 2
        return fmat, f'{name} @ {fref:.0f} MHz'
    return ds_sig.fourierbasis(psr, ncomp, T=tspan)[2], name


def mjd_to_year(mjd):
    return 2000.0 + (mjd - 51544.5) / 365.25


def year_to_mjd(year):
    return 51544.5 + (year - 2000.0) * 365.25


def output_dir(samples_dir, model, psr_name):
    """plot_output path for one pulsar, mirroring the samples tree
    (samples/<dataset>/... -> plot_output/<dataset>/...), same convention
    as plot_posteriors.py."""
    parts = os.path.normpath(samples_dir).split(os.sep)
    if 'samples' in parts:
        parts[len(parts) - 1 - parts[::-1].index('samples')] = 'plot_output'
        plot_dir = os.sep.join(parts)
    else:
        plot_dir = os.path.join(samples_dir, 'plot_output')
    return os.path.join(plot_dir, psr_name, model)


def plot_pulsar(psr_name, args):
    pdir = os.path.join(args.samples_dir, psr_name, args.model)
    nm_path = os.path.join(pdir, 'noise_model.json')
    if not os.path.isdir(pdir) or not os.path.exists(nm_path):
        print(f'{psr_name}: SKIP (no {args.model} run with noise_model.json)')
        return

    out = output_dir(args.samples_dir, args.model, psr_name)
    if args.skip_existing and os.path.exists(os.path.join(out, 'gp_timeseries.pdf')):
        print(f'{psr_name}: SKIP (already plotted, --skip_existing)')
        return

    cfg = mb.load_noise_model(nm_path)
    psr = ds.pulsar.Pulsar.read_feather(os.path.join(args.feathers_path, psr_name))
    tspan = ds_sig.getspan([psr])

    # conditional likelihood: same model WITHOUT deterministic delays —
    # discovery's conditional refuses ANY delay term. NOTE: this workaround
    # is SW-specific; if the noise-model schema ever grows other
    # deterministic terms (e.g. chromatic dips), they must ALL be disabled
    # here and hand-computed per draw like the SW one below.
    cfg_cond = json.loads(json.dumps(cfg))          # deep copy
    cfg_cond.get('deterministic', {})['solar_wind'] = False
    psl = mb.build_singlepulsar_likelihood(psr, cfg_cond)

    flat, short, lo, hi, constants = load_prior_layout(
        os.path.join(pdir, 'prior_dict.json'), psr_name)
    p, ll = load_chains(pdir, args.burn_epochs, 0)
    x = lo[None, :, None] + p * (hi - lo)[None, :, None]
    samples = x.transpose(0, 2, 1).reshape(-1, len(flat))

    # probe the conditional once to find the coefficient block of each GP
    med = draws_to_params(flat, np.median(samples, axis=0), constants)
    key = jr.PRNGKey(args.seed)
    key, template = psl.sample_conditional(key, med)
    blocks = {}
    for name in cfg.get('gps', {}):
        match = [k for k in template if f'_{name}_coefficients' in k]
        if match:
            blocks[name] = match[0]
    if not blocks:
        print(f'{psr_name}: SKIP (no GP coefficient blocks found)')
        return

    bases, labels = {}, {}
    for name in blocks:
        bases[name], labels[name] = display_basis(psr, name, cfg['gps'][name], tspan,
                                                  args.fref)
        bases[name] = np.asarray(bases[name])

    # deterministic solar wind: per-TOA shape at n_earth = 1, scaled by each
    # draw's own n_earth so both SW components stay mutually consistent
    sw_det_shape = None
    n_earth_name = f'{psr_name}_sw_n_earth'
    if cfg.get('deterministic', {}).get('solar_wind', False) and n_earth_name in flat:
        from model_builder import ds_solar
        sw_det_shape = np.asarray(ds_solar.make_solardm(psr)(1.0))
        # same reference-frequency rescaling as the SW GP basis
        sw_det_shape = sw_det_shape * (np.asarray(psr.freqs) / args.fref) ** 2

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(samples.shape[0], size=min(args.ndraws, samples.shape[0]),
                     replace=False)
    sigs = {name: [] for name in blocks}
    sw_det = []
    for i in idx:
        params_i = draws_to_params(flat, samples[i], constants)
        key, cond = psl.sample_conditional(key, params_i)
        for name, block in blocks.items():
            sigs[name].append(bases[name] @ np.asarray(cond[block]))
        if sw_det_shape is not None:
            sw_det.append(params_i[n_earth_name] * sw_det_shape)

    # what the residuals can actually see: each draw minus its timing-model
    # projection (weighted LS on psr.Mmat). A steep red noise puts most of its
    # power in quadratic-like modes that F0/F1 absorb -- the raw band shows
    # them, the residuals never contain them. Achromatic GPs only: the
    # chromatic rows are displayed at fref, not at the TOAs' own frequencies
    # the timing model acts on.
    M = np.asarray(psr.Mmat)
    M = M / np.linalg.norm(M, axis=0)
    w = 1.0 / np.asarray(psr.toaerrs) ** 2
    MtWM = M.T @ (w[:, None] * M)
    tmfree = {}
    for name in blocks:
        if cfg['gps'][name].get('basis', 'achromatic') == 'achromatic':
            s = np.asarray(sigs[name])
            tmfree[name] = s - (M @ np.linalg.solve(MtWM, (w[:, None] * M).T @ s.T)).T

    # assemble panels; the SW row shows the FULL solar-wind signal (det + GP)
    panels = []
    sw_gp_name = next((n for n in blocks
                       if cfg['gps'][n].get('basis') == 'solar'), None)
    for name in blocks:
        if name == sw_gp_name:
            continue
        panels.append((name, labels[name], np.asarray(sigs[name])))
    if sw_det_shape is not None or sw_gp_name is not None:
        sw = 0.0
        parts = []
        if sw_det_shape is not None:
            sw = sw + np.asarray(sw_det)
            parts.append('sw det')
        if sw_gp_name is not None:
            sw = sw + np.asarray(sigs[sw_gp_name])
            parts.append('swgp')
        panels.append(('sw', ' + '.join(parts) + f' @ {args.fref:.0f} MHz', sw))

    order = np.argsort(psr.toas)
    t_mjd = np.asarray(psr.toas)[order] / DAY
    res_us = np.asarray(psr.residuals)[order] * 1e6
    err_us = np.asarray(psr.toaerrs)[order] * 1e6

    npan = len(panels) + 1
    fig, axes = plt.subplots(npan, 1, figsize=(9, 2.3 * npan), sharex=True)
    axes = np.atleast_1d(axes)
    bands = {}
    for ax, (name, label, s) in zip(axes[:-1], panels):
        s = s[:, order] * 1e6
        q16, q50, q84 = np.percentile(s, [16, 50, 84], axis=0)
        bands[name] = (q16, q50, q84)
        ax.fill_between(t_mjd, q16, q84, alpha=0.3, color='steelblue')
        ax.plot(t_mjd, q50, lw=1.0, color='steelblue')
        ax.axhline(0.0, lw=0.5, color='k', alpha=0.4)
        ax.set_ylabel(f'{label}\n[μs]', fontsize=8)
    axes[-1].errorbar(t_mjd, res_us, yerr=err_us, fmt='.', ms=2, lw=0.5,
                      color='indianred', alpha=0.6)
    axes[-1].set_ylabel('residuals\n[μs]', fontsize=8)
    axes[-1].set_xlabel('MJD')
    # calendar years on top of the first row
    secax = axes[0].secondary_xaxis('top', functions=(mjd_to_year, year_to_mjd))
    secax.set_xlabel('year', fontsize=8)
    fig.suptitle(f'{psr_name} ({len(idx)} draws)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'gp_timeseries.pdf')
    fig.savefig(path); plt.close(fig)

    # underlying band data (μs, at args.fref for dm/chromatic rows), so a
    # separate lightweight script can overlay these against e.g. the
    # epoch-wise-measured DM time series without needing discovery/JAX again
    npz_path = os.path.join(out, 'gp_timeseries.npz')
    # the residuals the bands were conditioned on, so overlays never mix a
    # reconstruction with a different (e.g. rebuilt) feather's residuals
    npz_data = {'t_mjd': t_mjd, 'fref': args.fref, 'res_us': res_us, 'err_us': err_us}
    for name, (q16, q50, q84) in bands.items():
        npz_data[f'{name}_q16'] = q16
        npz_data[f'{name}_q50'] = q50
        npz_data[f'{name}_q84'] = q84
    for name, s in tmfree.items():
        q16, q50, q84 = np.percentile(s[:, order] * 1e6, [16, 50, 84], axis=0)
        npz_data[f'{name}_tmfree_q16'] = q16
        npz_data[f'{name}_tmfree_q50'] = q50
        npz_data[f'{name}_tmfree_q84'] = q84
    np.savez(npz_path, **npz_data)

    # per-pulsar provenance log next to the PDF (one sbatch sweep produces a
    # single SLURM stdout, so per-pulsar logging happens here)
    import datetime
    with open(os.path.join(out, 'gp_timeseries.log'), 'a') as fh:
        fh.write(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | '
                 f'model={args.model} fref={args.fref:.0f}MHz '
                 f'burn_epochs={args.burn_epochs} ndraws={len(idx)} seed={args.seed} | '
                 f'GPs: {", ".join(blocks)}'
                 + (f' + sw det' if sw_det_shape is not None else '') + '\n')
    print(f'{psr_name}: {len(blocks)} GPs ({", ".join(blocks)}) -> {path}')


def main():
    ap = argparse.ArgumentParser(description='Implied-GP time series plots')
    ap.add_argument('--samples_dir', required=True)
    ap.add_argument('--model', required=True, help='run label, e.g. dmrn')
    ap.add_argument('--feathers_path', required=True)
    ap.add_argument('--pulsar', nargs='+', default=None,
                    help='one or more pulsars (default: all in samples_dir)')
    ap.add_argument('--burn_epochs', type=int, default=2)
    ap.add_argument('--ndraws', type=int, default=100,
                    help='posterior draws for the conditional bands (default 100)')
    ap.add_argument('--fref', type=float, default=DEFAULT_FREF,
                    help='display reference frequency in MHz for the '
                         'chromatic rows (default: 1400)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip_existing', action='store_true',
                    help='Skip pulsars that already have a gp_timeseries.pdf '
                         '(resume an interrupted sweep without recomputing finished ones)')
    args = ap.parse_args()

    pulsars = args.pulsar if args.pulsar else \
              sorted(d for d in os.listdir(args.samples_dir) if d.startswith('J'))
    failed = []
    for psr_name in pulsars:
        try:
            plot_pulsar(psr_name, args)
        except Exception as e:
            # a sweep must survive individual pulsars (missing chains,
            # corrupted files, ...): report and move on
            print(f'{psr_name}: SKIP ({type(e).__name__}: {e})')
            failed.append(psr_name)
    if failed:
        print(f'\nskipped {len(failed)} pulsar(s): {", ".join(failed)}')


if __name__ == '__main__':
    main()
