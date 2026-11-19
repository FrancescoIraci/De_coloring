"""Posterior plots and diagnostics for single-pulsar noise analysis runs.

Reads the jexplore chains produced by spna_sampling.py from the dataset tree
<samples_dir>/<pulsar>/<model>/ (epochs_N.h5 + prior_dict.json) and mirrors
it under plot_output:

  plot_output/<dataset>/<pulsar>/<model>/corner.pdf
                                         posteriors_1d.pdf
                                         ll_trace.pdf
                                         summary.json

corner.pdf shows the stochastic non-WN parameters (red noise, DM, chromatic,
SW GP, n_earth); posteriors_1d.pdf all parameters grouped by component;
ll_trace.pdf the walker log-L traces (convergence check).

No discovery needed: everything is reconstructed from prior_dict.json and
the HDF5 files, so this can run anywhere (laptop or cluster login node).

Usage:
  python plot_posteriors.py --samples_dir samples/check --model wn_rn_dm
  python plot_posteriors.py --samples_dir samples/check --model wn_rn_dm \
      --pulsar J1909-3744 --burn_epochs 5
"""

import os
import re
import json
import glob
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import corner as corner_module
except ImportError:
    corner_module = None

# parameter-name suffixes used to classify parameters; longest match wins,
# group = whatever precedes the suffix (backend name for WN, GP name otherwise)
WN_SUFFIXES = ['efac', 'log10_tnequad', 'log10_t2equad', 'log10_ecorr']
GP_SUFFIXES = ['log10_A', 'gamma', 'log10_fb', 'alpha', 'log10_rho', 'n_earth', 'k_dropbin']
ALL_SUFFIXES = sorted(WN_SUFFIXES + GP_SUFFIXES, key=len, reverse=True)


# ── chain loading ───────────────────────────────────────────────────────────

def load_prior_layout(prior_path, psr):
    """Rebuild the flattened sampling layout from prior_dict.json.

    Returns (flat_names, short_names, lows, highs, constants): flat_names are
    the sampled dimensions in chain order, with vector params like
    ..._log10_rho(30) expanded to ..._log10_rho[i]; short_names have the
    pulsar prefix stripped.
    """
    with open(prior_path) as f:
        priors = json.load(f)  # insertion order == likelihood param order

    flat_names, lows, highs = [], [], []
    constants = {}
    for name, spec in priors.items():
        m = re.search(r'\((\d+)\)$', name)
        size = int(m.group(1)) if m else None
        base = name[:m.start()] if m else name

        if spec['dist'] == 'constant':
            constants[base] = spec['value']
            continue
        if spec['dist'] != 'uniform':
            raise ValueError(f"{name}: only uniform/constant priors are supported, got {spec['dist']}")

        if size is None:
            flat_names.append(base)
            lows.append(spec['min']); highs.append(spec['max'])
        else:
            for i in range(size):
                flat_names.append(f'{base}[{i}]')
                lows.append(spec['min']); highs.append(spec['max'])

    short = [n.replace(f'{psr}_', '', 1) for n in flat_names]
    return flat_names, short, np.array(lows), np.array(highs), constants


def load_chains(pulsar_dir, burn_epochs=3, burn_iters=0):
    """Concatenate the cold chain over all epoch files of one pulsar.

    Returns p (nwalker, ndim, iters) in unit-cube space and ll (nwalker, iters).
    nwalker is read from the file itself.
    """
    files = sorted(glob.glob(os.path.join(pulsar_dir, 'epochs_*.h5')),
                   key=lambda x: int(re.search(r'epochs_(\d+)', x).group(1)))
    if not files:
        raise FileNotFoundError(f'no epochs_*.h5 in {pulsar_dir}')
    nfiles = len(files)
    files = files[burn_epochs:]
    if not files:
        raise ValueError(f'{pulsar_dir}: burn_epochs discards all {nfiles} epochs')

    p_list, ll_list = [], []
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            key = list(f.keys())[0]           # group name matches epoch index
            ntemps = len(f[f'{key}/sampling/temps'][:])
            # jexplore interleaves temperatures (beta = tile(temps, nwalker)):
            # chain row i sits at temps[i % ntemps], so the cold chain is
            # every ntemps-th row — NOT the first nwalker rows.
            # Read contiguously then slice in memory: strided dataset reads
            # are very slow in h5py (per-element hyperslab selection).
            p_list.append(f[f'{key}/samples/p'][:][::ntemps])
            ll_list.append(f[f'{key}/samples/ll'][:][::ntemps, 0])

    p  = np.concatenate(p_list,  axis=-1)
    ll = np.concatenate(ll_list, axis=-1)
    return p[:, :, burn_iters:], ll[:, burn_iters:]


def read_run_meta(pulsar_dir):
    """Sampler configuration and wall time of the run.

    Walkers/temps/iters come from the h5 metadata; the sampling wall time is
    parsed from the slurm log that spna_sampling writes in the run directory
    (None when the log is missing, e.g. interactive runs).
    """
    files = sorted(glob.glob(os.path.join(pulsar_dir, 'epochs_*.h5')),
                   key=lambda x: int(re.search(r'epochs_(\d+)', x).group(1)))
    meta = {'n_epochs': len(files)}
    with h5py.File(files[0], 'r') as f:
        key = list(f.keys())[0]
        temps = f[f'{key}/sampling/temps'][:]
        meta['n_walkers'] = int(f[f'{key}/sampling/nwalker'][()])
        meta['n_temps'] = int(len(temps))
        meta['temp_max'] = round(float(temps.max()), 3)
        meta['iters_per_epoch'] = int(f[f'{key}/samples/ll'].shape[-1])

    sampling_minutes = None
    for logf in sorted(glob.glob(os.path.join(pulsar_dir, 'slurm_*.log'))):
        with open(logf, errors='ignore') as fh:
            for line in fh:
                m = re.search(r'Sampling done in ([\d.]+) min', line)
                if m:
                    sampling_minutes = float(m.group(1))
    meta['sampling_minutes'] = sampling_minutes
    return meta


def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time (in iterations) of one parameter.

    x: (nwalker, niters) post-burn samples of a single parameter. emcee-style
    estimate: per-walker autocorrelation function computed via FFT, averaged
    over the ensemble, integrated with Sokal's self-consistent window
    (window M = smallest lag with M >= c * tau(M)). Returns NaN for
    (near-)constant chains, where the ACF is undefined.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    npad = 1
    while npad < n:
        npad <<= 1
    f = np.fft.rfft(x, n=2 * npad, axis=1)
    acf = np.fft.irfft(f * np.conj(f), axis=1)[:, :n].mean(axis=0)
    if not np.isfinite(acf[0]) or acf[0] <= 0:
        return np.nan
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    window = np.arange(len(taus)) >= c * taus
    return float(taus[np.argmax(window)]) if window.any() else float(taus[-1])


def read_acceptance(pulsar_dir, burn_epochs=3):
    """Mean step-acceptance rates from the h5 'stats' groups (post-burn).

    Our sampler cycle is fixed by spna_sampling.py: TSwap, Stretch, DE,
    Gaussian random walk. TSwap is reported per temperature rung (the swap
    is recorded on the hotter member of each pair, so rung 0 reads ~0 by
    construction); the other moves are reported for the cold chain only.
    Returns {} when no stats groups are present.
    """
    files = sorted(glob.glob(os.path.join(pulsar_dir, 'epochs_*.h5')),
                   key=lambda x: int(re.search(r'epochs_(\d+)', x).group(1)))[burn_epochs:]
    acc_sum, cnt_sum, ntemps = None, None, None
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            key = list(f.keys())[0]
            if 'stats' not in f[key]:
                continue
            stats = f[key]['stats']['stats'][:]
            counts = f[key]['stats']['counts'][:]
            ntemps = len(f[key]['sampling']['temps'][:])
            if acc_sum is None:
                acc_sum, cnt_sum = stats, counts
            else:
                acc_sum, cnt_sum = acc_sum + stats, cnt_sum + counts
    if acc_sum is None or len(acc_sum) != 4:
        return {}
    rate = acc_sum / cnt_sum[:, None]          # (4 moves, nchains)
    cold = slice(0, None, ntemps)              # interleaved layout
    return {
        'tswap_per_temp': [round(float(rate[0, k::ntemps].mean()), 3)
                           for k in range(ntemps)],
        'stretch_cold': round(float(rate[1, cold].mean()), 3),
        'de_cold': round(float(rate[2, cold].mean()), 3),
        'rwalk_cold': round(float(rate[3, cold].mean()), 3),
    }


def group_of(short_name):
    """('NRT.NUPPI.1484', 'efac') for WN params, ('rednoise', 'gamma') for GPs."""
    base = re.sub(r'\[\d+\]$', '', short_name)
    for suf in ALL_SUFFIXES:
        if base == suf or base.endswith('_' + suf):
            return base[:-len(suf)].rstrip('_'), suf
    return base, base  # unrecognised: its own group


# ── plots (one PDF per pulsar each) ─────────────────────────────────────────

def plot_ll_trace(path, psr, ll, thin=10):
    fig, ax = plt.subplots(figsize=(9, 4))
    it = np.arange(ll.shape[1])[::thin]
    ax.plot(it, ll[:, ::thin].T, alpha=0.08, lw=0.4, color='steelblue')
    ax.set_title(f'{psr} — log-likelihood walker traces (post burn-in)')
    ax.set_xlabel('iteration'); ax.set_ylabel('log L')
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def plot_posteriors_1d(path, psr, short, samples):
    """One row of histograms per noise component (incl. per-backend WN)."""
    groups = {}
    for i, name in enumerate(short):
        g, _ = group_of(name)
        groups.setdefault(g if g else 'other', []).append(i)

    # vector params (freespectrum): one band panel instead of N histograms
    vec = {}
    for g, idxs in groups.items():
        vidx = [i for i in idxs if '[' in short[i]]
        if len(vidx) > 3:
            vec[g] = vidx
            groups[g] = [i for i in idxs if i not in vidx]

    nrows = sum(1 for g in groups if groups[g]) + len(vec)
    ncols = max([len(v) for v in groups.values() if v] + [1])
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False)

    row = 0
    for g in sorted(groups):
        if not groups[g]:
            continue
        for col, i in enumerate(sorted(groups[g], key=lambda i: short[i])):
            ax = axes[row][col]
            s = samples[:, i]
            ax.hist(s, bins=50, density=True, color='steelblue', alpha=0.75)
            med, lo, hi = np.percentile(s, [50, 16, 84])
            ax.axvline(med, color='crimson', lw=1.2)
            ax.axvspan(lo, hi, color='crimson', alpha=0.15)
            ax.set_title(short[i], fontsize=7)
            ax.tick_params(labelsize=6); ax.set_yticks([])
        for col in range(len(groups[g]), ncols):
            axes[row][col].set_visible(False)
        row += 1

    for g, vidx in vec.items():
        ax = axes[row][0]
        med = [np.percentile(samples[:, i], 50) for i in vidx]
        lo  = [np.percentile(samples[:, i], 16) for i in vidx]
        hi  = [np.percentile(samples[:, i], 84) for i in vidx]
        x = np.arange(len(vidx))
        ax.fill_between(x, lo, hi, color='steelblue', alpha=0.3)
        ax.plot(x, med, color='steelblue', lw=1.2)
        ax.set_title(f'{g} (free spectrum)', fontsize=8)
        ax.set_xlabel('bin'); ax.tick_params(labelsize=6)
        for col in range(1, ncols):
            axes[row][col].set_visible(False)
        row += 1

    fig.suptitle(psr, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path); plt.close(fig)


def plot_corner(path, psr, short, samples, suffixes, max_samples=20000,
                light=False):
    """Corner of the parameters whose suffix is in `suffixes`.

    light=True drops the scatter points and per-panel titles: needed for the
    white-noise corners of multi-backend pulsars (up to ~70 params), which
    would otherwise produce huge, slow PDFs.
    """
    idx = [i for i, n in enumerate(short)
           if group_of(n)[1] in suffixes and '[' not in n]
    if len(idx) < 2:
        return False
    sub = samples[:, idx]
    if sub.shape[0] > max_samples:
        sub = sub[np.random.default_rng(0).choice(sub.shape[0], max_samples, replace=False)]
    kwargs = dict(labels=[short[i] for i in idx],
                  label_kwargs={'fontsize': 8}, show_titles=True,
                  title_kwargs={'fontsize': 7})
    if light:
        kwargs.update(show_titles=False, plot_datapoints=False,
                      plot_density=True, label_kwargs={'fontsize': 5})
    fig = corner_module.corner(sub, **kwargs)
    fig.suptitle(psr, fontsize=12)
    fig.savefig(path); plt.close(fig)
    return True


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='SPNA posterior plots and diagnostics')
    ap.add_argument('--samples_dir', required=True,
                    help='Dataset directory (e.g. samples/check), one subdirectory per pulsar')
    ap.add_argument('--model', required=True,
                    help='Noise-model name: reads <samples_dir>/<pulsar>/<model>/')
    ap.add_argument('--plot_dir', default=None,
                    help='Output base directory (default: mirrors samples_dir under plot_output/)')
    ap.add_argument('--pulsar', default=None, help='Only this pulsar')
    ap.add_argument('--burn_epochs', type=int, default=3,
                    help='Epoch files to discard from the start (default: 3)')
    ap.add_argument('--burn_iters', type=int, default=0,
                    help='Extra iterations to discard after epoch burn-in (default: 0)')
    ap.add_argument('--thin', type=int, default=10, help='Thinning for trace plots')
    ap.add_argument('--skip_existing', action='store_true',
                    help='Skip pulsars that already have a summary.json in plot_dir '
                         '(resume an interrupted sweep without regenerating finished ones)')
    args = ap.parse_args()

    samples_dir = args.samples_dir.rstrip('/')
    if args.plot_dir:
        plot_dir = args.plot_dir
    else:
        # mirror the samples tree: samples/<dataset> -> plot_output/<dataset>
        parts = os.path.normpath(samples_dir).split(os.sep)
        if 'samples' in parts:
            parts[len(parts) - 1 - parts[::-1].index('samples')] = 'plot_output'
            plot_dir = os.sep.join(parts)
        else:
            plot_dir = os.path.join(samples_dir, 'plot_output')

    if corner_module is None:
        print('WARNING: corner package not available — corner plots will be skipped')

    pulsars = [args.pulsar] if args.pulsar else \
              sorted(d for d in os.listdir(samples_dir)
                     if os.path.isdir(os.path.join(samples_dir, d)) and d.startswith('J'))
    print(f'samples : {samples_dir}\nmodel   : {args.model}\nplots   : {plot_dir}\n'
          f'pulsars : {len(pulsars)}')

    for psr in pulsars:
        pdir = os.path.join(samples_dir, psr, args.model)
        if not os.path.isdir(pdir):
            print(f'{psr}: SKIP (no {args.model}/ run)')
            continue
        if args.skip_existing and os.path.exists(
                os.path.join(plot_dir, psr, args.model, 'summary.json')):
            print(f'{psr}: SKIP (already plotted, --skip_existing)')
            continue
        try:
            flat, short, lo, hi, constants = load_prior_layout(
                os.path.join(pdir, 'prior_dict.json'), psr)
            p, ll = load_chains(pdir, args.burn_epochs, args.burn_iters)
        except (FileNotFoundError, ValueError) as e:
            print(f'{psr}: SKIP ({e})')
            continue

        if p.shape[1] != len(flat):
            print(f'{psr}: SKIP (chain ndim {p.shape[1]} != prior layout {len(flat)})')
            continue

        # unit cube -> physical, then flatten walkers x iterations
        x = lo[None, :, None] + p * (hi - lo)[None, :, None]   # (nwalker, ndim, iters)
        samples = x.transpose(0, 2, 1).reshape(-1, len(flat))
        ll_flat = ll.reshape(-1)
        i_ml = int(np.argmax(ll_flat))

        # mixing diagnostic: in a well-mixed ensemble every walker has the same
        # mean logL (post burn); trapped walkers form strata with per-walker
        # means spread far beyond the within-walker fluctuations
        walker_means = ll.mean(axis=1)
        spread_between = float(np.std(walker_means))
        spread_within = float(np.mean(ll.std(axis=1)))
        mixing_ratio = spread_between / spread_within if spread_within > 0 else np.inf
        badly_mixed = mixing_ratio > 1.0

        # autocorrelation: one tau per parameter (each parameter is a
        # projection of the chain, and projections decorrelate at different
        # speeds); the slowest one bounds the effective sample count
        niters_kept = x.shape[2]
        taus = np.array([autocorr_time(x[:, i, :]) for i in range(len(flat))])
        # split R-hat across walkers, per parameter: catches walkers trapped
        # in different parameter-space regions even at similar logL levels
        nw_, nd_, ni_ = x.shape
        wmeans = x.mean(axis=2)
        W = x.var(axis=2, ddof=1).mean(axis=0)
        B = wmeans.var(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            rhat = np.sqrt((W * (ni_ - 1) / ni_ + B) / W)
        rhat_max = float(np.nanmax(rhat)) if np.isfinite(rhat).any() else None

        convergence = {'mixing_ratio': round(mixing_ratio, 3),
                       'badly_mixed': badly_mixed}
        if rhat_max is not None:
            convergence['rhat_max'] = round(rhat_max, 3)
            convergence['rhat_max_param'] = flat[int(np.nanargmax(rhat))]
        length_over_tau = None
        if np.isfinite(taus).any():
            i_slow = int(np.nanargmax(taus))
            tau_max = float(taus[i_slow])
            length_over_tau = niters_kept / tau_max
            convergence.update({
                'autocorr_time_max': round(tau_max, 1),
                'autocorr_time_max_param': flat[i_slow],
                'autocorr_time_median': round(float(np.nanmedian(taus)), 1),
                'n_effective_samples': int(x.shape[0] * niters_kept / tau_max),
                'chain_length_over_tau': round(length_over_tau, 1),
            })

        # Fourier component counts per GP, from the frozen noise model of the run
        fourier_components = {}
        nm_path = os.path.join(pdir, 'noise_model.json')
        if os.path.exists(nm_path):
            with open(nm_path) as f:
                nm = json.load(f)
            fourier_components = {name: int(gp['components'])
                                  for name, gp in nm.get('gps', {}).items()
                                  if 'components' in gp}

        # parameters piled up against their prior bounds (posterior mass
        # within 1% of the range from either edge): prior too narrow, or
        # unconstrained parameter
        span = hi - lo
        low_frac = (samples < lo + 0.01 * span).mean(axis=0)
        high_frac = (samples > hi - 0.01 * span).mean(axis=0)
        prior_railing = {}
        for i, name in enumerate(flat):
            for frac, side in ((low_frac[i], 'lower'), (high_frac[i], 'upper')):
                if frac > 0.05:
                    prior_railing[name] = {'edge': side, 'fraction': round(float(frac), 3)}

        acceptance = read_acceptance(pdir, args.burn_epochs)
        if acceptance:
            convergence['acceptance'] = acceptance

        run_meta = read_run_meta(pdir)
        run_meta['burn_epochs'] = args.burn_epochs
        run_meta['burn_iters'] = args.burn_iters

        summary = {'logL_max': float(ll_flat[i_ml]),
                   'n_params': len(flat),
                   'n_samples': int(samples.shape[0]),
                   'convergence': convergence,
                   'prior_railing': prior_railing,
                   'run': run_meta,
                   'fourier_components': fourier_components,
                   'constants': constants, 'params': {}}
        for i, name in enumerate(flat):
            med, q16, q84 = np.percentile(samples[:, i], [50, 16, 84])
            summary['params'][name] = {'median': float(med), 'q16': float(q16),
                                       'q84': float(q84), 'ml': float(samples[i_ml, i])}

        out = os.path.join(plot_dir, psr, args.model)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        plot_ll_trace(os.path.join(out, 'll_trace.pdf'), psr, ll, thin=args.thin)
        plot_posteriors_1d(os.path.join(out, 'posteriors_1d.pdf'), psr, short, samples)
        has_corner = corner_module is not None and \
            plot_corner(os.path.join(out, 'corner.pdf'), psr, short, samples,
                        suffixes=GP_SUFFIXES)
        if corner_module is not None:
            plot_corner(os.path.join(out, 'corner_wn.pdf'), psr, short, samples,
                        suffixes=WN_SUFFIXES, max_samples=10000,
                        light=len([n for n in short
                                   if group_of(n)[1] in WN_SUFFIXES]) > 12)
        warn = (f'  ⚠ BADLY MIXED (ratio {mixing_ratio:.1f}: groups of walkers stuck '
                f'at different logL levels — posteriors unreliable)') if badly_mixed else ''
        if length_over_tau is not None and length_over_tau < 50:
            warn += (f'  ⚠ SHORT CHAIN (length ≈ {length_over_tau:.0f}×tau, need ≳50: '
                     f'tau estimate itself uncertain — consider more epochs)')
        if rhat_max is not None and rhat_max > 1.1:
            warn += f'  ⚠ RHAT {rhat_max:.2f} ({convergence["rhat_max_param"]})'
        if prior_railing:
            worst = max(prior_railing.items(), key=lambda kv: kv[1]['fraction'])
            warn += (f'  ⚠ PRIOR RAILING: {len(prior_railing)} params '
                     f'(worst {worst[0]} {worst[1]["fraction"]:.0%} at {worst[1]["edge"]} edge)')
        neff = convergence.get('n_effective_samples')
        print(f'{psr}: {samples.shape[0]} samples'
              + (f' (n_eff≈{neff})' if neff else '')
              + f', {len(flat)} params, logL_max={ll_flat[i_ml]:.2f}'
              + ('' if has_corner else ' (no corner: <2 GP params)') + warn)

    print(f'done: {plot_dir}')


if __name__ == '__main__':
    main()
