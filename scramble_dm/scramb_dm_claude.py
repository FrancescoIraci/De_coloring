"""
Frequency-scrambling test for chromatic noise in pulsar timing residuals.

Model: Fourier components with chromatic index idx, amplitude ~ (nu_ref/nu)^idx
(RN: 0, DM: 2, Sv: 4), plus the timing model. The components in `scramble` get
randomly permuted observing frequencies, all others keep the true ones.
p-value = fraction of scrambles with lnL >= lnL with the true frequencies.

Usage:
  python scramb_dm.py --feathers 'data/*.feather' --custom-models custom_models.json \
      --outdir out --scramble DM --tm-marg gmat --niter 1000

custom_models.json: {"J1713+0747": {"RN": 30, "DM": 100, "Sv": null}, ...}
(number of Fourier frequencies per component; null or missing = not in the model)

tm_marg: 'gmat' : timing model marginalised with the G-matrix
         'fit'  : timing-model columns added to F (same lnL as 'gmat', cheaper)
         'none' : no timing model, as with G = identity
"""

import argparse, glob, json, os
from functools import partial

import numpy as np
import scipy.linalg as sl
import jax
jax.config.update("jax_enable_x64", True)   # float32 is not enough for TOAs/residuals
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHROM_IDX = {'RN': 0, 'DM': 2, 'Sv': 4}


def quantize(toas, dt=1.):
    """Epoch index of each TOA, same rule as discovery.signals.quantize."""
    isort = np.argsort(toas)
    bins = np.zeros(len(toas), dtype=int)
    b, v = 0, toas[isort[0]]
    for j in isort:
        if toas[j] - v > dt:
            v, b = toas[j], b + 1
        bins[j] = b
    return bins


def get_N(psr, noisedict, ecorr_dt=1.):
    """
    White-noise covariance: EFAC, EQUAD (t2equad or tnequad, from the key name), ECORR.
    Returns (N, has_ecorr): N is the diagonal (1D) without ECORR, the full N x N matrix with it.
    """
    backend_flags = np.asarray(psr.backend_flags)
    toaerrs = np.asarray(psr.toaerrs)
    white_var = np.zeros(len(toaerrs))
    ecorr_blocks = []

    for backend in np.unique(backend_flags):
        mask_backend = backend_flags == backend
        key = psr.name + '_' + backend + '_'
        efac = noisedict[key+'efac']
        if key+'log10_t2equad' in noisedict:
            white_var[mask_backend] = efac**2 * (toaerrs[mask_backend]**2 + 10**(2*noisedict[key+'log10_t2equad']))
        elif key+'log10_tnequad' in noisedict:
            white_var[mask_backend] = efac**2 * toaerrs[mask_backend]**2 + 10**(2*noisedict[key+'log10_tnequad'])
        else:
            white_var[mask_backend] = efac**2 * toaerrs[mask_backend]**2

        if key+'log10_ecorr' in noisedict:
            idx = np.where(mask_backend)[0]
            bins = quantize(np.asarray(psr.toas)[idx], ecorr_dt)
            ecorr2 = 10**(2*noisedict[key+'log10_ecorr'])
            ecorr_blocks += [(idx[bins == b], ecorr2) for b in np.unique(bins)]

    if not ecorr_blocks:
        return white_var, False
    N = np.diag(white_var)
    for epoch, ecorr2 in ecorr_blocks:
        N[np.ix_(epoch, epoch)] += ecorr2
    return N, True


def get_fourier_basis(toas, nfreq, Tspan):
    """sin and cos rows for f_k = k/Tspan, k = 1..nfreq, before the chromatic factor."""
    f = np.arange(1, nfreq+1) / Tspan
    F_sin = np.sin(2*np.pi*f[:, None] * toas[None, :])
    F_cos = np.cos(2*np.pi*f[:, None] * toas[None, :])
    return np.vstack((F_sin, F_cos))


@partial(jax.jit, static_argnames=('dense',))
def get_max_like(F, residuals, W, dense):
    """
    a_ml = (F N^-1 F^T)^-1 F N^-1 r   and   lnL_ml = -0.5 y N^-1 y,   y = r - a_ml F
    F has one row per basis function.

    W is the whitening matrix, W^T W = N^-1, so that F N^-1 F^T = F_w F_w^T etc.
    With the G-matrix, W^T W = G (G^T N G)^-1 G^T (the original GNG_inv, projected).
    W is 1D (1/sigma) unless dense=True.
    """
    F_w = F @ W.T if dense else F * W[None, :]
    r_w = W @ residuals if dense else W * residuals

    FNF = F_w @ F_w.T
    FNr = F_w @ r_w

    a_ml = jnp.linalg.solve(FNF, FNr)
    y = r_w - a_ml @ F_w
    lnL_ml = -0.5 * y @ y

    return a_ml, lnL_ml


def scramb_dm(psr, noisedict, custom_model, scramble=None, tm_marg='gmat', niter=1000,
              nu_ref=1400., seed=None, ecorr_dt=1., verbose=True):
    """
    custom_model : {'RN': 30, 'DM': 100, 'Sv': None}
    scramble     : components with permuted frequencies (default: all chromatic ones in the model)
    Returns lnL_scramble, lnL_ml_0, a_ml_0 (a_ml_0 rows: components in custom_model order, then TM).
    """
    toas = np.asarray(psr.toas, dtype=float)
    residuals = np.asarray(psr.residuals, dtype=float)
    freqs = np.asarray(psr.freqs, dtype=float)
    Mmat = np.asarray(psr.Mmat, dtype=float)
    Tspan = toas.max() - toas.min()

    comps = {c: n for c, n in custom_model.items() if n}
    if scramble is None:
        scramble = [c for c in comps if CHROM_IDX[c] != 0]
    scramble = [c for c in scramble if c in comps]
    if not scramble:
        raise ValueError('nothing to scramble: no requested component is in the model')

    # white noise and timing model: computed once, the same for every scramble
    N, has_ecorr = get_N(psr, noisedict, ecorr_dt)
    dense = (tm_marg == 'gmat') or has_ecorr

    # whitening matrix W, with W^T W = N^-1 (Cholesky instead of an explicit inverse)
    if tm_marg == 'gmat':
        n, m = Mmat.shape
        U, _, _ = np.linalg.svd(Mmat, full_matrices=True)
        G = U[:, m:]
        del U
        GNG = G.T @ (N @ G) if has_ecorr else G.T @ (N[:, None] * G)
        L = np.linalg.cholesky(GNG)                       # L L^T = G^T N G
        W = sl.solve_triangular(L, G.T, lower=True)       # W = L^-1 G^T
        del G, GNG, L
    elif has_ecorr:
        L = np.linalg.cholesky(N)
        W = sl.solve_triangular(L, np.eye(len(toas)), lower=True)   # W = L^-1
        del L
    else:
        W = 1. / np.sqrt(N)

    if tm_marg == 'fit':
        # orthonormal basis spanning the columns of Mmat: same fit, no scaling issues
        Um, _, _ = np.linalg.svd(Mmat, full_matrices=False)
        F_tm = Um.T
    else:
        F_tm = np.zeros((0, len(toas)))

    # Fourier rows (F_rn, F_dm, F_sv, ...) before the chromatic factor
    F_basis = {c: get_fourier_basis(toas, n, Tspan) for c, n in comps.items()}

    def get_F(freqs_scr):
        """Design matrix for one frequency assignment: each active component's
        Fourier rows times its own (nu_ref/freq)^idx chromatic factor -- using
        the SCRAMBLED frequencies for components under test, the TRUE ones for
        everything else -- stacked together with the timing model on top."""
        F_rows = []
        for c in comps:
            freqs_c = freqs_scr if c in scramble else freqs
            chrom_factor = (nu_ref / freqs_c) ** CHROM_IDX[c]
            F_rows.append(F_basis[c] * chrom_factor[None, :])
        F_rows.append(F_tm)
        return jnp.vstack(F_rows)

    residuals, W = jnp.asarray(residuals), jnp.asarray(W)
    if verbose:
        print('  ECORR included' if has_ecorr else '  no ECORR in noisedict')

    # true frequencies
    a_ml_0, lnL_ml_0 = get_max_like(get_F(freqs), residuals, W, dense)

    # scrambles: permutation, so the set of frequencies stays the same
    rng = np.random.default_rng(seed)
    lnL_scramble = []
    n_nan = 0
    for i in range(niter):
        if verbose:
            print(f'  nscramb {i+1}/{niter}', end='\r')
        freqs_scr = rng.permutation(freqs)
        a_ml, lnL_ml = get_max_like(get_F(freqs_scr), residuals, W, dense)
        if np.isfinite(lnL_ml):
            lnL_scramble.append(float(lnL_ml))
        else:
            n_nan += 1
    if verbose:
        print()
        if n_nan:
            print(f'  {n_nan} scrambles discarded (failed fit)')

    return np.array(lnL_scramble), float(lnL_ml_0), np.asarray(a_ml_0)


def p_value(lnL_ml_0, lnL_scramble):
    """Fraction of scrambles fitting at least as well as the true frequencies."""
    return (1 + np.sum(lnL_scramble >= lnL_ml_0)) / (1 + len(lnL_scramble))


def plot_scramble_hist(lnL_ml_0, lnL_scramble, psr_name, scramble, p, outpath):
    """Histogram of the scrambled-frequency lnL's, with the true-frequency
    lnL marked -- the actual evidence behind the p-value.

    A rare scramble can land on a near-singular frequency permutation and
    return a wildly wrong but technically finite lnL (seen in practice:
    -4.6e8 next to realistic values around -600) -- harmless to the
    rank-based p-value, but it would single-handedly set the histogram's
    x-range and crush every real bar into one pixel. Filtered from the
    *display* only (median-centered, 20x-IQR cutoff), with the excluded
    count noted on the plot; the p-value itself is computed upstream from
    the unfiltered array, as it should be.
    """
    med = np.median(lnL_scramble)
    iqr = np.percentile(lnL_scramble, 75) - np.percentile(lnL_scramble, 25)
    cutoff = max(20 * iqr, 1e-6)
    keep = np.abs(lnL_scramble - med) <= cutoff
    shown, n_excluded = lnL_scramble[keep], (~keep).sum()

    rank_pct = (lnL_scramble < lnL_ml_0).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(shown, bins=40, color="#6699CC", edgecolor="white", linewidth=0.6,
             alpha=0.85, label=f"scrambled\n(n={len(shown)}"
                                + (f", {n_excluded} excluded" if n_excluded else "") + ")")
    ax.axvline(lnL_ml_0, color="#CC3311", lw=2, ls="--", label="true")
    ax.set_xlabel("maximum-likelihood $\\ln L$")
    ax.set_ylabel("count")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.set_title(f"{psr_name}  --  scrambled: {', '.join(scramble)}\n"
                 f"p = {p:.3g}  (true value at the {rank_pct:.1f}th percentile)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Frequency-scrambling test for chromatic noise.')
    ap.add_argument('--feathers', required=True, help="glob, e.g. 'data/*.feather'")
    ap.add_argument('--custom-models', required=True, help='JSON with Fourier frequencies per component')
    ap.add_argument('--noisedict', help='JSON with white-noise values (default: the one in each feather)')
    ap.add_argument('--outdir', default='scramb_out')
    ap.add_argument('--scramble', nargs='+', help='components to scramble (default: all chromatic)')
    ap.add_argument('--tm-marg', default='gmat', choices=['gmat', 'fit', 'none'])
    ap.add_argument('--niter', type=int, default=1000)
    ap.add_argument('--nu-ref', type=float, default=1400.)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--psrs', nargs='+', help='only these pulsars')
    args = ap.parse_args()

    import discovery as ds

    custom_models = json.load(open(args.custom_models))
    noisedict_file = json.load(open(args.noisedict)) if args.noisedict else None
    os.makedirs(args.outdir, exist_ok=True)
    summary = []

    for path in sorted(glob.glob(args.feathers)):
        psr = ds.Pulsar.read_feather(path)
        if args.psrs and psr.name not in args.psrs:
            continue
        if psr.name not in custom_models:
            print(f'{psr.name}: not in {args.custom_models}, skipped')
            continue
        noisedict = noisedict_file if noisedict_file is not None else getattr(psr, 'noisedict', None)
        if not noisedict:
            print(f'{psr.name}: no noisedict, skipped')
            continue
        if getattr(psr, 'dmx', None):
            print(f'{psr.name}: WARNING, timing model contains DMX, which absorbs DM variations')

        custom_model = custom_models[psr.name]
        scramble = args.scramble or [c for c in custom_model if custom_model[c] and CHROM_IDX[c] != 0]
        print(f'{psr.name}  ({len(psr.toas)} TOAs)')

        lnL_scramble, lnL_ml_0, a_ml_0 = scramb_dm(psr, noisedict, custom_model, scramble,
                                                   args.tm_marg, args.niter, args.nu_ref, args.seed)
        p = p_value(lnL_ml_0, lnL_scramble)
        print(f'  p = {p:.4g}')
        summary.append((psr.name, p, lnL_ml_0, len(lnL_scramble)))

        tag = f'{psr.name}_{"-".join(scramble)}_{args.tm_marg}'
        np.savetxt(f'{args.outdir}/{tag}.txt', np.append(lnL_ml_0, lnL_scramble),
                   header='first value: lnL with true frequencies; then scrambles')
        plot_scramble_hist(lnL_ml_0, lnL_scramble, psr.name, scramble, p,
                            f'{args.outdir}/{tag}.png')

    with open(f'{args.outdir}/summary_{args.tm_marg}.txt', 'w') as f:
        f.write('# psr  p  lnL_ml_0  n_scrambles\n')
        for row in summary:
            f.write('{} {:.5g} {:.6f} {}\n'.format(*row))


if __name__ == '__main__':
    main()
