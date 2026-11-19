"""Build a single-pulsar discovery likelihood from a JSON noise-model file.

The JSON schema (see noise_models/*.json for examples):

{
  "description": "free text",
  "white_noise": {
      "sample":    true,       # sample efac/equad; if false, fix them
      "tnequad":   true,       # TempoNest equad convention
      "ecorr":     false,      # add per-backend ECORR GP
      "ecorr_selection": "all" # "all" or "lofar" (EPTA: ECORR only on LOFAR/NenuFAR);
                               # if no backend matches, ECORR is skipped with a warning
      "noisedict": null        # path to a JSON with fixed WN values
                               # (used only if sample=false; null -> feather noisedict)
  },
  "timing_model": {"svd": true},
  "gps": {
      "<name>": {"basis": "achromatic|dm|chromatic|solar",
                 "psd":   "powerlaw|brokenpowerlaw|freespectrum",
                 "components": 30,        # OR "min_period_yr": 1.5 — shortest
                                          # period in the basis, components =
                                          # Tspan / min_period_yr (per pulsar)
                 "fref": 1400.0}          # optional, dm/chromatic bases only
  },
  "deterministic": {"solar_wind": false},  # n_earth 1/r^2 solar-wind delay
  "priors": {
      "<regex-or-exact-name>": [min, max]  # uniform prior override
      "<regex-or-exact-name>": value       # scalar -> fix param to constant
  }
}

GP parameter names come out as <psr>_<name>_<arg>, e.g. J1909-3744_rednoise_log10_A.
The "chromatic" basis adds a free chromatic index <psr>_<name>_alpha (fix it via a
scalar prior override if you want e.g. alpha = 4).
"""

import re
import json
import inspect
import warnings
from functools import partial

import numpy as np
import jax.numpy as jnp

# Prefer the EPTA-patched discovery (LOFAR/NenuFAR ECORR integrated in
# makenoise_measurement); fall back to the stock package.
try:
    import discovery_new as ds
    from discovery_new import signals as ds_sig
    from discovery_new import solar as ds_solar
except ImportError:
    import discovery as ds
    from discovery import signals as ds_sig
    from discovery import solar as ds_solar

# the patched makenoise_measurement handles ECORR itself (Sherman-Morrison
# on observing epochs); stock discovery needs a separate makegp_ecorr GP
INTEGRATED_ECORR = 'ecorr_select' in inspect.signature(ds_sig.makenoise_measurement).parameters

BASES = {
    'achromatic': ds_sig.fourierbasis,
    'dm':         ds_sig.dmfourierbasis,        # alpha=2, baked in (see 'chromatic' below)
    'chromatic':  ds_sig.dmfourierbasis_alpha,   # FREE alpha (sampled, prior [1,7]) — needs
                                                 # the local discovery_new patch (parameter-
                                                 # dependent basis); a gp block with a fixed
                                                 # 'alpha' key below switches to the safe,
                                                 # unpatched fixed-alpha basis instead
    'solar':      ds_sig.dmfourierbasis_solar,
}

PSDS = {
    'powerlaw':       ds_sig.powerlaw,
    'brokenpowerlaw': ds_sig.brokenpowerlaw,
    'freespectrum':   ds_sig.freespectrum,
}

# EPTA convention: ECORR only on the wideband low-frequency receivers
LOFAR_BACKEND_RE = re.compile(r'(LOFAR|DE60[1-9]|NENUFAR)\.')

def selection_lofar_backends(psr):
    """Backend selection restricted to LOFAR/NenuFAR flags.

    Flags mapped to '' are dropped by discovery's per-backend machinery,
    so only the low-frequency backends get an ECORR slot.
    """
    flags = np.asarray(psr.backend_flags)
    mask = np.array([bool(LOFAR_BACKEND_RE.match(b)) for b in flags])
    return np.where(mask, flags, '')

ECORR_SELECTIONS = {
    'all':   ds_sig.selection_backend_flags,
    # the patched discovery ships its own lofar_backends (returns {flag: mask});
    # our flag-relabelling version is the fallback for stock discovery
    'lofar': getattr(ds_sig, 'lofar_backends', selection_lofar_backends),
}


def _has_ecorr_backends(psr, selection):
    """True if the selection yields at least one backend to put ECORR on."""
    flags = selection(psr)
    if isinstance(flags, dict):
        return len(flags) > 0
    return bool(np.any(np.asarray(flags) != ''))

# First match wins. The generic _log10_A/_gamma patterns exclude the
# solar-wind GP (negative lookahead): the solar basis carries different
# units, hence the very different ranges at the bottom. The SW entries rely
# on the GP being named 'swgp' in the noise-model JSON, as in the templates.
DEFAULT_PRIORS = {
    # white noise
    r".*_efac":                  [0.5, 5.0],
    r".*_t2equad":               [-9.0, -5.0],
    r".*_tnequad":               [-9.0, -5.0],
    r".*_log10_ecorr":           [-9.0, -2.0],
    # GPs (red noise, DM, chromatic, ...)
    r"(?!.*_swgp_).*_log10_A":   [-18.0, -11.0],
    r"(?!.*_swgp_).*_gamma":     [0.0, 7.0],
    r".*_log10_fb":              [-9.0, -6.5],
    r".*_alpha":                 [1.0, 7.0],
    r".*_log10_rho\(\d+\)":      [-9.0, -4.0],
    # solar wind
    r".*_swgp_log10_A":          [-12.0, 1.0],
    r".*_swgp_gamma":            [-6.0, 5.0],
    r".*_n_earth":               [0.0, 30.0],
}


def load_noise_model(path):
    with open(path, 'r') as f:
        return json.load(f)


# Empirical GPU-memory rule, calibrated on J1022+1001 (V100 32GB):
# the batched likelihood intermediates cost roughly
#   base = (ntoa + ecorr_epochs) * design_cols * 8   bytes per chain,
# and runs are safe while base * batch <= 4.5e9. Anchors from J1022
# (base = 95 MB/chain): batch 40 ran with a ~25 GiB compiled peak,
# 100 chains in a full vmap OOMed. Retune this constant for other GPUs.
EVAL_BATCH_SAFE_BYTES = 4.5e9


def suggest_eval_batch(psr, cfg, nchains):
    """Safe likelihood-evaluation chunk size for this pulsar and model.

    Returns None when the full ensemble fits in GPU memory (no chunking
    needed). The chunk size does not depend on nchains — more walkers just
    mean more sequential chunks — nchains only decides whether chunking is
    needed at all. Call after build_singlepulsar_likelihood, which resolves
    the per-GP component counts in cfg.
    """
    cols = psr.Mmat.shape[1]
    for gp in cfg.get('gps', {}).values():
        cols += 2 * int(gp['components'])

    epochs = 0
    wn = cfg.get('white_noise', {})
    if wn.get('ecorr', False) and hasattr(ds_sig, 'make_ecorr_Umat'):
        selection = ECORR_SELECTIONS[wn.get('ecorr_selection', 'all')]
        if _has_ecorr_backends(psr, selection):
            U = ds_sig.make_ecorr_Umat(psr, selection=selection)[0]
            epochs = 0 if U is None else U.shape[1]

    base = (len(psr.toas) + epochs) * cols * 8
    nmax = int(EVAL_BATCH_SAFE_BYTES // base)
    if nmax >= nchains:
        return None
    return max(10, (nmax // 10) * 10)


def build_singlepulsar_likelihood(psr, cfg):
    """Assemble a ds.PulsarLikelihood for one pulsar following the JSON config."""
    wn  = cfg.get('white_noise', {})
    tm  = cfg.get('timing_model', {})
    det = cfg.get('deterministic', {})
    tspan = ds_sig.getspan([psr])

    if wn.get('sample', True):
        noisedict = {}
    elif wn.get('noisedict'):
        with open(wn['noisedict'], 'r') as f:
            noisedict = json.load(f)
    else:
        noisedict = psr.noisedict

    ecorr_on = wn.get('ecorr', False)
    if ecorr_on:
        selection = ECORR_SELECTIONS[wn.get('ecorr_selection', 'all')]
        if not _has_ecorr_backends(psr, selection):
            warnings.warn(f"{psr.name}: ecorr requested but selection "
                          f"'{wn.get('ecorr_selection', 'all')}' matches no backend — skipping ECORR")
            ecorr_on = False

    if ecorr_on and INTEGRATED_ECORR:
        # patched discovery: ECORR lives inside the measurement-noise matrix;
        # fix_ecorr=False samples log10_ecorr, True takes it from the noisedict
        noise = ds_sig.makenoise_measurement(psr, noisedict, tnequad=wn.get('tnequad', True),
                                             ecorr=True, ecorr_select=selection,
                                             fix_ecorr=not wn.get('sample', True))
    else:
        noise = ds_sig.makenoise_measurement(psr, noisedict, tnequad=wn.get('tnequad', True))

    model = [
        psr.residuals,
        noise,
        ds_sig.makegp_timing(psr, svd=tm.get('svd', True)),
    ]

    # stock discovery: ECORR as a separate per-backend GP
    if ecorr_on and not INTEGRATED_ECORR:
        model.append(ds_sig.makegp_ecorr(psr, noisedict, selection=selection))

    for name, gp in cfg.get('gps', {}).items():
        gp_basis = gp.get('basis', 'achromatic')
        if gp_basis == 'chromatic' and 'alpha' in gp:
            # fixed chromatic index: same construction as 'dm' (Dm baked into
            # the basis matrix once, no sampled alpha, no parameter-dependent
            # basis) instead of the free-alpha ds_sig.dmfourierbasis_alpha
            basis = ds_sig.make_dmfourierbasis(alpha=float(gp['alpha']))
        else:
            basis = BASES[gp_basis]
        if 'fref' in gp:
            basis = partial(basis, fref=gp['fref'])
        psd = PSDS[gp.get('psd', 'powerlaw')]

        # number of Fourier components: either explicit, or derived from the
        # shortest period to include (components = Tspan / min_period_yr,
        # e.g. the EPTA solar-wind convention of stopping at 1/1.5yr)
        if 'components' in gp:
            ncomp = int(gp['components'])
        elif 'min_period_yr' in gp:
            tspan_yr = float(tspan) / (86400.0 * 365.25)
            ncomp = max(1, int(tspan_yr / float(gp['min_period_yr'])))
        else:
            raise KeyError(f"GP '{name}': specify either 'components' or 'min_period_yr'")
        gp['components'] = ncomp   # record the resolved count (frozen noise_model.json)

        model.append(ds_sig.makegp_fourier(psr, psd, ncomp,
                                           T=tspan, fourierbasis=basis, name=name))

    if det.get('solar_wind', False):
        model.append(ds_sig.makedelay(psr, ds_solar.make_solardm(psr), name='sw'))

    return ds.PulsarLikelihood(model)


def generate_priors(params, user_priors=None):
    """Map every likelihood parameter to a prior spec.

    user_priors (from the JSON "priors" block) win over DEFAULT_PRIORS;
    within each block the first matching pattern wins. A [min, max] pair
    gives a uniform prior, a scalar fixes the parameter to a constant.
    """
    merged = list((user_priors or {}).items()) + list(DEFAULT_PRIORS.items())
    priors = {}
    for name in params:
        for pattern, val in merged:
            if pattern == name or re.fullmatch(pattern, name):
                if isinstance(val, (int, float)):
                    priors[name] = {'dist': 'constant', 'value': float(val)}
                else:
                    priors[name] = {'dist': 'uniform',
                                    'min': float(val[0]), 'max': float(val[1])}
                break
        else:
            raise KeyError(f"No prior found for parameter '{name}': "
                           f"add it to the noise model 'priors' block")
    return priors


def _vector_size(name):
    """Size of a vector parameter like 'J1909-3744_rn_log10_rho(30)', else None."""
    m = re.search(r'\((\d+)\)$', name)
    return int(m.group(1)) if m else None


def build_transform(params_all, priors):
    """Flatten (possibly vector-valued) sampled params into a unit hypercube.

    Returns a dict with:
      params_sampled : names of sampled params (vector params count once)
      constants      : {name: jnp value} for fixed params
      ndim           : flattened dimensionality
      prior_transform: unit cube u -> physical flat vector x
      to_dict        : flat vector x -> params dict (constants NOT included)
    """
    params_sampled = [n for n in params_all if priors[n]['dist'] != 'constant']

    constants = {}
    for n in params_all:
        if priors[n]['dist'] == 'constant':
            size = _vector_size(n)
            v = priors[n]['value']
            constants[n] = jnp.full(size, v) if size else jnp.float64(v)

    layout, lbs, ubs = [], [], []
    offset = 0
    for n in params_sampled:
        size = _vector_size(n)          # None for scalars
        k = size if size else 1
        layout.append((n, offset, size))
        lbs += [priors[n]['min']] * k
        ubs += [priors[n]['max']] * k
        offset += k

    ndim = offset
    lbs, ubs = jnp.array(lbs), jnp.array(ubs)
    layout = tuple(layout)

    def prior_transform(u):
        return lbs + u * (ubs - lbs)

    def to_dict(x):
        return {n: (x[off] if size is None else x[off:off + size])
                for n, off, size in layout}

    return {'params_sampled': params_sampled, 'constants': constants,
            'ndim': ndim, 'prior_transform': prior_transform, 'to_dict': to_dict}
