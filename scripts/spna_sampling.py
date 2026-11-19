"""Single-pulsar noise analysis with discovery + jexplore.

Workflow: read one feather file, build the likelihood from a JSON noise-model
definition (see noise_models/), sample with the jexplore PT ensemble sampler.
No injections: this is meant for real data.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import json
import time
import argparse

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import model_builder as mb
from model_builder import ds   # discovery_new if on PYTHONPATH, stock discovery otherwise
import my_rwalk

import jexplore.tools.distributions as d
from jexplore.backends import DefaultBackend
from jexplore.sampler import JaxSampler, Steps
from jexplore.sampling import EpochMH, SamplingMH
from jexplore.steps.tswap import TSwap
from jexplore.steps.stretch import Stretch
from jexplore.steps.de import DEStep

import jaxlib
import jexplore
print(f"JAX {jax.__version__} | JAXLib {jaxlib.__version__}")
print(f"Devices: {jax.devices()}")
print(f"discovery package: {ds.__name__} ({os.path.dirname(ds.__file__)}) | integrated ecorr: {mb.INTEGRATED_ECORR}")
print(f"jexplore package: {os.path.dirname(jexplore.__file__)}")

# ── Configuration ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='jexplore single-pulsar noise analysis')

parser.add_argument('--feathers_path', required=True,
                    help='Directory containing the feather files')
parser.add_argument('--pulsar', required=True,
                    help='Pulsar name to analyse (e.g. J1909-3744)')
parser.add_argument('--noise_model', required=True,
                    help='Noise-model NAME: reads <outdir_base>/<pulsar>/<name>.json '
                         '(one JSON per pulsar, written by hand). A value ending '
                         'in .json is treated as an explicit file path instead.')

parser.add_argument('--outdir', default=None,
                    help='Full output directory. If omitted, derived as '
                         '<outdir_base>/<pulsar>/<label>/')
parser.add_argument('--outdir_base', default='samples',
                    help='Base output directory, i.e. the dataset directory '
                         '(default: ./samples)')
parser.add_argument('--label', default=None,
                    help='Run subdirectory name (default: noise-model name)')

parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Delete epoch files left in the output directory by a previous '
                         'run (default: refuse to start, mixing runs corrupts the chains)')
parser.add_argument('--eval_batch', default='auto',
                    help="Likelihood-evaluation chunk size (bounds GPU memory; needs the "
                         "patched jexplore). 'auto' (default): derived per pulsar from the "
                         "calibrated memory rule; an integer: fixed chunk; 'off': full vmap.")

parser.add_argument('--nwalker',  type=int,   default=300,        help='Walkers per temperature (default: 300)')
parser.add_argument('--ntemps',   type=int,   default=5,          help='Number of temperatures (default: 5)')
parser.add_argument('--temp_max', type=float, default=np.sqrt(3), help='Max temperature (default: sqrt(3))')
parser.add_argument('--niters',   type=int,   default=250,        help='Iterations per epoch (default: 250)')
parser.add_argument('--nepoch',   type=int,   default=10,         help='Number of epochs (default: 10)')
parser.add_argument('--seed',     type=int,   default=13,         help='Sampler seed (default: 13)')
parser.add_argument('--init_key', type=int,   default=42,         help='JAX key for initial points (default: 42)')

args = parser.parse_args()

label   = args.label or os.path.splitext(os.path.basename(args.noise_model))[0]
psr_dir = os.path.join(args.outdir_base, args.pulsar)

# per-pulsar noise model: samples/<dataset>/<pulsar>/<name>.json
if args.noise_model.endswith('.json'):
    noise_model_path = args.noise_model      # explicit file (e.g. local tests)
else:
    noise_model_path = os.path.join(psr_dir, f'{args.noise_model}.json')
    if not os.path.exists(noise_model_path):
        available = sorted(f for f in os.listdir(psr_dir) if f.endswith('.json')) \
                    if os.path.isdir(psr_dir) else []
        raise FileNotFoundError(
            f"No noise model '{args.noise_model}.json' in {psr_dir}. "
            f"Available there: {available or 'none — write the JSON first'}")

outdir = args.outdir or os.path.join(psr_dir, label)
os.makedirs(outdir, exist_ok=True)

# never write into the chains of a previous run: the h5 backend would either
# crash on existing datasets or silently mix incompatible ensembles
existing = sorted(f for f in os.listdir(outdir)
                  if f.startswith('epochs_') and f.endswith('.h5'))
if existing:
    if args.overwrite:
        for f in existing:
            os.remove(os.path.join(outdir, f))
        print(f"--overwrite: removed {len(existing)} epoch files from a previous run")
    else:
        raise FileExistsError(
            f"{outdir} already contains {len(existing)} epoch files from a previous "
            f"run. Move them away or rerun with --overwrite.")

print(f"feathers    : {args.feathers_path}")
print(f"pulsar      : {args.pulsar}")
print(f"noise model : {noise_model_path}")
print(f"outdir      : {outdir}")

# ── Load data and noise model ───────────────────────────────────────────────
feather_file = os.path.join(args.feathers_path, args.pulsar)
if not os.path.exists(feather_file):
    available = sorted(f for f in os.listdir(args.feathers_path) if f.startswith('J'))
    raise ValueError(f"Pulsar '{args.pulsar}' not found in {args.feathers_path}. "
                     f"Available: {available}")

psr = ds.pulsar.Pulsar.read_feather(feather_file)
print(f"Loaded {psr.name}: {len(psr.toas)} TOAs, "
      f"Tspan = {(psr.toas.max() - psr.toas.min()) / 86400 / 365.25:.1f} yr")

cfg = mb.load_noise_model(noise_model_path)
if cfg.get('description'):
    print(f"model       : {cfg['description']}")

# ── Build likelihood and priors ─────────────────────────────────────────────
psl   = mb.build_singlepulsar_likelihood(psr, cfg)
jlogl = jax.jit(psl.logL)
params_all = psl.logL.params
print(f"Model parameters ({len(params_all)}): {params_all}")

priors_all = mb.generate_priors(params_all, cfg.get('priors'))

tr = mb.build_transform(params_all, priors_all)
params_sampled  = tr['params_sampled']
constants       = tr['constants']
ndim            = tr['ndim']
prior_transform = tr['prior_transform']
to_dict         = tr['to_dict']
print(f"Sampling {ndim} dimensions ({len(params_sampled)} parameters), "
      f"fixing {len(constants)}: {list(constants.keys())}")

# likelihood-evaluation chunk (GPU memory guard) — needs cfg with resolved
# component counts, hence after build_singlepulsar_likelihood
nchains_total = args.ntemps * args.nwalker
if args.eval_batch == 'auto':
    eval_batch = mb.suggest_eval_batch(psr, cfg, nchains_total)
elif str(args.eval_batch).lower() in ('off', '0'):
    eval_batch = None
else:
    eval_batch = int(args.eval_batch)
if eval_batch:
    os.environ["JEXPLORE_EVAL_BATCH"] = str(eval_batch)
mode = 'auto' if args.eval_batch == 'auto' else 'manual'
print(f"Eval batch: {eval_batch if eval_batch else 'off (full vmap)'} "
      f"[{mode}, {nchains_total} chains]")

# provenance: priors actually used + the resolved noise model (component
# counts derived from min_period_yr are filled in by the model builder)
with open(os.path.join(outdir, 'prior_dict.json'), 'w') as f:
    json.dump(priors_all, f, indent=4)
with open(os.path.join(outdir, 'noise_model.json'), 'w') as f:
    json.dump(cfg, f, indent=4)

# ── Likelihood wrapper for sampler ──────────────────────────────────────────
@jax.jit
def loglike_single(u):
    params = to_dict(prior_transform(u))
    params.update(constants)
    logL = jlogl(params)
    return jnp.where(jnp.isfinite(logL), logL, -1e30)

# warm-up + timing
key = jax.random.key(args.init_key)
key, subkey = jax.random.split(key)
u_test = jax.random.uniform(subkey, shape=(ndim,), minval=1e-6, maxval=1 - 1e-6)
_ = loglike_single(u_test)
t0 = time.time()
loglike_single(u_test).block_until_ready()
print(f"Likelihood eval time (JIT): {time.time() - t0:.4f} s")

# ── Sampler setup ───────────────────────────────────────────────────────────
temps = jnp.geomspace(1, args.temp_max, args.ntemps)
print(f"Temperatures: {temps}")

class GaussianRandomWalkFixed(my_rwalk.MVRandomWalk):
    def __init__(self, mask=None):
        super().__init__(mask=mask, dist=d.Normal)

    def step(self, key, state):
        key, prop, qxy = self.proposal(key, state)
        new_state, acc = self.mh(key, state, prop, self.beta, qxy)
        return new_state, acc.astype(jnp.float64)

sampler_state = SamplingMH(
    dim=ndim, nwalker=args.nwalker, temps=temps,
    loglik=loglike_single,
    logprior=d.Uniform(dim=ndim, minval=0, maxval=1).leval,
)
sampler_steps = Steps([
    {TSwap(permute=True).builder:       1.0},
    {Stretch(permute=True).builder:     0.3},
    {DEStep(permute=True).builder:      0.7},
    {GaussianRandomWalkFixed().builder: 0.3},
])
sampler_backend = DefaultBackend(burn=0, outdir=outdir, epochs_per_file=1)

# ── Initial conditions: overdispersed draws from the prior ─────────────────
key, subkey = jax.random.split(key)
initial_points = jax.random.uniform(subkey, shape=(args.ntemps * args.nwalker, ndim),
                                    minval=1e-6, maxval=1 - 1e-6)
print(f"Initial points shape: {initial_points.shape}")
print(f"loglike at first initial point: {loglike_single(initial_points[0]):.2f}")

iepoch = EpochMH({"p": initial_points})

# ── Run ─────────────────────────────────────────────────────────────────────
sampler = JaxSampler(sampler_state, sampler_steps, sampler_backend)

print(f"Sampling: {args.nepoch} epochs x {args.niters} iterations, "
      f"{args.ntemps} temps x {args.nwalker} walkers ...", flush=True)
t_run = time.time()
fepoch = sampler.run(iepoch, niters=args.niters, nepoch=args.nepoch, seed=args.seed)
elapsed = time.time() - t_run

epoch_files = sorted(f for f in os.listdir(outdir) if f.startswith('epochs_'))
print(f"Sampling done in {elapsed / 60:.1f} min "
      f"({elapsed / (args.nepoch * args.niters):.2f} s/iter avg)")
print(f"Wrote {len(epoch_files)} epoch files to {outdir}")
