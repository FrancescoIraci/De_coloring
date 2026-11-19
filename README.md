# De_coloring

DM(t) correction of pulsar timing data from LOFAR, and a test of whether a
Bayesian noise analysis still needs a stochastic DM term once the correction
is applied. Pulsars: J1012+5307, J1640+2224, J1744-1134.

The DM(t) curve is measured epoch by epoch from LOFAR, smoothed with a
Matérn-3/2 GP, and subtracted from the TOAs. Noise models (`noise_models/`):
`rnonly` (white noise + ECORR + achromatic red noise) and `dmrn` (`rnonly` +
DM GP), sampled with `spna_sampling.py` (discovery + jexplore) on the
CC-IN2P3 GPU nodes.

## Datasets

All restricted to the LOFAR time span, one feather per pulsar:

| directory | stations | DM(t) correction |
|---|---|---|
| `feathers_combined_{corrected,uncorrected}` | all | yes / no |
| `feathers_highfreq_{corrected,uncorrected}` | high frequency only (GMRT included) | yes / no |
| `feathers_lofaronly_{corrected,uncorrected}` | LOFAR only | yes / no |
| `feathers_lofaronly_corrected_mjd57000-60000` | LOFAR only, J1012+5307, MJD 57000-60000 | yes |

"Uncorrected" leaves DM1/DM2 free and NE_SW = 0.

## Layout

| path | content |
|---|---|
| `J*/` | timing inputs read by the pipeline (par, tim) and the two DM(t) plots used in the summary |
| `scripts/` | pipeline, noise analysis and plotting (see below) |
| `scripts/utils/` | noise-model builder for discovery |
| `noise_models/` | `rnonly.json`, `dmrn.json` |
| `plot_output/<dataset>/<pulsar>/<model>/` | per-run `summary.json` and `gp_timeseries.npz` |
| `plot_output/compare/` | comparison plots (corners, DM(t), red-noise time series) |
| `scramble_dm/` | frequency-scrambling test for chromatic noise |
| `summary/` | LaTeX summary (`summary.pdf`) |
| `notebooks/` | interactive exploration |

Not tracked: chains (`samples/`, on the cluster), per-run corner/trace plots,
and the per-pulsar working directories `J*/dm_correction_out/` (epoch-wise DM
cache, materialized tims, refit pars), all regenerable from the scripts.

## Workflow

1. **Feathers** (local, `igwn-py38-feathers` env with tempo2/libstempo/enterprise):
   `conda run -n igwn-py38-feathers python scripts/make_feathers_matched.py`
   builds the six datasets; `scripts/make_feather_mjdcut.py` builds an
   MJD-restricted variant with the timing model refit.
2. **Noise runs** (cluster): `scripts/run_spna.sh <feathers_dir> <noise_model.json> --pulsar <name>`
   via `sbatch` (`scripts/launch_template.sh` for array jobs).
3. **Posteriors** (cluster): `scripts/plot_posteriors.py` per run (traces,
   corners, `summary.json`); `scripts/make_summary_compare_plots.sh` for every
   comparison in the summary, with each run's burn-in.
4. **GP reconstructions** (cluster): `scripts/run_gp_timeseries.sh <dataset> <model> <pulsar> --burn_epochs N`.
5. **Local plots and summary**: `scripts/compare_dm_timeseries.py`,
   `scripts/compare_rn_timeseries.py`, `summary/make_color_key_thumbnails.py`,
   then `pdflatex summary.tex` (twice) in `summary/`.

Plot colors are colorblind-safe and fixed per dataset (`DATASET_STYLE` in
`scripts/compare_posteriors.py`); solid lines are DM+RN, dashed RN only.

## Provenance

Timing data from the EPTA deduplicated delivery (`no_duplicates/`). The
earlier history of this project, including the other nine DR2low pulsars
processed with the same pipeline, is in the archived repository.
