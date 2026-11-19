#!/bin/sh
# Implied-GP time series (scripts/plot_gp_timeseries.py), inside the same
# singularity container as run_spna.sh but CPU-only (one Woodbury solve per
# draw -- no GPU needed, no gpu_v100 partition). Reconstructs the DM(t) (and
# red-noise) curve implied by a finished SPNA run's posterior, to compare
# against the epoch-wise DM time series the correction pipeline measured
# directly from LOFAR (dm_correction_out/*.png).
#
# Can run interactively on the login node (no sbatch needed, it's fast) or
# batched:
#   ./scripts/run_gp_timeseries.sh feathers_combined_corrected dmrn J1012+5307
#   sbatch scripts/run_gp_timeseries.sh feathers_combined_corrected dmrn
#     (no pulsar -> all pulsars found in that dataset's sample tree)
#
# Positional arguments: $1 = feathers directory (= dataset name, matches the
# sample tree under samples/), $2 = noise-model label (matches the run's
# subdirectory name, e.g. dmrn/rnonly -- NOT a path this time, plot_gp_
# timeseries.py reads samples/<dataset>/<pulsar>/<model>/noise_model.json
# itself), $3 = optional single pulsar (default: every pulsar with that run).
# Rest forwarded as EXTRA_FLAGS (--burn_epochs, --ndraws, --fref, ...).
#
# >>> CONFIRM BEFORE RUNNING <<< same paths as run_spna.sh
#SBATCH --job-name=gpts_dmcorr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --mail-user=firaci@apc.in2p3.fr
#SBATCH --mail-type=END,FAIL

FEATHERS_DIR=$1
MODEL=$2
shift 2
PULSAR_FLAG=""
case "$1" in
    J*) PULSAR_FLAG="--pulsar $1"; shift ;;
esac
EXTRA_FLAGS="$@"

# >>> CONFIRM: where this repo (De_coloring) lives on the cluster <<<
PROJECT_DIR=/sps/lisaf/firaci/De_coloring
SCRIPT=${PROJECT_DIR}/scripts/plot_gp_timeseries.py

# >>> CONFIRM: discovery_new location (not git-tracked) <<<
EPTA_DIR=/sps/lisaf/firaci/epta-dr3

SAMPLES_DIR=${PROJECT_DIR}/samples/${FEATHERS_DIR}
FEATHERS_PATH=${PROJECT_DIR}/${FEATHERS_DIR}

echo "========================================="
echo "feathers    : ${FEATHERS_PATH}"
echo "samples_dir : ${SAMPLES_DIR}"
echo "model       : ${MODEL}"
echo "pulsar      : ${PULSAR_FLAG:-all}"
echo "extra flags : ${EXTRA_FLAGS}"
echo "========================================="

date

singularity exec \
  --bind /sps/lisaf/smanzini/,/sps/lisaf/firaci/,/pbs/home/f/firaci/ \
  --bind /sps/lisaf/smanzini/python_extra/astropy/tests:/usr/local/lib/python3.13/dist-packages/astropy/tests \
  /sps/lisaf/smanzini/discovery_results/discovery_jax_gpu.sif \
  bash -c "
    pip install six --break-system-packages --quiet
    export MPLCONFIGDIR=/tmp/mpl_\$USER
    export JAX_PLATFORMS=cpu
    # EPTA_DIR first so 'import discovery_new' picks up the patched fork;
    # scripts/ for plot_posteriors.py (imported by plot_gp_timeseries.py),
    # scripts/utils for model_builder.py
    export PYTHONPATH=${PROJECT_DIR}/scripts:${PROJECT_DIR}/scripts/utils:${EPTA_DIR}:\$PYTHONPATH

    python ${SCRIPT} \
        --samples_dir ${SAMPLES_DIR} \
        --model ${MODEL} \
        --feathers_path ${FEATHERS_PATH} \
        ${PULSAR_FLAG} \
        ${EXTRA_FLAGS}
  "

date
