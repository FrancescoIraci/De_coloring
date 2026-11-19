#!/bin/sh
# SLURM JOB: runs spna_sampling.py on a GPU node inside the singularity
# container. Adapted from EPTADR3/jexplore-noise/scripts/run_spna_sampling.sh
# for the De_coloring DM(t)-correction test (3 pulsars, 4 feather datasets,
# 2+ noise models) -- NOT the full 60-pulsar DR3 campaign, so the
# --config_table machinery of the original is dropped; one pulsar per job,
# passed via --pulsar in EXTRA_FLAGS or resolved from the array index.
#
# Positional arguments: $1 = feathers directory, $2 = noise-model JSON path
# (an explicit path, e.g. .../noise_models/dmrn.json -- our noise_models/*.json
# are standalone files, not per-pulsar registry JSONs, so always pass a path
# ending in .json, never a bare NAME); everything after those is forwarded
# verbatim to spna_sampling.py.
#
# >>> CONFIRM BEFORE RUNNING <<<
# PROJECT_DIR, the container path, and the bind mounts below are copied from
# the working jexplore-noise setup as of the last check (2026-10) -- confirm
# they still hold (discovery_new/jexplore_patched paths especially; those are
# NOT git-tracked, so cloning this repo does not bring them along) and adjust
# the SBATCH directives (--mail-user, --partition, --mem, --time) as needed.
#SBATCH --job-name=spna_dmcorr
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=firaci@apc.in2p3.fr
#SBATCH --mail-type=END,FAIL

FEATHERS_DIR=$1
NOISE_MODEL=$2      # path to a noise_models/*.json (ends in .json -> used directly)
shift 2
EXTRA_FLAGS="$@"    # forwarded to spna_sampling.py (e.g. --pulsar J1012+5307 --nepoch 20)

# >>> CONFIRM: where this repo (De_coloring) lives on the cluster <<<
PROJECT_DIR=/sps/lisaf/firaci/De_coloring
SCRIPT=${PROJECT_DIR}/scripts/spna_sampling.py

# >>> CONFIRM: discovery_new / jexplore_patched locations (not git-tracked) <<<
EPTA_DIR=/sps/lisaf/firaci/epta-dr3   # holds discovery_new/ and jexplore_patched/

# outputs: samples/<dataset>/<pulsar>/<label>/ -- <dataset> = feathers dir's
# own basename (e.g. feathers_highfreq_corrected), so each of our 4 datasets
# keeps its own sample tree
DATASET=$(basename ${FEATHERS_DIR})
OUTDIR_BASE=${PROJECT_DIR}/samples/${DATASET}

# Array mode: map the array index to the N-th feather in alphabetical order.
# Outside an array job SLURM_ARRAY_TASK_ID is unset and the pulsar must come
# from EXTRA_FLAGS (--pulsar ...); spna_sampling.py errors out otherwise.
PULSAR_FLAG=""
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    PULSAR_NAME=$(ls ${FEATHERS_DIR} | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
    PULSAR_FLAG="--pulsar ${PULSAR_NAME}"
    echo "Array task  : ${SLURM_ARRAY_TASK_ID}  ->  pulsar: ${PULSAR_NAME}"
else
    PULSAR_NAME=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--pulsar[= ]\([^ ]*\).*/\1/p')
fi

# label = noise-model basename unless --label overrides it in EXTRA_FLAGS
# (same model, different sampler settings)
RUN_LABEL=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--label[= ]\([^ ]*\).*/\1/p')
RUN_LABEL=${RUN_LABEL:-$(basename ${NOISE_MODEL} .json)}

# From here on, log next to the samples: samples/<dataset>/<pulsar>/<label>/slurm_<jobid>.log
if [ -n "${PULSAR_NAME}" ]; then
    RUN_DIR=${OUTDIR_BASE}/${PULSAR_NAME}/${RUN_LABEL}
    mkdir -p ${RUN_DIR}
    echo "log continues in ${RUN_DIR}/slurm_${SLURM_JOB_ID}.log"
    exec >> ${RUN_DIR}/slurm_${SLURM_JOB_ID}.log 2>&1
fi

NWALKER=300
NITERS=250
NEPOCH=20
SEED=13

echo "========================================="
echo "SLURM job   : ${SLURM_JOB_ID}"
echo "feathers    : ${FEATHERS_DIR}"
echo "noise model : ${NOISE_MODEL}"
echo "script      : ${SCRIPT}"
echo "nwalker     : ${NWALKER}"
echo "niters      : ${NITERS}"
echo "nepoch      : ${NEPOCH}"
echo "seed        : ${SEED}"
echo "extra flags : ${EXTRA_FLAGS}"
echo "========================================="

date
singularity cache clean --force

singularity exec --nv \
  --bind /sps/lisaf/smanzini/,/sps/lisaf/firaci/,/pbs/home/f/firaci/ \
  --bind /sps/lisaf/smanzini/python_extra/astropy/tests:/usr/local/lib/python3.13/dist-packages/astropy/tests \
  --bind /usr/local/cuda-13.1/targets/x86_64-linux/lib:/usr/local/cuda/lib64 \
  /sps/lisaf/smanzini/discovery_results/discovery_jax_gpu.sif \
  bash -c "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH

    pip install six --break-system-packages --quiet
    # EPTA_DIR first so 'import discovery_new' picks up the patched fork
    # (model_builder falls back to the container's stock discovery otherwise);
    # jexplore_patched shadows the container's jexplore (chunked likelihood
    # evaluation, identical behaviour when JEXPLORE_EVAL_BATCH is unset)
    export PYTHONPATH=${PROJECT_DIR}/scripts/utils:${EPTA_DIR}/jexplore_patched:${EPTA_DIR}:\$PYTHONPATH
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

    echo '=== JAX backend check ==='
    python -c \"import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())\"

    python ${SCRIPT} \
        --feathers_path ${FEATHERS_DIR} \
        --noise_model   ${NOISE_MODEL}  \
        --outdir_base   ${OUTDIR_BASE}  \
        --nwalker ${NWALKER} \
        --niters  ${NITERS}  \
        --nepoch  ${NEPOCH}  \
        --seed    ${SEED}    \
        ${PULSAR_FLAG}       \
        ${EXTRA_FLAGS}
  "

date
