#!/bin/bash
# LAUNCHER TEMPLATE — don't edit this file for day-to-day runs: copy it to a
# gitignored working launcher and tweak that one freely, e.g.
#   cp launch_template.sh launch_highfreq_dmrn.sh
#
# This script runs nothing itself, it only sbatch-es the actual job
# (run_spna.sh) with the right feathers dir + noise model. Our run matrix is
# small (3 pulsars x up to 4 datasets x 2-3 noise models), so this is meant
# to be copied once per (dataset, noise_model) combination you want to run,
# not driven by a config table like the original 60-pulsar campaign was.
#
#   ALL 3 pulsars (array job): leave NPSRS set -> sbatch --array=0-2, each
#     array task analyses a different pulsar (see run_spna.sh).
#   SINGLE pulsar: unset NPSRS and pass it via EXTRA_FLAGS:
#     NPSRS=""
#     EXTRA_FLAGS="--pulsar J1012+5307"

# >>> CONFIRM: where this repo (De_coloring) lives on the cluster <<<
PROJECT_DIR=/sps/lisaf/firaci/De_coloring

# one of: feathers_highfreq_corrected, feathers_lofaronly_corrected,
#         feathers_combined_corrected, feathers_combined_uncorrected
FEATHERS_DIR=${PROJECT_DIR}/feathers_highfreq_corrected

# path to a noise_models/*.json (NOT a bare name -- ours are standalone
# files, not per-pulsar registry JSONs)
NOISE_MODEL=${PROJECT_DIR}/noise_models/dmrn.json

NPSRS=$(ls ${FEATHERS_DIR} | wc -l)    # one array task per feather (= 3); unset for a single run
MAX_CONCURRENT=3                        # all 3 at once is fine at this scale
EXTRA_FLAGS=""                          # forwarded to spna_sampling.py (e.g. --nepoch 5 for a smoke test)

SCRIPTS_DIR="$(dirname "$0")"
LOGDIR=${PROJECT_DIR}/logs
mkdir -p ${LOGDIR}
LOG=${LOGDIR}/spna_%A_%a.log

ARRAY_OPT=""
if [ -n "${NPSRS}" ]; then
    ARRAY_OPT="--array=0-$((NPSRS - 1))"
    if [ -n "${MAX_CONCURRENT}" ]; then
        ARRAY_OPT="${ARRAY_OPT}%${MAX_CONCURRENT}"
    fi
fi

sbatch ${ARRAY_OPT} --output=${LOG} ${SCRIPTS_DIR}/run_spna.sh ${FEATHERS_DIR} ${NOISE_MODEL} ${EXTRA_FLAGS}
