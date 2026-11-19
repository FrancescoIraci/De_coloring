#!/bin/bash
# Regenerates every corner-plot comparison referenced in summary/sections/*.tex
# (each as one joint corner: RN and DM parameters together, rnonly runs only in
# the RN panels), plus one 3-station-set + rnonly corner per pulsar
# (<pulsar>_joint.pdf), each run with its own burn-in
# (DATASET:MODEL:BURN_EPOCHS, values chosen from the ll_trace.pdf inspections).
# Needs samples/, so run it on the cluster, from anywhere:
#   bash scripts/make_summary_compare_plots.sh
set -euo pipefail
cd "$(dirname "$0")/.."

C=feathers_combined_corrected
U=feathers_combined_uncorrected
HF=feathers_highfreq_corrected
HFU=feathers_highfreq_uncorrected
LO=feathers_lofaronly_corrected
LOU=feathers_lofaronly_uncorrected

cmp() {
    local psr=$1 name=$2
    shift 2
    python scripts/compare_posteriors.py --pulsar "$psr" --joint \
        --out "plot_output/compare/${name}.pdf" --runs "$@"
}

joint() {
    local psr=$1
    shift
    python scripts/compare_posteriors.py --pulsar "$psr" --joint --runs "$@"
}

# burn-in epochs per run: highfreq corrected (both models), lofaronly corrected
# dmrn / rnonly, combined corrected dmrn / rnonly, combined uncorrected dmrn /
# rnonly, lofaronly uncorrected (both models), highfreq uncorrected rnonly
common() {
    local p=$1 hf=$2 lod=$3 lor=$4 cd=$5 cr=$6 ud=$7 ur=$8 lou=$9 hfu=${10}
    cmp "$p" "$p"                                        $HF:rnonly:$hf $HF:dmrn:$hf
    cmp "$p" "${p}_lofaronly_corrected_rnonly_vs_dmrn"   $LO:rnonly:$lor $LO:dmrn:$lod
    cmp "$p" "${p}_corrected_dmrn_3way"                  $HF:dmrn:$hf $LO:dmrn:$lod $C:dmrn:$cd
    cmp "$p" "${p}_uncorrected_vs_corrected"             $U:dmrn:$ud $C:dmrn:$cd
    cmp "$p" "${p}_lofaronly_corr_vs_uncorr_dmrn"        $LO:dmrn:$lod $LOU:dmrn:$lou
    cmp "$p" "${p}_lofaronly_corr_vs_uncorr_rnonly"      $LO:rnonly:$lor $LOU:rnonly:$lou
    cmp "$p" "${p}_rnonly-corrected_vs_dmrn-uncorrected" $C:rnonly:$cr $U:dmrn:$ud
    cmp "$p" "${p}_lofaronly_uncorrected_rnonly_vs_dmrn" $LOU:rnonly:$lou $LOU:dmrn:$lou
    cmp "$p" "${p}_rnonly_highfreqU_vs_lofaronlyU_vs_combinedU" \
        $HFU:rnonly:$hfu $LOU:rnonly:$lou $U:rnonly:$ur
    joint "$p" $C:dmrn:$cd $HF:dmrn:$hf $LO:dmrn:$lod $C:rnonly:$cr
}

#        pulsar      hf lod lor cd cr ud ur lou hfu
common J1012+5307   4  7   4  12 19 8  8  4   4
common J1640+2224   4  4   4  4  4  4  4  4   4
common J1744-1134   4  6   6  16 16 12 12 6   4

cmp J1012+5307 J1012+5307_rnonly_combined_vs_highfreq $C:rnonly:19 $HF:rnonly:4
cmp J1012+5307 J1012+5307_corrected_rnonly_3way $HF:rnonly:4 $LO:rnonly:4 $C:rnonly:19
cmp J1640+2224 J1640+2224_corrected_rnonly_3way $HF:rnonly:4 $LO:rnonly:4 $C:rnonly:4
cmp J1744-1134 J1744-1134_corrected_rnonly_3way $HF:rnonly:4 $LO:rnonly:6 $C:rnonly:16
