"""Six small single-color frequency-vs-MJD thumbnails per pulsar, one per
(dataset, model) case -- a real-data shape tinted in that case's color,
meant to sit next to that pulsar's own corner-plot comparisons so the
color key is pulsar-specific, not one generic example borrowed from
elsewhere. Frequency coverage only depends on the dataset (not the noise
model), so the two model columns of the same dataset row share the same
shape, just a different tint -- and "combined, corrected" / "combined,
uncorrected" share the same shape too (the correction doesn't drop TOAs,
just adjusts residuals); only "highfreq" actually differs (no LOFAR/NenuFAR).

No axis labels/ticks -- this is a compact key, not an analysis plot.

Local only, discovery-local env:
    /opt/anaconda3/envs/discovery-local/bin/python3 summary/make_color_key_thumbnails.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import discovery as ds

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "assets")

sys.path.insert(0, os.path.join(REPO, "scripts"))
from compare_posteriors import case_style  # noqa: E402

PULSARS = ["J1012+5307", "J1640+2224", "J1744-1134"]

# (output name, feather dataset, model) -- colors and line styles come from
# case_style in scripts/compare_posteriors.py, so the key always matches
# the plots; the frame is dashed for RN-only, like the plots' contours
CASES = [
    ("caseCorrDMRN",   "feathers_combined_corrected",    "dmrn"),
    ("caseHFDMRN",     "feathers_highfreq_corrected",    "dmrn"),
    ("caseUncDMRN",    "feathers_combined_uncorrected",  "dmrn"),
    ("caseCorrRN",     "feathers_combined_corrected",    "rnonly"),
    ("caseHFRN",       "feathers_highfreq_corrected",    "rnonly"),
    ("caseUncRN",      "feathers_combined_uncorrected",  "rnonly"),
    ("caseLoCorrDMRN", "feathers_lofaronly_corrected",   "dmrn"),
    ("caseLoUncDMRN",  "feathers_lofaronly_uncorrected", "dmrn"),
    ("caseLoCorrRN",   "feathers_lofaronly_corrected",   "rnonly"),
    ("caseLoUncRN",    "feathers_lofaronly_uncorrected", "rnonly"),
    ("caseHFURN",      "feathers_highfreq_uncorrected",  "rnonly"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for pulsar in PULSARS:
        cache = {}
        for name, dataset, model in CASES:
            color, _, ls = case_style((dataset, model), 0)
            if dataset not in cache:
                psr = ds.Pulsar.read_feather(os.path.join(REPO, dataset, pulsar))
                cache[dataset] = (np.asarray(psr.toas) / 86400.0, np.asarray(psr.freqs))
            t, f = cache[dataset]

            fig, ax = plt.subplots(figsize=(1.8, 1.05))
            ax.scatter(t, f, s=2.2, c=color, linewidths=0, alpha=0.85)
            ax.set_yscale("log")
            # set_xticks([])/set_yticks([]) only clear the MAJOR ticks --
            # on a log axis with a narrow range (e.g. highfreq-only, under
            # a decade), matplotlib falls back to labelled MINOR ticks so
            # the axis isn't blank, which set_yticks([]) doesn't touch.
            # tick_params(which="both") kills both, unconditionally.
            ax.tick_params(axis="both", which="both", length=0,
                            labelbottom=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_linestyle(ls)
                spine.set_edgecolor(color)
            fig.tight_layout(pad=0.05)

            out_path = os.path.join(OUT, f"key_{pulsar}_{name}.pdf")
            fig.savefig(out_path)
            plt.close(fig)
            print(f"{pulsar}/{name} ({dataset}): {len(t)} TOAs -> {out_path}")


if __name__ == "__main__":
    main()
