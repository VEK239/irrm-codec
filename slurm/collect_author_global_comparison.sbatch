#!/bin/bash
#SBATCH --job-name=irrm-author-global-final
#SBATCH --partition=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/final-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/final-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
OLD="$REPO/experiments/trb/external-model-benchmark"
BASE="$REPO/experiments/trb/external-global-author-v1"
VDJDB_EVAL="${1:?pass completed VDJdb evaluation path}"
OUT="$BASE/final-$SLURM_JOB_ID"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs"
cd "$REPO"

"$PYTHON" benchmark/collect_author_global_comparison.py --old-reconstruction-summary "$OLD/results/reconstruction/reconstruction_summary.csv" --old-pgen-summary "$OLD/results/pgen/pgen_summary.csv" --new-reconstruction-dir "$BASE/reconstruction" --new-pgen-dir "$BASE/pgen" --control-pgen-dir "$BASE/pgen-onehot-fixed" --reference-vdjdb-metrics "$REPO/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728/metrics.json" --external-vdjdb-summary "$VDJDB_EVAL/external_retrieval_summary.tsv" --output-dir "$OUT"
sha256sum "$OUT/final_author_global_comparison.tsv" "$OUT/final_author_global_comparison.md" "$OUT/MANIFEST.json" > "$OUT/HASHES.sha256"
echo "status=complete output=$OUT"
