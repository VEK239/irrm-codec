#!/bin/bash
#SBATCH --job-name=irrm-author-global-vdjdb
#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/vdjdb-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/vdjdb-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-global-author-v1"
COHORT="$REPO/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv"
REFERENCE="$REPO/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728"
GATE="${AUTHOR_GLOBAL_PREFLIGHT:-$BASE/preflight-1427529/PREFLIGHT.json}"
OUT="$BASE/vdjdb/evaluation-$SLURM_JOB_ID"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/vdjdb"
cd "$REPO"

"$PYTHON" benchmark/evaluate_vdjdb_author_global.py --cohort "$COHORT" --preflight "$GATE" --representations-dir "$BASE/vdjdb/representations" --reference-evaluation "$REFERENCE" --output-dir "$OUT"
sha256sum "$OUT/RESULTS.json" "$OUT/external_metrics.json" "$OUT/external_retrieval_summary.tsv" "$OUT/external_per_epitope.tsv" > "$OUT/HASHES.sha256"
echo "status=complete output=$OUT"
