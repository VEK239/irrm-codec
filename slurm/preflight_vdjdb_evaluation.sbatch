#!/bin/bash -l
#SBATCH --job-name=irrm-vdjdb-eval-gate
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VALIDATION_ROOT="$ROOT_DIR/experiments/trb/vdjdb-data-anchor-validation"
COHORT_DIR="$VALIDATION_ROOT/cohort-preflight-1426341"
PYTHON_BIN="${IRRM_PYTHON:-$(command -v python)}"
OUTPUT="$VALIDATION_ROOT/evaluation-code-preflight-${SLURM_JOB_ID}.json"

test -x "$PYTHON_BIN"
test -f "$COHORT_DIR/PREFLIGHT.json"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m scripts.preflight_vdjdb_evaluation \
  --cohort "$COHORT_DIR/cohort.tsv" \
  --cohort-preflight "$COHORT_DIR/PREFLIGHT.json" \
  --evaluation-script "$ROOT_DIR/benchmark/evaluate_vdjdb_frozen.py" \
  --output "$OUTPUT"
