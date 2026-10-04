#!/bin/bash -l
#SBATCH --job-name=irrm-vdjdb-eval
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VALIDATION_ROOT="$ROOT_DIR/experiments/trb/vdjdb-data-anchor-validation"
COHORT_DIR="$VALIDATION_ROOT/cohort-preflight-1426341"
OUTPUT_DIR="$VALIDATION_ROOT/evaluation-${SLURM_JOB_ID}"
TOKENIZER="$ROOT_DIR/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
PYTHON_BIN="${RTP_CODEC_PYTHON:-$(command -v python)}"

test -x "$PYTHON_BIN"
for condition in r p rp; do
  test -f "$ROOT_DIR/experiments/trb/data-anchor-$condition-seed42/best.pt"
  test -f "$ROOT_DIR/experiments/trb/data-anchor-$condition-seed42/run_config.json"
done
test -f "$ROOT_DIR/experiments/trb/anchor-data-rtp-seed42/best.pt"
test -f "$COHORT_DIR/PREFLIGHT.json"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m rtp_codec.benchmarks.downstream.evaluate_vdjdb_frozen \
  --cohort "$COHORT_DIR/cohort.tsv" --cohort-preflight "$COHORT_DIR/PREFLIGHT.json" \
  --tokenizer "$TOKENIZER" --output-dir "$OUTPUT_DIR" \
  --model "r:$ROOT_DIR/experiments/trb/data-anchor-r-seed42/best.pt:$ROOT_DIR/experiments/trb/data-anchor-r-seed42/run_config.json" \
  --model "p:$ROOT_DIR/experiments/trb/data-anchor-p-seed42/best.pt:$ROOT_DIR/experiments/trb/data-anchor-p-seed42/run_config.json" \
  --model "rp:$ROOT_DIR/experiments/trb/data-anchor-rp-seed42/best.pt:$ROOT_DIR/experiments/trb/data-anchor-rp-seed42/run_config.json" \
  --model "rtp:$ROOT_DIR/experiments/trb/anchor-data-rtp-seed42/best.pt:$ROOT_DIR/experiments/trb/anchor-data-rtp-seed42/run_config.json" \
  --batch-size 128 --max-pairs-per-label 2000 --bootstrap 2000 --permutations 1000 --seed 42
