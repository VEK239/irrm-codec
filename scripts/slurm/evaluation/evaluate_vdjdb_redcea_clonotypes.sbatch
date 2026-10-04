#!/bin/bash -l
#SBATCH --job-name=irrm-redcea-eval
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PREFLIGHT_ROOT="${1:?Preflight job directory is required}"
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$ROOT/experiments/trb/vdjdb-redcea-clonotypes"
OUTPUT="$BASE/evaluation-${SLURM_JOB_ID}"
TOKENIZER="$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
PYTHON="${RTP_CODEC_PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

test -x "$PYTHON"
test -f "$PREFLIGHT_ROOT/cohort/PREFLIGHT.json"
test -f "$PREFLIGHT_ROOT/cohort/cohort.tsv"
test -f "$TOKENIZER"
test ! -e "$OUTPUT"
cd "$ROOT"
"$PYTHON" -m rtp_codec.benchmarks.downstream.evaluate_vdjdb_redcea_clonotypes \
  --cohort "$PREFLIGHT_ROOT/cohort/cohort.tsv" \
  --preflight "$PREFLIGHT_ROOT/cohort/PREFLIGHT.json" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUTPUT" \
  --model "r:$ROOT/experiments/trb/data-anchor-r-seed42/best.pt:$ROOT/experiments/trb/data-anchor-r-seed42/run_config.json" \
  --model "t:$ROOT/experiments/trb/data-anchor-t-seed42/best.pt:$ROOT/experiments/trb/data-anchor-t-seed42/run_config.json" \
  --model "p:$ROOT/experiments/trb/data-anchor-p-seed42/best.pt:$ROOT/experiments/trb/data-anchor-p-seed42/run_config.json" \
  --model "rt:$ROOT/experiments/trb/data-anchor-rt-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rt-seed42/run_config.json" \
  --model "rp:$ROOT/experiments/trb/data-anchor-rp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rp-seed42/run_config.json" \
  --model "tp:$ROOT/experiments/trb/data-anchor-tp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-tp-seed42/run_config.json" \
  --model "rtp:$ROOT/experiments/trb/anchor-data-rtp-seed42/best.pt:$ROOT/experiments/trb/anchor-data-rtp-seed42/run_config.json" \
  --batch-size 128 \
  --max-positive-candidates 20 \
  --negative-ratio 4 \
  --bootstrap 2000 \
  --silhouette-sample 10000 \
  --seed 42
