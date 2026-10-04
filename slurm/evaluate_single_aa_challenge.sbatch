#!/bin/bash -l
#SBATCH --job-name=irrm-aa1-eval
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
TARGET_ROOT="${1:?Validated target job directory required}"
ROOT="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$ROOT/experiments/trb/single-aa-challenge"
OUT="$BASE/evaluation-${SLURM_JOB_ID}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
TOKENIZER="$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
test -f "$TARGET_ROOT/targets/TARGETS.json"
test ! -e "$OUT"
cd "$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PYTHON" -m benchmark.evaluate_single_aa_challenge \
  --prepared "$TARGET_ROOT/prepared" --targets "$TARGET_ROOT/targets" \
  --tokenizer "$TOKENIZER" --output-dir "$OUT" --bootstrap 0 --batch-size 128 --seed 42 \
  --model "r:$ROOT/experiments/trb/data-anchor-r-seed42/best.pt:$ROOT/experiments/trb/data-anchor-r-seed42/run_config.json" \
  --model "t:$ROOT/experiments/trb/data-anchor-t-seed42/best.pt:$ROOT/experiments/trb/data-anchor-t-seed42/run_config.json" \
  --model "p:$ROOT/experiments/trb/data-anchor-p-seed42/best.pt:$ROOT/experiments/trb/data-anchor-p-seed42/run_config.json" \
  --model "rt:$ROOT/experiments/trb/data-anchor-rt-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rt-seed42/run_config.json" \
  --model "rp:$ROOT/experiments/trb/data-anchor-rp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rp-seed42/run_config.json" \
  --model "tp:$ROOT/experiments/trb/data-anchor-tp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-tp-seed42/run_config.json" \
  --model "rtp:$ROOT/experiments/trb/anchor-data-rtp-seed42/best.pt:$ROOT/experiments/trb/anchor-data-rtp-seed42/run_config.json"
