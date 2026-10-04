#!/bin/bash -l
#SBATCH --job-name=irrm-wp-sweep
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
VOCAB_SIZE="${1:?WordPiece vocabulary size is required}"
case "$VOCAB_SIZE" in
  44|64|128|256|512|1024|2048|4096) ;;
  *) echo "Refusing unregistered WordPiece vocabulary size" >&2; exit 2 ;;
esac

ROOT_DIR="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER_ROOT="$ROOT_DIR/experiments/trb/tokenizers/wordpiece-sweep-seed42"
TOKENIZER_DIR="$TOKENIZER_ROOT/v${VOCAB_SIZE}-seed42"
TOKENIZER_PATH="$TOKENIZER_DIR/tokenizer.json"
EXPERIMENT_ID="wp-sweep-v${VOCAB_SIZE}-rtp-seed42"
OUTPUT_DIR="$ROOT_DIR/experiments/trb/$EXPERIMENT_ID"
PYTHON_BIN="${IRRM_PYTHON:-$(command -v python)}"

test -f "$DATA_DIR/READY.json"
test -f "$TOKENIZER_ROOT/preflight.json"
test -f "$TOKENIZER_PATH"
mkdir -p "$OUTPUT_DIR"
if test -e "$OUTPUT_DIR/best.pt" || test -e "$OUTPUT_DIR/test_metrics.json"; then
  echo "Refusing to overwrite completed experiment: $OUTPUT_DIR" >&2; exit 3
fi

cd "$ROOT_DIR"
printf '{"job_id":"%s","experiment_id":"%s","vocab_size":%s,"node":"%s","cuda_visible_devices":"%s"}\n' \
  "$SLURM_JOB_ID" "$EXPERIMENT_ID" "$VOCAB_SIZE" "${SLURMD_NODENAME:-unknown}" \
  "${CUDA_VISIBLE_DEVICES:-unknown}" > "$OUTPUT_DIR/slurm_runtime.json"
"$PYTHON_BIN" -m irrm_codec.train_multitask \
  --data-dir "$DATA_DIR" --standardizer-path "$DATA_DIR/target_standardizer.npz" \
  --output-dir "$OUTPUT_DIR" --train-subset all --pgen-target log10_pgen_1mm \
  --tokenizer-type wordpiece --tokenizer-path "$TOKENIZER_PATH" \
  --max-sequence-len 40 --d-model 320 --latent-dim 128 --nhead 8 \
  --encoder-layers 4 --decoder-layers 4 --ff-dim 1280 --dropout 0.1 \
  --tcremp-head-dim 1024 --pgen-head-dim 256 --decoder-memory-tokens 4 \
  --tcremp-loss-weight 1 --pgen-loss-weight 1 --reconstruction-loss-weight 1 \
  --tcremp-mse-fraction 0.7 --pgen-huber-delta 0.5 --label-smoothing 0 \
  --batch-size 64 --epochs 40 --lr 0.0003 --weight-decay 0.0001 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 --early-stopping-patience 0 \
  --scheduler-factor 0.5 --scheduler-patience 2 --scheduler-min-lr 0.000001 \
  --seed 42 --num-workers 8 --device cuda --amp --no-progress --no-save-test-predictions
