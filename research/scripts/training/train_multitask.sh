#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/benchmark/trb}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/multitask/char}"
TRAIN_SUBSET="${TRAIN_SUBSET:-all}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-char}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-42}"

ARGS=(
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --train-subset "$TRAIN_SUBSET"
  --tokenizer-type "$TOKENIZER_TYPE"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --seed "$SEED"
)

if [[ "$TOKENIZER_TYPE" != "char" ]]; then
  if [[ -z "$TOKENIZER_PATH" ]]; then
    echo "TOKENIZER_PATH is required for TOKENIZER_TYPE=$TOKENIZER_TYPE" >&2
    exit 2
  fi
  ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
fi

python -m rtp_codec.training.multitask "${ARGS[@]}" "$@"
