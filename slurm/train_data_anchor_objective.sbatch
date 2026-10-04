#!/bin/bash -l
#SBATCH --job-name=irrm-data-obj
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/slurm-%x-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/slurm-%x-%j.err

set -euo pipefail
CONDITION="${1:?Condition r, p, or rp is required}"
case "$CONDITION" in
  r)  RECON=1; TCREMP=0; PGEN=0 ;;
  p)  RECON=0; TCREMP=0; PGEN=1 ;;
  rp) RECON=1; TCREMP=0; PGEN=1 ;;
  *) echo "Refusing unregistered DATA-ANCHOR objective condition" >&2; exit 2 ;;
esac

ROOT_DIR="/home/evlasova/irrm-codec"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER="$ROOT_DIR/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
MATRIX_ROOT="$ROOT_DIR/experiments/trb/data-anchor-objective-matrix"
PREFLIGHT="$MATRIX_ROOT/PREFLIGHT.json"
CONDITION_CONFIG="$MATRIX_ROOT/configs/$CONDITION.json"
OUTPUT_DIR="$ROOT_DIR/experiments/trb/data-anchor-$CONDITION-seed42"
PYTHON_BIN="/home/evlasova/.conda/envs/irrm-codec/bin/python"

test -x "$PYTHON_BIN"
test -f "$DATA_DIR/READY.json"
test -f "$TOKENIZER"
test -f "$PREFLIGHT"
test -f "$CONDITION_CONFIG"
"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="ready" and all(p["checks"].values()); c=json.load(open(sys.argv[2])); assert c["condition"]==sys.argv[3]' "$PREFLIGHT" "$CONDITION_CONFIG" "$CONDITION"
mkdir -p "$OUTPUT_DIR"
if test -e "$OUTPUT_DIR/best.pt" || test -e "$OUTPUT_DIR/test_metrics.json"; then
  echo "Refusing to overwrite completed experiment: $OUTPUT_DIR" >&2; exit 3
fi
cd "$ROOT_DIR"
printf '{"job_id":"%s","condition":"%s","node":"%s","cuda_visible_devices":"%s","preflight_sha256":"%s","condition_config_sha256":"%s"}\n' \
  "$SLURM_JOB_ID" "$CONDITION" "${SLURMD_NODENAME:-unknown}" "${CUDA_VISIBLE_DEVICES:-unknown}" \
  "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" "$(sha256sum "$CONDITION_CONFIG" | cut -d' ' -f1)" \
  > "$OUTPUT_DIR/slurm_runtime.json"
"$PYTHON_BIN" -m irrm_codec.train_multitask \
  --data-dir "$DATA_DIR" --standardizer-path "$DATA_DIR/target_standardizer.npz" \
  --output-dir "$OUTPUT_DIR" --train-subset all --pgen-target log10_pgen_1mm \
  --tokenizer-type data_anchor --tokenizer-path "$TOKENIZER" \
  --max-sequence-len 40 --d-model 320 --latent-dim 128 --nhead 8 \
  --encoder-layers 4 --decoder-layers 4 --ff-dim 1280 --dropout 0.1 \
  --tcremp-head-dim 1024 --pgen-head-dim 256 --decoder-memory-tokens 4 \
  --tcremp-loss-weight "$TCREMP" --pgen-loss-weight "$PGEN" --reconstruction-loss-weight "$RECON" \
  --tcremp-mse-fraction 0.7 --pgen-huber-delta 0.5 --label-smoothing 0 \
  --batch-size 64 --epochs 40 --lr 0.0003 --weight-decay 0.0001 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 --early-stopping-patience 0 \
  --scheduler-factor 0.5 --scheduler-patience 2 --scheduler-min-lr 0.000001 \
  --seed 42 --num-workers 8 --device cuda --amp --no-progress --no-save-test-predictions
