#!/bin/bash -l
#SBATCH --job-name=tcremp-geom-t
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
VARIANT="${1:?Variant centered or relational required}"
case "$VARIANT" in
  centered) C=0.30; P=0.00 ;;
  relational) C=0.15; P=0.35 ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
ROOT="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="$ROOT/experiments/trb/tcremp-geometry-t-${VARIANT}-seed42"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
test ! -e "$OUT/best.pt"
cd "$ROOT"
PYTHONPATH="$ROOT" "$PYTHON" -m irrm_codec.train_multitask \
  --data-dir "$ROOT/data/benchmark/trb" --standardizer-path "$ROOT/data/benchmark/trb/target_standardizer.npz" \
  --output-dir "$OUT" --train-subset all --pgen-target log10_pgen_1mm \
  --tokenizer-type data_anchor --tokenizer-path "$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json" \
  --max-sequence-len 40 --d-model 320 --latent-dim 128 --nhead 8 \
  --encoder-layers 4 --decoder-layers 4 --ff-dim 1280 --dropout 0.1 \
  --tcremp-head-dim 1024 --pgen-head-dim 256 --decoder-memory-tokens 4 \
  --tcremp-loss-weight 1 --pgen-loss-weight 0 --reconstruction-loss-weight 0 \
  --tcremp-mse-fraction 1 --tcremp-centered-cosine-weight "$C" \
  --tcremp-pairwise-log-distance-weight "$P" \
  --batch-size 64 --epochs 40 --lr 0.0003 --weight-decay 0.0001 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 --early-stopping-patience 0 \
  --scheduler-factor 0.5 --scheduler-patience 2 --scheduler-min-lr 0.000001 \
  --seed 42 --num-workers 8 --device cuda --amp --no-progress --no-save-test-predictions