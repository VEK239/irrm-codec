#!/bin/bash -l
#SBATCH --job-name=tcremp-geom-smoke
#SBATCH --partition=short
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/slurm-%x-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/slurm-%x-%j.err
set -euo pipefail
ROOT=/home/evlasova/irrm-codec
OUT="$ROOT/experiments/trb/tcremp-geometry-smoke-${SLURM_JOB_ID}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
cd "$ROOT"
PYTHONPATH="$ROOT" "$PYTHON" tests/test_tcremp_geometry_loss.py
PYTHONPATH="$ROOT" "$PYTHON" -m irrm_codec.train_multitask \
  --data-dir "$ROOT/data/benchmark/trb" --standardizer-path "$ROOT/data/benchmark/trb/target_standardizer.npz" \
  --output-dir "$OUT" --train-subset all --pgen-target log10_pgen_1mm \
  --tokenizer-type data_anchor --tokenizer-path "$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json" \
  --max-sequence-len 40 --d-model 320 --latent-dim 128 --nhead 8 \
  --encoder-layers 4 --decoder-layers 4 --ff-dim 1280 --dropout 0.1 \
  --tcremp-head-dim 1024 --pgen-head-dim 256 --decoder-memory-tokens 4 \
  --tcremp-loss-weight 1 --pgen-loss-weight 0 --reconstruction-loss-weight 0 \
  --tcremp-mse-fraction 1 --tcremp-centered-cosine-weight 0.15 \
  --tcremp-pairwise-log-distance-weight 0.35 \
  --batch-size 4 --epochs 1 --lr 0.0003 --weight-decay 0.0001 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 --early-stopping-patience 0 \
  --scheduler-factor 0.5 --scheduler-patience 2 --scheduler-min-lr 0.000001 \
  --seed 42 --num-workers 0 --device cuda --amp --no-progress \
  --max-train-batches 2 --max-eval-batches 2 --skip-generation-metrics --no-save-test-predictions