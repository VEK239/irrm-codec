#!/bin/bash
#SBATCH --job-name=irrm-ext-recon
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-17%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/recon-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/recon-%A-%a.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark"
RUNTIME="$ROOT/runtime-v2"
REPRESENTATIONS=(onehot tcremp esm2_8m tcr_bert sceptr rtp)
SEEDS=(42 43 44)
REPRESENTATION="${REPRESENTATIONS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$RUNTIME/site-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/reconstruction"
cd "$REPO"

"$PYTHON" benchmark/train_external_model_decoder.py \
  --dataset-dir "$ROOT/cohort" \
  --bottleneck-dir "$ROOT/bottleneck" \
  --output-dir "$ROOT/reconstruction" \
  --representation "$REPRESENTATION" \
  --bottleneck-dim 64 \
  --train-subset all \
  --seed "$SEED" \
  --epochs 60 \
  --batch-size 256 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --patience 8 \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted representation=$REPRESENTATION seed=$SEED"
