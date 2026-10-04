#!/bin/bash
#SBATCH --job-name=irrm-ext-pgen
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/pgen-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/pgen-%A-%a.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark"
ARMS=(sceptr_mlp tcr_bert_mlp esm2_8m_mlp rtp_mlp)
SEEDS=(42 43 44)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/pgen"
cd "$REPO"

"$PYTHON" benchmark/train_external_model_pgen.py \
  --dataset-dir "$ROOT/cohort" \
  --representations-dir "$ROOT/cohort/representations" \
  --output-dir "$ROOT/pgen" \
  --arm "$ARM" \
  --target log10_pgen_1mm \
  --train-subset all \
  --seed "$SEED" \
  --epochs 60 \
  --batch-size 256 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --patience 12 \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted arm=$ARM seed=$SEED"
