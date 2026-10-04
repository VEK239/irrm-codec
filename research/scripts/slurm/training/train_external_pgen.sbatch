#!/bin/bash
#SBATCH --job-name=irrm-ext-pgen
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%2
#SBATCH --output=pgen-%A-%a.out
#SBATCH --error=pgen-%A-%a.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark"
ARMS=(sceptr_mlp tcr_bert_mlp esm2_8m_mlp rtp_mlp)
SEEDS=(42 43 44)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/pgen"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.pgen.train_external_model_pgen \
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
