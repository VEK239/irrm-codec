#!/bin/bash -l
#SBATCH --job-name=irrm-rtp128-recon
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%2
#SBATCH --output=recon-%A-%a.out
#SBATCH --error=recon-%A-%a.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/native-rtp128-v1"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$REPO"

test "$("$PYTHON" -c 'import json; print(json.load(open("experiments/trb/external-model-benchmark/native-rtp128-v1/PREFLIGHT.json"))["status"])')" = accepted

"$PYTHON" -m rtp_codec.benchmarks.reconstruction.train_external_model_decoder \
  --dataset-dir "$BASE/cohort" \
  --bottleneck-dir "$ROOT/bottleneck" \
  --output-dir "$ROOT/reconstruction" \
  --representation rtp \
  --bottleneck-dim 128 \
  --train-subset all \
  --seed "$SEED" \
  --epochs 60 \
  --batch-size 256 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --patience 8 \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted representation=rtp native_dim=128 seed=$SEED"
