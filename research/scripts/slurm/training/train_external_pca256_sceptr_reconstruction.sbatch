#!/bin/bash -l
#SBATCH --job-name=irrm-pca256-sceptr-recon
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%2
#SBATCH --output=recon-sceptr-%A-%a.out
#SBATCH --error=recon-sceptr-%A-%a.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca256-v1"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/reconstruction"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.reconstruction.train_external_model_decoder \
  --dataset-dir "$BASE/cohort" --bottleneck-dir "$ROOT/bottleneck" \
  --output-dir "$ROOT/reconstruction" --representation sceptr_residue \
  --bottleneck-dim 256 --train-subset all --seed "$SEED" --epochs 60 \
  --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 8 \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted representation=sceptr_residue seed=$SEED"
