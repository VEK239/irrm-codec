#!/bin/bash -l
#SBATCH --job-name=irrm-res-recon
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-8%2
#SBATCH --output=recon-%A-%a.out
#SBATCH --error=recon-%A-%a.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-v1"
REPRESENTATIONS=(esm2_8m tcr_bert sceptr)
SEEDS=(42 43 44)
REPRESENTATION="${REPRESENTATIONS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/reconstruction"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.reconstruction.train_external_residue_decoder \
  --dataset-dir "$BASE/cohort" \
  --residue-dir "$ROOT/representations" \
  --output-dir "$ROOT/reconstruction" \
  --representation "$REPRESENTATION" \
  --seed "$SEED" \
  --epochs 60 \
  --batch-size 64 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --patience 8 \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted representation=$REPRESENTATION seed=$SEED"
