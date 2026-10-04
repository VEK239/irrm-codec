#!/bin/bash
#SBATCH --job-name=irrm-author-global-recon
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%2
#SBATCH --output=recon-%A-%a.out
#SBATCH --error=recon-%A-%a.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
OLD="$REPO/experiments/trb/external-model-benchmark"
BASE="$REPO/experiments/trb/external-global-author-v1"
REPS=(esm2_35m sceptr_cdr3)
SEEDS=(42 43 44)
REP="${REPS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$OLD/runtime-v2/site-packages:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/reconstruction"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.reconstruction.train_external_model_decoder --dataset-dir "$OLD/cohort" --bottleneck-dir "$BASE/bottleneck" --output-dir "$BASE/reconstruction" --representation "$REP" --bottleneck-dim 64 --train-subset all --seed "$SEED" --epochs 60 --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 8 --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted representation=$REP seed=$SEED"
