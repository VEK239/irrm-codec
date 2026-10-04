#!/bin/bash
#SBATCH --job-name=irrm-author-global-controls
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%2
#SBATCH --output=pgen-controls-%A-%a.out
#SBATCH --error=pgen-controls-%A-%a.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
OLD="$REPO/experiments/trb/external-model-benchmark"
BASE="$REPO/experiments/trb/external-global-author-v1"
ARMS=(onehot_mlp tcremp_mlp)
SEEDS=(42 43 44)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"

export PYTHONPATH="$OLD/runtime-v2/site-packages:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/pgen"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.pgen.train_external_model_pgen --dataset-dir "$OLD/cohort" --representations-dir "$OLD/cohort/representations" --output-dir "$BASE/pgen" --arm "$ARM" --target log10_pgen_1mm --train-subset all --seed "$SEED" --epochs 60 --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 12 --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted arm=$ARM seed=$SEED"
