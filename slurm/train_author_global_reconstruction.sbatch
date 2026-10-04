#!/bin/bash
#SBATCH --job-name=irrm-author-global-recon
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/recon-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/recon-%A-%a.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
OLD="$REPO/experiments/trb/external-model-benchmark"
BASE="$REPO/experiments/trb/external-global-author-v1"
REPS=(esm2_35m sceptr_cdr3)
SEEDS=(42 43 44)
REP="${REPS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$OLD/runtime-v2/site-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/reconstruction"
cd "$REPO"

"$PYTHON" benchmark/train_external_model_decoder.py --dataset-dir "$OLD/cohort" --bottleneck-dir "$BASE/bottleneck" --output-dir "$BASE/reconstruction" --representation "$REP" --bottleneck-dim 64 --train-subset all --seed "$SEED" --epochs 60 --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 8 --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted representation=$REP seed=$SEED"
