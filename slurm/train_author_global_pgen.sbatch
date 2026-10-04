#!/bin/bash
#SBATCH --job-name=irrm-author-global-pgen
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/pgen-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/pgen-%A-%a.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
OLD="$REPO/experiments/trb/external-model-benchmark"
BASE="$REPO/experiments/trb/external-global-author-v1"
ARMS=(esm2_35m_mlp sceptr_cdr3_mlp)
SEEDS=(42 43 44)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$OLD/runtime-v2/site-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/pgen"
cd "$REPO"

"$PYTHON" benchmark/train_external_model_pgen.py --dataset-dir "$OLD/cohort" --representations-dir "$OLD/cohort/representations" --output-dir "$BASE/pgen" --arm "$ARM" --target log10_pgen_1mm --train-subset all --seed "$SEED" --epochs 60 --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 12 --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted arm=$ARM seed=$SEED"
