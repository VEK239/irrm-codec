#!/bin/bash -l
#SBATCH --job-name=irrm-pca256-pgen
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/pgen-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/pgen-%A-%a.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca256-v1"
ARMS=(esm2_8m_residue_pca256_mlp tcr_bert_residue_pca256_mlp sceptr_residue_pca256_mlp)
SEEDS=(42 43 44)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/pgen"
cd "$REPO"
"$PYTHON" benchmark/train_external_model_pgen.py \
  --dataset-dir "$BASE/cohort" --representations-dir "$ROOT/bottleneck" \
  --output-dir "$ROOT/pgen" --arm "$ARM" --target log10_pgen_1mm \
  --train-subset all --seed "$SEED" --epochs 60 --batch-size 256 \
  --lr 0.001 --weight-decay 0.0001 --patience 12 \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted arm=$ARM seed=$SEED"
