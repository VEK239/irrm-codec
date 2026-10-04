#!/bin/bash -l
#SBATCH --job-name=irrm-pca256-recon
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/recon-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/recon-%A-%a.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca256-v1"
REPRESENTATIONS=(esm2_8m_residue tcr_bert_residue sceptr_residue)
SEEDS=(42 43 44)
REPRESENTATION="${REPRESENTATIONS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/reconstruction"
cd "$REPO"
"$PYTHON" benchmark/train_external_model_decoder.py \
  --dataset-dir "$BASE/cohort" --bottleneck-dir "$ROOT/bottleneck" \
  --output-dir "$ROOT/reconstruction" --representation "$REPRESENTATION" \
  --bottleneck-dim 256 --train-subset all --seed "$SEED" --epochs 60 \
  --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 8 \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted representation=$REPRESENTATION seed=$SEED"
