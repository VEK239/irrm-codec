#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca128-heads
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-v1/logs/head-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-v1/logs/head-%A-%a.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-v1"
REPRESENTATIONS=(esm2_8m_residue tcr_bert_residue)
ARMS=(esm2_8m_residue_pca128_mlp tcr_bert_residue_pca128_mlp)
SEEDS=(42 43 44)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/reconstruction" "$ROOT/pgen"
cd "$REPO"

if (( SLURM_ARRAY_TASK_ID < 6 )); then
  REP_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
  SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
  ARM="${ARMS[$REP_INDEX]}"
  SEED="${SEEDS[$SEED_INDEX]}"
  "$PYTHON" benchmark/train_external_model_pgen.py \
    --dataset-dir "$BASE/cohort" --representations-dir "$ROOT/bottleneck" \
    --output-dir "$ROOT/pgen" --arm "$ARM" --target log10_pgen_1mm \
    --train-subset all --seed "$SEED" --epochs 60 --batch-size 256 \
    --lr 0.001 --weight-decay 0.0001 --patience 12 \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
  echo "status=accepted task=pgen arm=$ARM seed=$SEED"
else
  OFFSET=$((SLURM_ARRAY_TASK_ID - 6))
  REP_INDEX=$((OFFSET / 3))
  SEED_INDEX=$((OFFSET % 3))
  REPRESENTATION="${REPRESENTATIONS[$REP_INDEX]}"
  SEED="${SEEDS[$SEED_INDEX]}"
  "$PYTHON" benchmark/train_external_model_decoder.py \
    --dataset-dir "$BASE/cohort" --bottleneck-dir "$ROOT/bottleneck" \
    --output-dir "$ROOT/reconstruction" --representation "$REPRESENTATION" \
    --bottleneck-dim 128 --train-subset all --seed "$SEED" --epochs 60 \
    --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 8 \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
  echo "status=accepted task=reconstruction representation=$REPRESENTATION seed=$SEED"
fi
