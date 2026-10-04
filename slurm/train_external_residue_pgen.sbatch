#!/bin/bash -l
#SBATCH --job-name=irrm-res-pgen
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-v1/logs/pgen-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-v1/logs/pgen-%A-%a.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-v1"
REPRESENTATIONS=(esm2_8m tcr_bert sceptr)
SEEDS=(42 43 44)
REPRESENTATION="${REPRESENTATIONS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/pgen"
cd "$REPO"
"$PYTHON" benchmark/train_external_residue_pgen.py \
  --dataset-dir "$BASE/cohort" \
  --residue-dir "$ROOT/representations" \
  --output-dir "$ROOT/pgen" \
  --representation "$REPRESENTATION" \
  --target log10_pgen_1mm \
  --seed "$SEED" \
  --epochs 60 \
  --batch-size 128 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --patience 12 \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "status=accepted representation=$REPRESENTATION seed=$SEED"
