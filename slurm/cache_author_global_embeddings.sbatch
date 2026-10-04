#!/bin/bash
#SBATCH --job-name=irrm-author-global-cache
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/cache-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/cache-%A-%a.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-global-author-v1"
RUNTIME="$REPO/experiments/trb/external-model-benchmark/runtime-v2"
LOCKED="$REPO/experiments/trb/external-model-benchmark/cohort"
LOCKED_REPS="$LOCKED/representations"
REDCEA="$REPO/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv"
REDCEA_REPS="$BASE/vdjdb/representations"

export PYTHONPATH="$RUNTIME/site-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME/huggingface"
export TORCH_HOME="$RUNTIME/torch"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$BASE/logs" "$REDCEA_REPS"
cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  "$PYTHON" benchmark/cache_external_model_embeddings.py --dataset-dir "$LOCKED" --output-dir "$LOCKED_REPS" --encoders esm2_35m --device cuda --batch-size 256 --threads "${SLURM_CPUS_PER_TASK:-8}"
  "$PYTHON" benchmark/cache_external_model_embeddings.py --dataset-dir "$LOCKED" --dataset-file "$REDCEA" --output-dir "$REDCEA_REPS" --encoders esm2_35m tcr_bert --device cuda --batch-size 256 --threads "${SLURM_CPUS_PER_TASK:-8}"
  sha256sum "$LOCKED_REPS/esm2_35m.npy" "$REDCEA_REPS/esm2_35m.npy" "$REDCEA_REPS/tcr_bert.npy"
else
  "$PYTHON" benchmark/cache_external_model_embeddings.py --dataset-dir "$LOCKED" --output-dir "$LOCKED_REPS" --encoders sceptr_cdr3 --device cuda --batch-size 256 --threads "${SLURM_CPUS_PER_TASK:-8}"
  "$PYTHON" benchmark/cache_external_model_embeddings.py --dataset-dir "$LOCKED" --dataset-file "$REDCEA" --output-dir "$REDCEA_REPS" --encoders sceptr sceptr_cdr3 --device cuda --batch-size 256 --threads "${SLURM_CPUS_PER_TASK:-8}"
  sha256sum "$LOCKED_REPS/sceptr_cdr3.npy" "$REDCEA_REPS/sceptr.npy" "$REDCEA_REPS/sceptr_cdr3.npy"
fi
echo "status=accepted task=$SLURM_ARRAY_TASK_ID"
