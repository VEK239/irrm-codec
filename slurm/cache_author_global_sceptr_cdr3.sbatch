#!/bin/bash
#SBATCH --job-name=irrm-sceptr-cdr3-redcea
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/sceptr-cdr3-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/sceptr-cdr3-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-global-author-v1"
RUNTIME="$REPO/experiments/trb/external-model-benchmark/runtime-v2"
LOCKED="$REPO/experiments/trb/external-model-benchmark/cohort"
REDCEA="$REPO/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv"
OUT="$BASE/vdjdb/representations"

export PYTHONPATH="$RUNTIME/site-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME/huggingface"
export TORCH_HOME="$RUNTIME/torch"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$BASE/logs" "$OUT"
cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" benchmark/cache_external_model_embeddings.py --dataset-dir "$LOCKED" --dataset-file "$REDCEA" --output-dir "$OUT" --encoders sceptr_cdr3 --device cuda --batch-size 256 --threads "${SLURM_CPUS_PER_TASK:-8}"
sha256sum "$OUT/sceptr_cdr3.npy"
echo "status=accepted"
