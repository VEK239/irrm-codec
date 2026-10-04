#!/bin/bash
#SBATCH --job-name=irrm-ext-cache
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/cache-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/cache-%A-%a.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
RUNTIME="$REPO/experiments/trb/external-model-benchmark/runtime-v2"
COHORT="$REPO/experiments/trb/external-model-benchmark/cohort"
ENCODERS=(esm2_8m tcr_bert sceptr)
ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"

export PYTHONPATH="$RUNTIME/site-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME/huggingface"
export TORCH_HOME="$RUNTIME/torch"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$REPO/experiments/trb/external-model-benchmark/logs"
cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "encoder=$ENCODER python=$PYTHON"
"$PYTHON" benchmark/cache_external_model_embeddings.py \
  --dataset-dir "$COHORT" \
  --output-dir "$COHORT/representations" \
  --encoders "$ENCODER" \
  --rtp-embeddings "$COHORT/representations/rtp.npy" \
  --device cuda \
  --batch-size 256 \
  --threads "${SLURM_CPUS_PER_TASK:-8}"

sha256sum "$COHORT/representations/$ENCODER.npy"
echo "status=accepted encoder=$ENCODER"
