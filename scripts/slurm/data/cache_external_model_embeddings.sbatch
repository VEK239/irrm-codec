#!/bin/bash
#SBATCH --job-name=irrm-ext-cache
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%2
#SBATCH --output=cache-%A-%a.out
#SBATCH --error=cache-%A-%a.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
RUNTIME="$REPO/experiments/trb/external-model-benchmark/runtime-v2"
COHORT="$REPO/experiments/trb/external-model-benchmark/cohort"
ENCODERS=(esm2_8m tcr_bert sceptr)
ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"

export PYTHONPATH="$RUNTIME/site-packages:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME/huggingface"
export TORCH_HOME="$RUNTIME/torch"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$REPO/experiments/trb/external-model-benchmark/logs"
cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "encoder=$ENCODER python=$PYTHON"
"$PYTHON" -m rtp_codec.benchmarks.representations.cache_external_model_embeddings \
  --dataset-dir "$COHORT" \
  --output-dir "$COHORT/representations" \
  --encoders "$ENCODER" \
  --rtp-embeddings "$COHORT/representations/rtp.npy" \
  --device cuda \
  --batch-size 256 \
  --threads "${SLURM_CPUS_PER_TASK:-8}"

sha256sum "$COHORT/representations/$ENCODER.npy"
echo "status=accepted encoder=$ENCODER"
