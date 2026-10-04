#!/bin/bash -l
#SBATCH --job-name=irrm-res-cache
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
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-v1"
RUNTIME="$BASE/runtime-v2"
ENCODERS=(esm2_8m tcr_bert sceptr)
ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONPATH="$RUNTIME/site-packages:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME/huggingface"
export TORCH_HOME="$RUNTIME/torch"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/representations"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m rtp_codec.benchmarks.representations.cache_external_residue_embeddings \
  --dataset-dir "$BASE/cohort" \
  --output-dir "$ROOT/representations" \
  --encoders "$ENCODER" \
  --device cuda \
  --batch-size 128 \
  --threads "${SLURM_CPUS_PER_TASK:-8}"
sha256sum "$ROOT/representations/${ENCODER}_residue.npy" "$ROOT/representations/${ENCODER}_residue_mask.npy"
echo "status=accepted encoder=$ENCODER"
