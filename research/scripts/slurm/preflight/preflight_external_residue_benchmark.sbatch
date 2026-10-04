#!/bin/bash -l
#SBATCH --job-name=irrm-res-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=preflight-%j.out
#SBATCH --error=preflight-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-v1"
RUNTIME="$BASE/runtime-v2"
SMOKE="$ROOT/preflight-$SLURM_JOB_ID/smoke"
export PYTHONPATH="$RUNTIME/site-packages:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME/huggingface"
export TORCH_HOME="$RUNTIME/torch"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$SMOKE"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.representations.cache_external_residue_embeddings \
  --dataset-dir "$BASE/cohort" \
  --output-dir "$SMOKE" \
  --encoders esm2_8m tcr_bert sceptr \
  --device cpu \
  --batch-size 8 \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --limit 8
"$PYTHON" -m rtp_codec.benchmarks.representations.preflight_external_residue_benchmark \
  --cohort-dir "$BASE/cohort" \
  --smoke-dir "$SMOKE" \
  --output "$ROOT/preflight-$SLURM_JOB_ID/PREFLIGHT.json"
sha256sum "$ROOT/preflight-$SLURM_JOB_ID/PREFLIGHT.json"
echo "status=accepted job=$SLURM_JOB_ID"
