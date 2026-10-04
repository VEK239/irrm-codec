#!/bin/bash -l
#SBATCH --job-name=vdjdb-irrm-gpu
#SBATCH --partition=short
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -euo pipefail
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="$ROOT/experiments/trb/vdjdb-runtime-benchmark/irrm-gpu-${SLURM_JOB_ID}"
mkdir -p "$OUT"
cd "$ROOT"
${RTP_CODEC_PYTHON:-python} -m rtp_codec.benchmarks.runtime.benchmark_rtp_runtime \
  --cohort "$ROOT/experiments/trb/vdjdb-uniform100-v1/tcremp-1429926/cohort.airr.tsv" \
  --sequence-column junction_aa \
  --checkpoint "$ROOT/experiments/trb/anchor-data-rtp-seed42/best.pt" \
  --tokenizer "$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json" \
  --output "$OUT/RESULTS.json" --device cuda --batch-size 512
