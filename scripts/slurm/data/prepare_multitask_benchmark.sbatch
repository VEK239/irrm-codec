#!/usr/bin/env bash
#SBATCH --job-name=irrm-prepare
#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"
: "${AIRR_PATH:?Set AIRR_PATH to the source TRB AIRR table}"
: "${EMBEDDINGS_PATH:?Set EMBEDDINGS_PATH to the aligned TCRemP Parquet file}"

OUTPUT_DIR="${OUTPUT_DIR:-data/benchmark/trb}"
PGEN_THREADS="${PGEN_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
PGEN_CHUNK_SIZE="${PGEN_CHUNK_SIZE:-1000}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-1024}"

python -m rtp_codec.benchmarks.datasets.prepare_splits \
  --airr-path "$AIRR_PATH" \
  --embeddings-path "$EMBEDDINGS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --pgen-threads "$PGEN_THREADS" \
  --pgen-chunk-size "$PGEN_CHUNK_SIZE" \
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
