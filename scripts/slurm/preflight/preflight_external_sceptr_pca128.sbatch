#!/bin/bash -l
#SBATCH --job-name=irrm-sceptr-pca128-pre
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=preflight-%j.out
#SBATCH --error=preflight-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs"
cd "$REPO"
test -f "$BASE/cohort/dataset.parquet"
test -f "$BASE/cohort/manifests/train.tsv"
test -f "$BASE/residue-v1/representations/sceptr_residue.npy"
"$PYTHON" -m py_compile \
  src/rtp_codec/benchmarks/representations/fit_external_residue_pca256.py \
  src/rtp_codec/benchmarks/reporting/finalize_external_residue_pca_single.py \
  src/rtp_codec/benchmarks/representations/preflight_external_sceptr_pca128.py \
  src/rtp_codec/benchmarks/reconstruction/train_external_model_decoder.py \
  src/rtp_codec/benchmarks/pgen/train_external_model_pgen.py
"$PYTHON" -m rtp_codec.benchmarks.representations.preflight_external_sceptr_pca128 \
  --dataset-dir "$BASE/cohort" \
  --residue-dir "$BASE/residue-v1/representations" \
  --output "$ROOT/PREFLIGHT.json"
bash -n \
  scripts/slurm/data/fit_external_sceptr_pca128.sbatch \
  scripts/slurm/reports/finalize_external_sceptr_pca128.sbatch \
  scripts/slurm/training/train_external_sceptr_pca128_heads.sbatch \
  scripts/slurm/reports/collect_external_sceptr_pca128.sbatch
echo status=accepted
