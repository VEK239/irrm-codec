#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca128-pre
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=preflight-%j.out
#SBATCH --error=preflight-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs"
cd "$REPO"
"$PYTHON" -m py_compile \
  src/rtp_codec/benchmarks/representations/audit_external_residue_alignment.py \
  src/rtp_codec/benchmarks/representations/fit_external_residue_pca256.py \
  src/rtp_codec/benchmarks/reporting/finalize_external_residue_pca256.py \
  src/rtp_codec/benchmarks/reconstruction/train_external_model_decoder.py \
  src/rtp_codec/benchmarks/pgen/train_external_model_pgen.py \
  src/rtp_codec/benchmarks/reporting/combine_external_pca128_results.py
"$PYTHON" -m rtp_codec.benchmarks.representations.audit_external_residue_alignment \
  --dataset-dir "$BASE/cohort" --residue-dir "$BASE/residue-v1/representations" \
  --output "$ROOT/RESIDUE_AUDIT.json" --seed 42
bash -n research/scripts/slurm/data/fit_external_residue_pca128_all.sbatch \
  research/scripts/slurm/reports/finalize_external_residue_pca128_all.sbatch \
  research/scripts/slurm/training/train_external_residue_pca128_remaining.sbatch \
  research/scripts/slurm/reports/collect_external_residue_pca128_all.sbatch
echo status=accepted
