#!/bin/bash -l
#SBATCH --job-name=irrm-sceptr-pca128-pre
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/preflight-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/preflight-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs"
cd "$REPO"
test -f "$BASE/cohort/dataset.parquet"
test -f "$BASE/cohort/manifests/train.tsv"
test -f "$BASE/residue-v1/representations/sceptr_residue.npy"
"$PYTHON" -m py_compile \
  benchmark/fit_external_residue_pca256.py \
  benchmark/finalize_external_residue_pca_single.py \
  benchmark/preflight_external_sceptr_pca128.py \
  benchmark/train_external_model_decoder.py \
  benchmark/train_external_model_pgen.py
"$PYTHON" benchmark/preflight_external_sceptr_pca128.py \
  --dataset-dir "$BASE/cohort" \
  --residue-dir "$BASE/residue-v1/representations" \
  --output "$ROOT/PREFLIGHT.json"
bash -n \
  slurm/fit_external_sceptr_pca128.sbatch \
  slurm/finalize_external_sceptr_pca128.sbatch \
  slurm/train_external_sceptr_pca128_heads.sbatch \
  slurm/collect_external_sceptr_pca128.sbatch
echo status=accepted
