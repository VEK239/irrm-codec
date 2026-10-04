#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca128-pre
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-v1/logs/preflight-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-v1/logs/preflight-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs"
cd "$REPO"
"$PYTHON" -m py_compile \
  benchmark/audit_external_residue_alignment.py \
  benchmark/fit_external_residue_pca256.py \
  benchmark/finalize_external_residue_pca256.py \
  benchmark/train_external_model_decoder.py \
  benchmark/train_external_model_pgen.py \
  benchmark/combine_external_pca128_results.py
"$PYTHON" benchmark/audit_external_residue_alignment.py \
  --dataset-dir "$BASE/cohort" --residue-dir "$BASE/residue-v1/representations" \
  --output "$ROOT/RESIDUE_AUDIT.json" --seed 42
bash -n slurm/fit_external_residue_pca128_all.sbatch \
  slurm/finalize_external_residue_pca128_all.sbatch \
  slurm/train_external_residue_pca128_remaining.sbatch \
  slurm/collect_external_residue_pca128_all.sbatch
echo status=accepted
