#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca128-sum
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-v1/logs/summary-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-v1/logs/summary-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-v1"
SCEPTR_ROOT="$BASE/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/results/reconstruction" "$ROOT/results/pgen"
cd "$REPO"
"$PYTHON" benchmark/collect_external_reconstruction.py \
  --runs-dir "$ROOT/reconstruction" --output-dir "$ROOT/results/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" --bottleneck-dim 128
"$PYTHON" benchmark/collect_external_pgen.py \
  --runs-dir "$ROOT/pgen" --output-dir "$ROOT/results/pgen"
"$PYTHON" benchmark/combine_external_pca128_results.py \
  --main-root "$ROOT" --sceptr-root "$SCEPTR_ROOT"
sha256sum "$ROOT/results/reconstruction"/* "$ROOT/results/pgen"/* \
  "$ROOT/results/all_baselines"/* > "$ROOT/results/HASHES.sha256"
echo status=accepted
