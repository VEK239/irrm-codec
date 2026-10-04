#!/bin/bash -l
#SBATCH --job-name=irrm-sceptr-pca128-sum
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/summary-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/summary-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/results/reconstruction" "$ROOT/results/pgen"
cd "$REPO"
"$PYTHON" benchmark/collect_external_reconstruction.py \
  --runs-dir "$ROOT/reconstruction" --output-dir "$ROOT/results/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" --bottleneck-dim 128
"$PYTHON" benchmark/collect_external_pgen.py \
  --runs-dir "$ROOT/pgen" --output-dir "$ROOT/results/pgen"
sha256sum "$ROOT/results/reconstruction"/* "$ROOT/results/pgen"/* \
  > "$ROOT/results/HASHES.sha256"
echo status=accepted
