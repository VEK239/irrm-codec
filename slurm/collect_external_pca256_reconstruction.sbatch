#!/bin/bash -l
#SBATCH --job-name=irrm-pca256-recon-sum
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/recon-summary-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/recon-summary-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca256-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/results/reconstruction"
cd "$REPO"
"$PYTHON" benchmark/collect_external_reconstruction.py \
  --runs-dir "$ROOT/reconstruction" --output-dir "$ROOT/results/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" --bottleneck-dim 256
sha256sum "$ROOT/results/reconstruction"/* > "$ROOT/results/reconstruction/HASHES.sha256"
echo status=accepted
