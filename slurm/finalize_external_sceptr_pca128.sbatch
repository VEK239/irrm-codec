#!/bin/bash -l
#SBATCH --job-name=irrm-sceptr-pca128-gate
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/gate-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/gate-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
"$PYTHON" benchmark/finalize_external_residue_pca_single.py \
  --output-dir "$ROOT/bottleneck" --representation sceptr \
  --rows 99430 --components 128
sha256sum "$ROOT/bottleneck"/* > "$ROOT/bottleneck/HASHES.sha256"
