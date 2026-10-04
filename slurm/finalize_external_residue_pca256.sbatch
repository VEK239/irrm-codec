#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca-gate
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/finalize-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/finalize-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca256-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
"$PYTHON" benchmark/finalize_external_residue_pca256.py --output-dir "$ROOT/bottleneck"
sha256sum "$ROOT/bottleneck"/*_256.npy "$ROOT/bottleneck/representations.json" \
  "$ROOT/bottleneck/bottleneck_summary.json" > "$ROOT/bottleneck/HASHES.sha256"
echo status=accepted
