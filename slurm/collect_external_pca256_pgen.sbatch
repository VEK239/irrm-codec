#!/bin/bash -l
#SBATCH --job-name=irrm-pca256-pgen-sum
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/pgen-summary-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca256-v1/logs/pgen-summary-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca256-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/results/pgen"
cd "$REPO"
"$PYTHON" benchmark/collect_external_pgen.py --runs-dir "$ROOT/pgen" --output-dir "$ROOT/results/pgen"
sha256sum "$ROOT/results/pgen"/* > "$ROOT/results/pgen/HASHES.sha256"
echo status=accepted
