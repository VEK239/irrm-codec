#!/bin/bash -l
#SBATCH --job-name=finalize-epitope-prediction
#SBATCH --partition=short
#SBATCH --constraint=hpc
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/epitope-prediction-v1/logs/finalize-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/epitope-prediction-v1/logs/finalize-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
OUT="${EPITOPE_PREDICTION_OUT:-$REPO/experiments/trb/epitope-prediction-v1/evaluation-1429982}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
cd "$REPO"
"$PYTHON" benchmark/finalize_vdjdb_epitope_prediction.py --output-dir "$OUT"
sha256sum "$OUT"/* > "$OUT/HASHES.sha256"
