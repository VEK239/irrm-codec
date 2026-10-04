#!/bin/bash -l
#SBATCH --job-name=finalize-epitope-prediction
#SBATCH --partition=short
#SBATCH --constraint=hpc
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=finalize-%j.out
#SBATCH --error=finalize-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
OUT="${EPITOPE_PREDICTION_OUT:-$REPO/experiments/trb/epitope-prediction-v1/evaluation-1429982}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.downstream.finalize_vdjdb_epitope_prediction --output-dir "$OUT"
sha256sum "$OUT"/* > "$OUT/HASHES.sha256"
