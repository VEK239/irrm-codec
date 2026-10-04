#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca-gate
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=finalize-%j.out
#SBATCH --error=finalize-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca256-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.reporting.finalize_external_residue_pca256 --output-dir "$ROOT/bottleneck"
sha256sum "$ROOT/bottleneck"/*_256.npy "$ROOT/bottleneck/representations.json" \
  "$ROOT/bottleneck/bottleneck_summary.json" > "$ROOT/bottleneck/HASHES.sha256"
echo status=accepted
