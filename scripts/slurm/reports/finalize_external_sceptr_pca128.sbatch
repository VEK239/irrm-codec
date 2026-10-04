#!/bin/bash -l
#SBATCH --job-name=irrm-sceptr-pca128-gate
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=gate-%j.out
#SBATCH --error=gate-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.reporting.finalize_external_residue_pca_single \
  --output-dir "$ROOT/bottleneck" --representation sceptr \
  --rows 99430 --components 128
sha256sum "$ROOT/bottleneck"/* > "$ROOT/bottleneck/HASHES.sha256"
