#!/bin/bash -l
#SBATCH --job-name=irrm-pca256-recon-sum
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=recon-summary-%j.out
#SBATCH --error=recon-summary-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca256-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/results/reconstruction"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.reconstruction.collect_external_reconstruction \
  --runs-dir "$ROOT/reconstruction" --output-dir "$ROOT/results/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" --bottleneck-dim 256
sha256sum "$ROOT/results/reconstruction"/* > "$ROOT/results/reconstruction/HASHES.sha256"
echo status=accepted
