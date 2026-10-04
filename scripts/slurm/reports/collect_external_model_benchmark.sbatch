#!/bin/bash
#SBATCH --job-name=irrm-ext-collect
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=collect-%j.out
#SBATCH --error=collect-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/results/reconstruction" "$ROOT/results/pgen"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.reconstruction.collect_external_reconstruction \
  --runs-dir "$ROOT/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" \
  --bottleneck-dim 64 \
  --output-dir "$ROOT/results/reconstruction"
"$PYTHON" -m rtp_codec.benchmarks.pgen.collect_external_pgen \
  --runs-dir "$ROOT/pgen" \
  --output-dir "$ROOT/results/pgen"
find "$ROOT/results" -type f ! -name HASHES.sha256 ! -name HASHES.sha256.tmp -print0 | sort -z | xargs -0 sha256sum > "$ROOT/results/HASHES.sha256.tmp"
mv "$ROOT/results/HASHES.sha256.tmp" "$ROOT/results/HASHES.sha256"
echo "status=accepted"
