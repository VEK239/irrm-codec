#!/bin/bash -l
#SBATCH --job-name=irrm-res-collect
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=collect-%j.out
#SBATCH --error=collect-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.reporting.collect_external_residue_benchmark \
  --reconstruction-dir "$ROOT/reconstruction" \
  --pgen-dir "$ROOT/pgen" \
  --output-dir "$ROOT/results"
find "$ROOT/results" -maxdepth 1 -type f ! -name HASHES.sha256 -print0 | sort -z | xargs -0 sha256sum > "$ROOT/research/results/HASHES.sha256"
cat "$ROOT/research/results/HASHES.sha256"
echo "status=complete"
