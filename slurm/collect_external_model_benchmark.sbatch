#!/bin/bash
#SBATCH --job-name=irrm-ext-collect
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/collect-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/collect-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/results/reconstruction" "$ROOT/results/pgen"
cd "$REPO"

"$PYTHON" benchmark/collect_external_reconstruction.py \
  --runs-dir "$ROOT/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" \
  --bottleneck-dim 64 \
  --output-dir "$ROOT/results/reconstruction"
"$PYTHON" benchmark/collect_external_pgen.py \
  --runs-dir "$ROOT/pgen" \
  --output-dir "$ROOT/results/pgen"
find "$ROOT/results" -type f ! -name HASHES.sha256 ! -name HASHES.sha256.tmp -print0 | sort -z | xargs -0 sha256sum > "$ROOT/results/HASHES.sha256.tmp"
mv "$ROOT/results/HASHES.sha256.tmp" "$ROOT/results/HASHES.sha256"
echo "status=accepted"
