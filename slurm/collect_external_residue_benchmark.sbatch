#!/bin/bash -l
#SBATCH --job-name=irrm-res-collect
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-v1/logs/collect-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-v1/logs/collect-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$REPO"
"$PYTHON" benchmark/collect_external_residue_benchmark.py \
  --reconstruction-dir "$ROOT/reconstruction" \
  --pgen-dir "$ROOT/pgen" \
  --output-dir "$ROOT/results"
find "$ROOT/results" -maxdepth 1 -type f ! -name HASHES.sha256 -print0 | sort -z | xargs -0 sha256sum > "$ROOT/results/HASHES.sha256"
cat "$ROOT/results/HASHES.sha256"
echo "status=complete"
