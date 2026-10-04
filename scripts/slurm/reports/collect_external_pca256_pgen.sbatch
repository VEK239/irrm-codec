#!/bin/bash -l
#SBATCH --job-name=irrm-pca256-pgen-sum
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=pgen-summary-%j.out
#SBATCH --error=pgen-summary-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca256-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/results/pgen"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.pgen.collect_external_pgen --runs-dir "$ROOT/pgen" --output-dir "$ROOT/results/pgen"
sha256sum "$ROOT/results/pgen"/* > "$ROOT/results/pgen/HASHES.sha256"
echo status=accepted
