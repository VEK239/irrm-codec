#!/bin/bash -l
#SBATCH --job-name=irrm-sceptr-pca128-sum
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=summary-%j.out
#SBATCH --error=summary-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/research/results/reconstruction" "$ROOT/research/results/pgen"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.reconstruction.collect_external_reconstruction \
  --runs-dir "$ROOT/reconstruction" --output-dir "$ROOT/research/results/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" --bottleneck-dim 128
"$PYTHON" -m rtp_codec.benchmarks.pgen.collect_external_pgen \
  --runs-dir "$ROOT/pgen" --output-dir "$ROOT/research/results/pgen"
sha256sum "$ROOT/research/results/reconstruction"/* "$ROOT/research/results/pgen"/* \
  > "$ROOT/research/results/HASHES.sha256"
echo status=accepted
