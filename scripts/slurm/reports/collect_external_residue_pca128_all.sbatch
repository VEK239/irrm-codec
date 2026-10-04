#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca128-sum
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=summary-%j.out
#SBATCH --error=summary-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-v1"
SCEPTR_ROOT="$BASE/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/results/reconstruction" "$ROOT/results/pgen"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.reconstruction.collect_external_reconstruction \
  --runs-dir "$ROOT/reconstruction" --output-dir "$ROOT/results/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" --bottleneck-dim 128
"$PYTHON" -m rtp_codec.benchmarks.pgen.collect_external_pgen \
  --runs-dir "$ROOT/pgen" --output-dir "$ROOT/results/pgen"
"$PYTHON" -m rtp_codec.benchmarks.reporting.combine_external_pca128_results \
  --main-root "$ROOT" --sceptr-root "$SCEPTR_ROOT"
sha256sum "$ROOT/results/reconstruction"/* "$ROOT/results/pgen"/* \
  "$ROOT/results/all_baselines"/* > "$ROOT/results/HASHES.sha256"
echo status=accepted
