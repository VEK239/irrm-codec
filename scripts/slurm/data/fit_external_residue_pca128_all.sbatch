#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca128
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%2
#SBATCH --output=pca-%A-%a.out
#SBATCH --error=pca-%A-%a.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-v1"
REPRESENTATIONS=(esm2_8m tcr_bert sceptr)
REPRESENTATION="${REPRESENTATIONS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/bottleneck"
cd "$REPO"
"$PYTHON" -m rtp_codec.benchmarks.representations.fit_external_residue_pca256 \
  --dataset-dir "$BASE/cohort" --residue-dir "$BASE/residue-v1/representations" \
  --output-dir "$ROOT/bottleneck" --representation "$REPRESENTATION" \
  --components 128 --oversample 32 --power-iterations 4 --batch-size 2048 --seed 42
echo "status=accepted representation=$REPRESENTATION"
