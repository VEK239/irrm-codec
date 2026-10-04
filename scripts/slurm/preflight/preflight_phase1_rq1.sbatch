#!/bin/bash -l
#SBATCH --job-name=irrm-rq1-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=preflight-%j.out
#SBATCH --error=preflight-%j.err

set -euo pipefail

ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

${RTP_CODEC_PYTHON:-python} -m rtp_codec.experiments.preflight.preflight_phase1_rq1 \
  --data-dir "$ROOT_DIR/data/benchmark/trb" \
  --output-path "$ROOT_DIR/experiments/trb/phase1-rq1-char-seed42/preflight.json"
