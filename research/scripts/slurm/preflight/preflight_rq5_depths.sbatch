#!/bin/bash -l
#SBATCH --job-name=irrm-rq5-depth-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=preflight-%j.out
#SBATCH --error=preflight-%j.err

set -euo pipefail
ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${RTP_CODEC_PYTHON:-$(command -v python)}"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m rtp_codec.experiments.preflight.preflight_rq5_depths \
  --baseline-run-config "$ROOT_DIR/experiments/trb/rq5-char-rtp-d128-seed42/run_config.json" \
  --d128-test-metrics "$ROOT_DIR/experiments/trb/rq5-char-rtp-d128-seed42/test_metrics.json" \
  --d320-test-metrics "$ROOT_DIR/experiments/trb/rq1-char-rtp-seed42/test_metrics.json" \
  --d512-test-metrics "$ROOT_DIR/experiments/trb/rq5-char-rtp-d512-seed42/test_metrics.json" \
  --data-dir "$ROOT_DIR/data/benchmark/trb" \
  --output-path "$ROOT_DIR/experiments/trb/rq5-char-rtp-d128-depths-seed42/preflight.json"
