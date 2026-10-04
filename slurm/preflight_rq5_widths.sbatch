#!/bin/bash -l
#SBATCH --job-name=irrm-rq5-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/rq5-char-rtp-widths-seed42/preflight-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/rq5-char-rtp-widths-seed42/preflight-%j.err

set -euo pipefail
ROOT_DIR="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${IRRM_PYTHON:-$(command -v python)}"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m scripts.preflight_rq5_widths \
  --baseline-run-config "$ROOT_DIR/experiments/trb/rq1-char-rtp-seed42/run_config.json" \
  --baseline-test-metrics "$ROOT_DIR/experiments/trb/rq1-char-rtp-seed42/test_metrics.json" \
  --data-dir "$ROOT_DIR/data/benchmark/trb" \
  --output-path "$ROOT_DIR/experiments/trb/rq5-char-rtp-widths-seed42/preflight.json"
