#!/bin/bash -l
#SBATCH --job-name=irrm-rq1-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/phase1-rq1-char-seed42/preflight-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/phase1-rq1-char-seed42/preflight-%j.err

set -euo pipefail

ROOT_DIR="/home/evlasova/irrm-codec"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

/home/evlasova/.conda/envs/irrm-codec/bin/python scripts/preflight_phase1_rq1.py \
  --data-dir "$ROOT_DIR/data/benchmark/trb" \
  --output-path "$ROOT_DIR/experiments/trb/phase1-rq1-char-seed42/preflight.json"
