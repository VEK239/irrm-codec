#!/bin/bash -l
#SBATCH --job-name=irrm-rtp-errana
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"; export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
${RTP_CODEC_PYTHON:-python} -m rtp_codec.benchmarks.analysis.evaluate_rtp_error_analysis --data-dir "$ROOT_DIR/data/benchmark/trb" --tokenizer "$ROOT_DIR/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json" --checkpoint "$ROOT_DIR/experiments/trb/anchor-data-rtp-seed42/best.pt" --output-dir "$ROOT_DIR/experiments/trb/rtp-data-anchor-error-analysis-$SLURM_JOB_ID" --batch-size 64 --num-workers 8 --device cuda
