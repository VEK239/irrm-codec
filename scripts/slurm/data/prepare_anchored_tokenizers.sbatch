#!/bin/bash -l
#SBATCH --job-name=irrm-anchor-preflight
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER_ROOT="$ROOT_DIR/experiments/trb/tokenizers/anchored-seed42"
SOURCE_ROOT="/projects/immunestatus/vdjrearm"
PYTHON_BIN="${RTP_CODEC_PYTHON:-$(command -v python)}"

test -x "$PYTHON_BIN"
test -f "$DATA_DIR/READY.json"
test ! -e "$TOKENIZER_ROOT/READY.json"
mkdir -p "$TOKENIZER_ROOT"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m rtp_codec.experiments.tokenizers.prepare_anchored_tokenizers \
  --data-dir "$DATA_DIR" \
  --output-root "$TOKENIZER_ROOT" \
  --external-corpus "$SOURCE_ROOT/sample/TRB_1e7.tsv.gz" \
  --generation-script "$SOURCE_ROOT/scripts/sample_murugan.sh" \
  --olga-model-dir "$SOURCE_ROOT/model/Homo+sapiens/TRB_orig" \
  --external-rows 500000 --seed 42
"$PYTHON_BIN" -m rtp_codec.experiments.preflight.preflight_anchored_tokenizers \
  --data-dir "$DATA_DIR" \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --baseline-run-config "$ROOT_DIR/experiments/trb/rq5-char-rtp-d128-seed42/run_config.json" \
  --output-path "$TOKENIZER_ROOT/preflight.json"
"$PYTHON_BIN" -c 'import json,pathlib; p=pathlib.Path("/home/evlasova/irrm-codec/experiments/trb/tokenizers/anchored-seed42"); r=json.loads((p/"preflight.json").read_text()); assert r["status"]=="ready"; (p/"READY.json").write_text(json.dumps({"status":"ready","preflight_job_id":__import__("os").environ["SLURM_JOB_ID"]},indent=2))'
