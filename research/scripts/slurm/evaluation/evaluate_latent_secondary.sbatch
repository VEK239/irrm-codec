#!/bin/bash -l
#SBATCH --job-name=irrm-latent-secondary
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
MODE="${1:?mode must be preflight or evaluate}"
PREFLIGHT_ROOT="${2:-}"
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$ROOT/experiments/trb/latent-secondary"
OUTPUT="$BASE/${MODE}-${SLURM_JOB_ID}"
DATA="$ROOT/data/benchmark/trb"
TOKENIZER="$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
PYTHON="${RTP_CODEC_PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

test -x "$PYTHON"
test -f "$DATA/READY.json"
test -f "$TOKENIZER"
test ! -e "$OUTPUT"
mkdir -p "$BASE"
cd "$ROOT"

COMMON=(
  --data-dir "$DATA"
  --tokenizer "$TOKENIZER"
  --output-dir "$OUTPUT"
  --model "r:$ROOT/experiments/trb/data-anchor-r-seed42/best.pt:$ROOT/experiments/trb/data-anchor-r-seed42/run_config.json"
  --model "t:$ROOT/experiments/trb/data-anchor-t-seed42/best.pt:$ROOT/experiments/trb/data-anchor-t-seed42/run_config.json"
  --model "p:$ROOT/experiments/trb/data-anchor-p-seed42/best.pt:$ROOT/experiments/trb/data-anchor-p-seed42/run_config.json"
  --model "rt:$ROOT/experiments/trb/data-anchor-rt-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rt-seed42/run_config.json"
  --model "rp:$ROOT/experiments/trb/data-anchor-rp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rp-seed42/run_config.json"
  --model "tp:$ROOT/experiments/trb/data-anchor-tp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-tp-seed42/run_config.json"
  --model "rtp:$ROOT/experiments/trb/anchor-data-rtp-seed42/best.pt:$ROOT/experiments/trb/anchor-data-rtp-seed42/run_config.json"
  --batch-size 128
  --seed 42
)

if [[ "$MODE" == "preflight" ]]; then
  "$PYTHON" -m compileall -q src/rtp_codec/benchmarks/analysis/evaluate_latent_secondary.py src/tests/test_latent_secondary.py
  "$PYTHON" - <<'PY'
import runpy
namespace = runpy.run_path('src/tests/test_latent_secondary.py')
for name, value in sorted(namespace.items()):
    if name.startswith('test_') and callable(value):
        value()
print('direct tests passed')
PY
  "$PYTHON" -m rtp_codec.benchmarks.analysis.evaluate_latent_secondary "${COMMON[@]}" --preflight-only
elif [[ "$MODE" == "evaluate" ]]; then
  test -n "$PREFLIGHT_ROOT"
  test -f "$PREFLIGHT_ROOT/PREFLIGHT.json"
  "$PYTHON" -m rtp_codec.benchmarks.analysis.evaluate_latent_secondary \
    "${COMMON[@]}" --accepted-preflight "$PREFLIGHT_ROOT/PREFLIGHT.json"
else
  echo "unsupported mode: $MODE" >&2
  exit 2
fi
