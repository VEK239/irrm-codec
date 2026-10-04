#!/bin/bash -l
#SBATCH --job-name=irrm-latent-suff
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$ROOT/experiments/trb/latent-sufficiency"
OUTPUT="$BASE/evaluation-${SLURM_JOB_ID}"
DATA="$ROOT/data/benchmark/trb"
TOKENIZER="$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
PYTHON="/home/evlasova/.conda/envs/irrm-codec/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

test -x "$PYTHON"
test -f "$DATA/READY.json"
test -f "$DATA/target_standardizer.npz"
test -f "$TOKENIZER"
test ! -e "$OUTPUT"
mkdir -p "$BASE"
cd "$ROOT"

"$PYTHON" -m compileall -q benchmark/evaluate_latent_sufficiency.py tests/test_latent_sufficiency.py
"$PYTHON" - <<'PY'
import runpy
scope = runpy.run_path("tests/test_latent_sufficiency.py")
for name in sorted(scope):
    if name.startswith("test_"):
        scope[name]()
print("direct latent-sufficiency tests passed")
PY

"$PYTHON" -m benchmark.evaluate_latent_sufficiency \
  --data-dir "$DATA" \
  --normalizer "$DATA/target_standardizer.npz" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUTPUT" \
  --model "r:$ROOT/experiments/trb/data-anchor-r-seed42/best.pt:$ROOT/experiments/trb/data-anchor-r-seed42/run_config.json" \
  --model "t:$ROOT/experiments/trb/data-anchor-t-seed42/best.pt:$ROOT/experiments/trb/data-anchor-t-seed42/run_config.json" \
  --model "p:$ROOT/experiments/trb/data-anchor-p-seed42/best.pt:$ROOT/experiments/trb/data-anchor-p-seed42/run_config.json" \
  --model "rt:$ROOT/experiments/trb/data-anchor-rt-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rt-seed42/run_config.json" \
  --model "rp:$ROOT/experiments/trb/data-anchor-rp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rp-seed42/run_config.json" \
  --model "tp:$ROOT/experiments/trb/data-anchor-tp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-tp-seed42/run_config.json" \
  --model "rtp:$ROOT/experiments/trb/anchor-data-rtp-seed42/best.pt:$ROOT/experiments/trb/anchor-data-rtp-seed42/run_config.json" \
  --batch-size 256 --alphas 0.01,1,100 --bootstrap 2000 --seed 42

test -s "$OUTPUT/RESULTS.json"
test -s "$OUTPUT/summary.tsv"
test -s "$OUTPUT/OUTPUT_HASHES.json"
