#!/bin/bash
#SBATCH --job-name=irrm-author-global-gate
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/preflight-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/preflight-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-global-author-v1"
RUNTIME="$REPO/experiments/trb/external-model-benchmark/runtime-v2"
LOCKED="$REPO/experiments/trb/external-model-benchmark/cohort/dataset.parquet"
REDCEA="$REPO/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv"
OUT="$BASE/preflight-$SLURM_JOB_ID"

export PYTHONPATH="$RUNTIME/site-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME/huggingface"
export TORCH_HOME="$RUNTIME/torch"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$BASE/logs"
cd "$REPO"

"$PYTHON" -m py_compile benchmark/cache_external_model_embeddings.py benchmark/preflight_author_global_embeddings.py benchmark/train_external_model_pgen.py
"$PYTHON" benchmark/preflight_author_global_embeddings.py --locked-dataset "$LOCKED" --redcea-cohort "$REDCEA" --locked-sha256 44aa9f8f73c23c0e42ef2bc113a5e4ee3f49fcc335e3c84f0d2eacd59021c191 --redcea-sha256 b3e5852e6952b8f4f529a60287aa76d4308dae37151b7f2becfb4ff87bd98575 --output-dir "$OUT"
sha256sum "$OUT/PREFLIGHT.json" > "$OUT/HASHES.sha256"
echo "status=accepted output=$OUT"
