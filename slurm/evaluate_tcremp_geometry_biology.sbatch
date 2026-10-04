#!/bin/bash -l
#SBATCH --job-name=tcremp-geom-bio
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
VARIANT="${1:?Variant centered or relational required}"
case "$VARIANT" in centered|relational) ;; *) exit 2 ;; esac
ROOT="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL="$ROOT/experiments/trb/tcremp-geometry-t-${VARIANT}-seed42"
OUT="$ROOT/experiments/trb/vdjdb-tcremp-geometry-${VARIANT}-${SLURM_JOB_ID}"
COHORT="$ROOT/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv"
TOKENIZER="$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
test -f "$MODEL/best.pt"
mkdir -p "$OUT"
cd "$ROOT"
PYTHONPATH="$ROOT" "$PYTHON" benchmark/extract_vdjdb_latent.py \
  --cohort "$COHORT" --checkpoint "$MODEL/best.pt" --tokenizer "$TOKENIZER" \
  --output "$OUT/embeddings_t.npy" --batch-size 512 --device cuda
PYTHONPATH="$ROOT" "$PYTHON" benchmark/evaluate_vdjdb_decoded_tcremp.py \
  --cohort "$COHORT" --rtp "$OUT/embeddings_t.npy" \
  --checkpoint "$MODEL/best.pt" --standardizer "$MODEL/target_standardizer.npz" \
  --uniform-evaluation "$ROOT/experiments/trb/vdjdb-uniform100-v1/evaluation-1427813" \
  --existing-separation "$ROOT/experiments/trb/vdjdb-uniform100-v1/separation-1428106" \
  --native-tcremp "$ROOT/experiments/trb/vdjdb-uniform100-v1/tcremp-1429926/tcremp.npy" \
  --output-dir "$OUT/decoded_eval" --batch-size 256 --device cuda
