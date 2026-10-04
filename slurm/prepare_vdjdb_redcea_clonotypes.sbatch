#!/bin/bash -l
#SBATCH --job-name=irrm-redcea-pre
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$ROOT/experiments/trb/vdjdb-redcea-clonotypes"
OUTPUT="$BASE/preflight-${SLURM_JOB_ID}"
SOURCE="$OUTPUT/cluster_members_TRB.txt"
PYTHON="/home/evlasova/.conda/envs/irrm-codec/bin/python"
COMMIT="77961e2bec255311176d5168b4b8534fdbee24e4"
BLOB="904320e903c646dc266639eb570461893d29aa2d"
URL="https://raw.githubusercontent.com/antigenomics/vdjdb-motifs/${COMMIT}/results/redcea/cluster_members_TRB.txt"

test -x "$PYTHON"
test -f "$ROOT/data/benchmark/trb/READY.json"
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
cd "$ROOT"
"$PYTHON" -m compileall -q \
  benchmark/prepare_vdjdb_redcea_clonotypes.py \
  benchmark/evaluate_vdjdb_redcea_clonotypes.py
"$PYTHON" -c "import runpy; m=runpy.run_path('tests/test_vdjdb_redcea_clonotypes.py'); m['test_preparation_deduplicates_removes_ambiguity_overlap_and_applies_support']()"
"$PYTHON" -c "import runpy; m=runpy.run_path('tests/test_vdjdb_redcea_evaluation.py'); m['test_perfect_epitope_geometry_scores_perfect_retrieval']()"
curl --fail --location --retry 3 --output "$SOURCE" "$URL"
ACTUAL_BLOB="$(git hash-object "$SOURCE")"
test "$ACTUAL_BLOB" = "$BLOB"
"$PYTHON" -m benchmark.prepare_vdjdb_redcea_clonotypes \
  --cluster-members "$SOURCE" \
  --benchmark-dir "$ROOT/data/benchmark/trb" \
  --output-dir "$OUTPUT/cohort" \
  --min-clonotypes 20 \
  --max-length 40 \
  --source-git-commit "$COMMIT" \
  --source-git-blob "$BLOB"
