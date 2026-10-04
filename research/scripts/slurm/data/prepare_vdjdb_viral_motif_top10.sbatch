#!/bin/bash -l
#SBATCH --job-name=irrm-viral10-pre
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
MODE="${1:-prepare}"
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SNAPSHOT="/projects/immunestatus/vdjdb/vdjdb-2025-07-30"
OUTPUT="$ROOT/experiments/trb/vdjdb-viral-motif-top10/${MODE}-${SLURM_JOB_ID}"
SPECIES="$ROOT/research/examples/data/viral_antigen_species_2025_07_30.txt"
PYTHON="${RTP_CODEC_PYTHON:-python}"
mkdir -p "$ROOT/experiments/trb/vdjdb-viral-motif-top10"
cd "$ROOT"
if test "$MODE" = "inventory"; then
  "$PYTHON" -m rtp_codec.benchmarks.datasets.prepare_vdjdb_viral_motif_top10 \
    --vdjdb-tsv "$SNAPSHOT/vdjdb.txt" --cluster-members "$SNAPSHOT/cluster_members.txt" \
    --motif-pwms "$SNAPSHOT/motif_pwms.txt" --benchmark-dir "$ROOT/data/benchmark/trb" \
    --output-dir "$OUTPUT" --inventory-only --min-score 1 --seed 42
else
  test -f "$SPECIES"
  "$PYTHON" -m rtp_codec.benchmarks.datasets.prepare_vdjdb_viral_motif_top10 \
    --vdjdb-tsv "$SNAPSHOT/vdjdb.txt" --cluster-members "$SNAPSHOT/cluster_members.txt" \
    --motif-pwms "$SNAPSHOT/motif_pwms.txt" --benchmark-dir "$ROOT/data/benchmark/trb" \
    --output-dir "$OUTPUT" --viral-species-list "$SPECIES" --min-score 1 \
    --requested-min-positives 40 --controls-per-positive 4 \
    --max-pairs-per-epitope 5000 --seed 42
fi
