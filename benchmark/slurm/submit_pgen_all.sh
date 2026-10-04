#!/usr/bin/env bash
# Submit the whole Pgen benchmark as one dependency chain, so it can be started and left.
#
#   bash benchmark/slurm/submit_pgen_all.sh
#
# Stages, each waiting for the previous one to succeed:
#   1. pretrain the sequence-to-TCRemP encoder      (~10 min)
#   2. main sweep: 6 arms x 2 targets x 3 sizes x 3 seeds   (96 runs)
#   3. extra seeds for the two IRRM arms, 7 more each        (84 runs)
#
# Existing results are archived rather than left in place: run names are identical
# between submissions, so mixing a finished constant-learning-rate sweep with a new
# cosine-schedule one would silently average two different configurations together.

set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/tank/scratch/mpodsytnik/irrm-codec}"
REPO="${REPO:-/nfs/home/mpodsytnik/irrm-codec}"
PGEN_DIR="$SCRATCH/artifacts/benchmark/pgen"
FORWARD_DIR="$SCRATCH/artifacts/benchmark/forward_encoder"

mkdir -p "$SCRATCH/logs"

stamp="$(date +%Y%m%d_%H%M%S)"
for dir in "$PGEN_DIR" "$FORWARD_DIR"; do
  if [ -d "$dir" ]; then
    mv "$dir" "${dir}_constlr_${stamp}"
    echo "archived $(basename "$dir") -> $(basename "${dir}")_constlr_${stamp}"
  fi
done

cd "$REPO"

forward_id=$(sbatch --parsable benchmark/slurm/forward_encoder.sbatch)
echo "1/3 forward encoder  : $forward_id"

main_id=$(sbatch --parsable --dependency=afterok:"$forward_id" benchmark/slurm/pgen_array.sbatch)
echo "2/3 main sweep       : $main_id  (after $forward_id)"

# The extra seeds only cover 1k and 10k here; the main sweep already supplies three seeds
# per cell, and SUBSETS controls which sizes get topped up to ten.
seeds_id=$(sbatch --parsable --dependency=afterok:"$main_id" \
  --export=ALL,SUBSETS="1k 10k all" --array=0-83%2 benchmark/slurm/pgen_seeds_array.sbatch)
echo "3/3 extra seeds      : $seeds_id  (after $main_id)"

echo
echo "watch with:  squeue -u \$USER"
echo "count with:  ls $PGEN_DIR/*/metrics.json 2>/dev/null | wc -l   # expect 180"
