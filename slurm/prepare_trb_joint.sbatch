#!/bin/bash -l
#SBATCH --job-name=irrm-trb-prepare
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/evlasova/irrm-codec/data/benchmark/trb-prepare-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/data/benchmark/trb-prepare-%j.err

set -euo pipefail

ROOT_DIR="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_DIR="$ROOT_DIR/data/benchmark/trb"
cd "$ROOT_DIR"

/home/evlasova/.conda/envs/irrm-codec/bin/python -m benchmark.prepare_trb_joint \
  --airr-path /projects/immunestatus/vdjrearm/airr_format/trb_background_100k.tsv \
  --embeddings-path /projects/immunestatus/vdjrearm/tcremp/trb_background_100k_tcremp.parquet \
  --representations-path /projects/immunestatus/vdjrearm/tcremp/trb_background_100k_tcremp_representations.tsv \
  --pgen-path /projects/immunestatus/vdjrearm/pgen_1mm_background_100k/trb_background_100k_pgen.tsv \
  --tcremp-writer-source /home/evlasova/tcremp-git/tcremp/tcremp/tcremp_run.py \
  --output-dir "$OUTPUT_DIR" \
  --seed 42 \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --max-len 40 \
  --embedding-batch-size 256 \
  --standardizer-chunk-size 1024
