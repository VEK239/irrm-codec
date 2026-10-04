#!/bin/bash -l
#SBATCH --job-name=vdjdb-u100-sep
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=separation-%j.out
#SBATCH --error=separation-%j.err

set -euo pipefail
repo=/home/evlasova/irrm-codec
out=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/separation-${SLURM_JOB_ID}
cd "$repo"
${RTP_CODEC_PYTHON:-python} -m rtp_codec.benchmarks.downstream.evaluate_vdjdb_uniform100_separation \
  --cohort /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv \
  --native-evaluation /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728 \
  --external-representations /home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/vdjdb/representations \
  --uniform-evaluation /home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/evaluation-1427813 \
  --output-dir "$out"
