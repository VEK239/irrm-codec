#!/bin/bash -l
#SBATCH --job-name=vdjdb-tcremp-geom
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
cd /home/evlasova/irrm-codec
PYTHONPATH="${RTP_CODEC_ROOT:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" ${RTP_CODEC_PYTHON:-python} -m rtp_codec.benchmarks.downstream.diagnose_vdjdb_decoded_tcremp_geometry
