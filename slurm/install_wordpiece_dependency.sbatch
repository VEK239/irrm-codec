#!/bin/bash -l
#SBATCH --job-name=irrm-wp-dependency
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/tokenizers/wordpiece-seed42/slurm-%x-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/tokenizers/wordpiece-seed42/slurm-%x-%j.err

set -euo pipefail
PYTHON_BIN="/home/evlasova/.conda/envs/irrm-codec/bin/python"
"$PYTHON_BIN" -m pip install --disable-pip-version-check "tokenizers==0.23.1"
"$PYTHON_BIN" -c 'import tokenizers; assert tokenizers.__version__ == "0.23.1"; print(tokenizers.__version__)'
