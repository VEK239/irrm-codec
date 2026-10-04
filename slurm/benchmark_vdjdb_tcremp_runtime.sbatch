#!/bin/bash -l
#SBATCH --job-name=vdjdb-tcremp-runtime
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail
THREADS="${1:?Pass the requested TCRemP worker count (1 or 16)}"
case "$THREADS" in 1|16) ;; *) echo "threads must be 1 or 16" >&2; exit 2;; esac
ROOT="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="$ROOT/experiments/trb/vdjdb-runtime-benchmark/tcremp-${THREADS}cpu-${SLURM_JOB_ID}"
TCREMP_ROOT=/home/evlasova/tcremp-git/tcremp
PYTHON=/home/evlasova/.conda/envs/vdjdb-redcea/bin/python
mkdir -p "$OUT/raw_tcremp"
cd "$ROOT"
STARTED="$(date -Iseconds)"
SECONDS=0
PYTHONPATH="$TCREMP_ROOT:$ROOT" "$PYTHON" -m tcremp.tcremp_run \
  -i "$ROOT/experiments/trb/vdjdb-uniform100-v1/tcremp-1429926/cohort.airr.tsv" \
  -o "$OUT/raw_tcremp" -e cohort -c TRB -s HomoSapiens -x clone_id \
  -p "$TCREMP_ROOT/tcremp/resources/tcremp_prototypes_TRB.tsv" \
  -np "$THREADS" -llen 4 -hlen 41 --skip-clustering 2>&1 | tee "$OUT/tcremp.log"
python - "$OUT/RESULTS.json" "$THREADS" "$SECONDS" "$STARTED" <<'PY'
import json, sys
from pathlib import Path
path, threads, seconds, started = map(str, sys.argv[1:])
Path(path).write_text(json.dumps({"status": "complete", "threads": int(threads), "wall_seconds_including_load_and_write": int(seconds), "started": started}, indent=2) + "\n")
PY
