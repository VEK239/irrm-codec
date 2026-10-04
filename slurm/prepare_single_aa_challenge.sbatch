#!/bin/bash -l
#SBATCH --job-name=irrm-aa1-targets
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PREFLIGHT="${1:?Accepted preflight directory required}"
ROOT="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BASE="$ROOT/experiments/trb/single-aa-challenge"
OUT="$BASE/targets-${SLURM_JOB_ID}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
TCREMP_PYTHON=/home/evlasova/.conda/envs/vdjdb-redcea/bin/python
TCREMP_ROOT=/home/evlasova/tcremp-git/tcremp
PROTOTYPES="$TCREMP_ROOT/tcremp/resources/tcremp_prototypes_TRB.tsv"
MIRPY_PROVENANCE=/home/evlasova/.conda/envs/vdjdb-redcea/lib/python3.11/site-packages/mir-1.0.0.dist-info/direct_url.json
test "$("$PYTHON" -c "import json; print(json.load(open('$PREFLIGHT/PREFLIGHT.json'))['status'])")" = accepted
test ! -e "$OUT"
mkdir -p "$OUT/raw_tcremp"
test "$(git -C "$TCREMP_ROOT" rev-parse HEAD)" = 372bbb64aa9572e1e2ba04a0060a9c03cd7796c6
test "$(sha256sum "$PROTOTYPES" | cut -d' ' -f1)" = 38cbb5e75d900ef61476bf3af2ed2ff0150a05988bfe568946cd3de5e167f36b
grep -q 468ddae2107da212bb0db9497a3dc9cc44b00eaf "$MIRPY_PROVENANCE"
cd "$ROOT"
"$PYTHON" -m benchmark.prepare_single_aa_challenge \
  --data-dir "$ROOT/data/benchmark/trb" --output-dir "$OUT/prepared" \
  --max-parents 2000 --variants-per-parent 3 --seed 42
PYTHONPATH="$TCREMP_ROOT:$ROOT" "$TCREMP_PYTHON" -m tcremp.tcremp_run \
  -i "$OUT/prepared/mutants_airr.tsv" -o "$OUT/raw_tcremp" -e mutants \
  -c TRB -s HomoSapiens -x clone_id -p "$PROTOTYPES" -np 32 \
  -llen 4 -hlen 41 --skip-clustering
"$PYTHON" -m irrm_codec.calc_pgen_1mm \
  --airr-path "$OUT/prepared/mutants_airr.tsv" --output-path "$OUT/mutants_pgen.tsv" \
  --chain TRB --species human --locus beta --clone-id-col clone_id \
  --cdr3-col junction_aa --threads 32 --chunk-size 50
"$PYTHON" -m benchmark.validate_single_aa_targets \
  --prepared "$OUT/prepared" --pgen "$OUT/mutants_pgen.tsv" \
  --representations "$OUT/raw_tcremp/mutants_tcremp_representations.tsv" \
  --tcremp "$OUT/raw_tcremp/mutants_tcremp.parquet" --output-dir "$OUT/targets"
sha256sum "$OUT/prepared/PREPARED.json" "$OUT/targets/TARGETS.json" > "$OUT/SHA256SUMS"
cp "$MIRPY_PROVENANCE" "$OUT/mirpy_direct_url.json"
sha256sum "$TCREMP_PYTHON" "$MIRPY_PROVENANCE" "$PROTOTYPES" >> "$OUT/SHA256SUMS"
