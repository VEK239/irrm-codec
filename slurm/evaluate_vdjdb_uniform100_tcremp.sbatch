#!/bin/bash -l
#SBATCH --job-name=vdjdb-tcremp
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=16:00:00

set -euo pipefail
ROOT=/home/evlasova/irrm-codec
OUT="$ROOT/experiments/trb/vdjdb-uniform100-v1/tcremp-${SLURM_JOB_ID}"
COHORT="$ROOT/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv"
PAIRS="$ROOT/experiments/trb/vdjdb-uniform100-v1/evaluation-1427813/uniform_pairs.npz"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
TCREMP_PYTHON=/home/evlasova/.conda/envs/vdjdb-redcea/bin/python
TCREMP_ROOT=/home/evlasova/tcremp-git/tcremp
mkdir -p "$OUT/raw_tcremp"
cd "$ROOT"
"$PYTHON" - "$COHORT" "$OUT/cohort.airr.tsv" <<'PY'
import sys
import pandas as pd
x = pd.read_csv(sys.argv[1], sep="\t")
pd.DataFrame({"clone_id": range(len(x)), "junction_aa": x.cdr3, "v_call": x.v_call, "j_call": x.j_call, "locus": "TRB"}).to_csv(sys.argv[2], sep="\t", index=False)
PY
PYTHONPATH="$TCREMP_ROOT:$ROOT" "$TCREMP_PYTHON" -m tcremp.tcremp_run -i "$OUT/cohort.airr.tsv" -o "$OUT/raw_tcremp" -e cohort -c TRB -s HomoSapiens -x clone_id -p "$TCREMP_ROOT/tcremp/resources/tcremp_prototypes_TRB.tsv" -np "$SLURM_CPUS_PER_TASK" -llen 4 -hlen 41 --skip-clustering
"$PYTHON" - "$COHORT" "$PAIRS" "$OUT/raw_tcremp/cohort_tcremp.parquet" "$OUT" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark.evaluate_vdjdb_uniform100_separation import separation_metrics
cohort, pairs, raw, out = map(Path, sys.argv[1:])
x, p, r = pd.read_csv(cohort, sep="\t"), np.load(pairs), pd.read_parquet(raw)
v = r.drop(columns="clone_id").select_dtypes(include=[np.number]); v.index = r.clone_id.astype(int)
if v.shape[1] != 9000 or not v.index.is_unique or set(v.index) != set(range(len(x))): raise ValueError("Invalid TCRemP alignment")
m = v.loc[np.arange(len(x))].to_numpy(np.float32)
if not np.isfinite(m).all(): raise ValueError("Non-finite TCRemP")
np.save(out / "tcremp.npy", m, allow_pickle=False)
q, e = separation_metrics(m[p["original_indices"]], p["within"], p["between"], x.iloc[p["original_indices"]].label.astype(str).to_numpy())
e.insert(0, "model", "TCRemP prototypes")
e.to_csv(out / "per_epitope.tsv", sep="\t", index=False)
s = {"model": "TCRemP prototypes", "macro_auroc": float(e.auroc.mean()), "macro_cohen_d": float(e.cohen_d.mean()), "queries": len(q), "epitopes": len(e)}
pd.DataFrame([s]).to_csv(out / "summary.tsv", sep="\t", index=False)
(out / "RESULTS.json").write_text(json.dumps({"status": "complete", "summary": s, "matrix_shape": list(m.shape), "protocol": "cosine distance on Fig. 2 fixed uniform-100 pairs"}, indent=2) + "\n")
PY
