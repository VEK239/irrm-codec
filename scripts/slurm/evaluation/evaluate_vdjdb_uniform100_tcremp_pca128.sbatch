#!/bin/bash -l
#SBATCH --job-name=vdjdb-tcremp128
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=16:00:00

set -euo pipefail
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="$ROOT/experiments/trb/vdjdb-uniform100-v1/tcremp-pca128-${SLURM_JOB_ID}"
TRAIN_DATA="$ROOT/experiments/trb/external-model-benchmark/cohort"
TRAIN_TCREMP="$TRAIN_DATA/representations/tcremp.npy"
VDJDB_TCREMP="$ROOT/experiments/trb/vdjdb-uniform100-v1/tcremp-1429926/tcremp.npy"
COHORT="$ROOT/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv"
PAIRS="$ROOT/experiments/trb/vdjdb-uniform100-v1/evaluation-1427813/uniform_pairs.npz"
PYTHON="${RTP_CODEC_PYTHON:-python}"
mkdir -p "$OUT"
cd "$ROOT"
"$PYTHON" - "$TRAIN_TCREMP" "$TRAIN_DATA/manifests/train.tsv" "$VDJDB_TCREMP" "$COHORT" "$PAIRS" "$OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from rtp_codec.benchmarks.downstream.evaluate_vdjdb_uniform100_separation import separation_metrics
from rtp_codec.benchmarks.representations.fit_external_model_bottleneck import fit_projection, transform_all

train_matrix_path, train_rows_path, vdjdb_matrix_path, cohort_path, pairs_path, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
train_matrix = np.load(train_matrix_path, mmap_mode="r")
train_rows = pd.read_csv(train_rows_path, sep="\t")["row_index"].to_numpy(dtype=int)
vdjdb = np.load(vdjdb_matrix_path, mmap_mode="r")
if train_matrix.shape[1] != 9000 or vdjdb.shape != (68302, 9000):
    raise ValueError(f"Unexpected TCRemP shapes: train={train_matrix.shape}, VDJdb={vdjdb.shape}")
class Log:
    def info(self, *args): print(*args, flush=True)
scaler, pca = fit_projection(train_matrix, train_rows, 128, 4096, 42, Log())
dump({"scaler": scaler, "pca": pca, "fit_rows": len(train_rows), "source": str(train_matrix_path)}, out / "tcremp_pca128.joblib")
projected = transform_all(vdjdb, scaler, pca, 4096)
np.save(out / "tcremp_pca128.npy", projected, allow_pickle=False)
cohort = pd.read_csv(cohort_path, sep="\t")
pairs = np.load(pairs_path)
original = pairs["original_indices"]
query, epitope = separation_metrics(projected[original], pairs["within"], pairs["between"], cohort.iloc[original].label.astype(str).to_numpy())
query.insert(0, "model", "TCRemP prototypes PCA-128")
epitope.insert(0, "model", "TCRemP prototypes PCA-128")
query.to_parquet(out / "per_query.parquet", index=False)
epitope.to_csv(out / "per_epitope.tsv", sep="\t", index=False, lineterminator="\n")
summary = {"model": "TCRemP prototypes PCA-128", "macro_auroc": float(epitope.auroc.mean()), "macro_cohen_d": float(epitope.cohen_d.mean()), "queries": len(query), "epitopes": len(epitope)}
pd.DataFrame([summary]).to_csv(out / "summary.tsv", sep="\t", index=False, lineterminator="\n")
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()
(out / "RESULTS.json").write_text(json.dumps({"status": "complete", "summary": summary, "projection": {"method": "StandardScaler plus IncrementalPCA", "output_dim": 128, "fit_data": str(train_matrix_path), "fit_rows": len(train_rows), "fit_split": str(train_rows_path), "seed": 42}, "source_tcremp": {"path": str(vdjdb_matrix_path), "sha256": sha256(vdjdb_matrix_path), "shape": list(vdjdb.shape)}, "protocol": "cosine distance on the identical Fig. 2 fixed uniform-100 pairs"}, indent=2, sort_keys=True) + "\n")
PY
