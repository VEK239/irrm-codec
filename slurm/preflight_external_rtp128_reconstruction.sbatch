#!/bin/bash -l
#SBATCH --job-name=irrm-rtp128-pre
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/native-rtp128-v1/logs/preflight-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/native-rtp128-v1/logs/preflight-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/native-rtp128-v1"
SOURCE="$BASE/cohort/representations/rtp.npy"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/bottleneck" "$ROOT/reconstruction" "$ROOT/results"
cd "$REPO"

"$PYTHON" -m py_compile \
  benchmark/train_external_model_decoder.py \
  benchmark/collect_external_reconstruction.py
bash -n \
  slurm/train_external_rtp128_reconstruction.sbatch \
  slurm/collect_external_rtp128_reconstruction.sbatch

"$PYTHON" - "$SOURCE" "$BASE/cohort" "$ROOT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irrm_codec.inverse_model import InverseModel

source, cohort, root = map(Path, sys.argv[1:])
matrix = np.load(source, mmap_mode="r")
assert matrix.shape == (99_430, 128), matrix.shape
for start in range(0, len(matrix), 8192):
    assert np.isfinite(np.asarray(matrix[start:start + 8192])).all()

splits = {}
sets = {}
for name in ("train", "val", "test"):
    rows = pd.read_csv(cohort / "manifests" / f"{name}.tsv", sep="\t")["row_index"].to_numpy()
    splits[name] = int(len(rows))
    sets[name] = set(map(int, rows))
assert splits == {"train": 79_544, "val": 9_943, "test": 9_943}, splits
assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
assert len(sets["train"] | sets["val"] | sets["test"]) == 99_430

smoke = InverseModel(embedding_dim=128, max_len=40)(torch.zeros(2, 128))
assert smoke.shape == (2, 40, 25) and torch.isfinite(smoke).all()

target = root / "bottleneck" / "rtp_128.npy"
if target.is_symlink() or target.exists():
    target.unlink()
target.symlink_to(source)
assert target.resolve() == source.resolve()

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = {
    "status": "accepted",
    "representation": "rtp",
    "interface": "native_128d_no_projection",
    "source": str(source),
    "source_sha256": sha256(source),
    "shape": list(matrix.shape),
    "dtype": str(matrix.dtype),
    "finite": True,
    "splits": splits,
    "split_overlap": 0,
    "decoder": "independently_trained_InverseModel",
    "seeds": [42, 43, 44],
    "environment_python": sys.executable,
}
(root / "PREFLIGHT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY

sha256sum \
  benchmark/train_external_model_decoder.py \
  benchmark/collect_external_reconstruction.py \
  slurm/preflight_external_rtp128_reconstruction.sbatch \
  slurm/train_external_rtp128_reconstruction.sbatch \
  slurm/collect_external_rtp128_reconstruction.sbatch \
  "$ROOT/PREFLIGHT.json" > "$ROOT/PREFLIGHT_HASHES.sha256"
echo status=accepted
