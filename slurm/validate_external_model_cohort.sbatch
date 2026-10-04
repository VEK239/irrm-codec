#!/bin/bash
#SBATCH --job-name=irrm-ext-validate
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/validate-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/validate-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
RUNTIME="$REPO/experiments/trb/external-model-benchmark/runtime-v2"
ROOT="$REPO/experiments/trb/external-model-benchmark"
COHORT="$ROOT/cohort"
export PYTHONPATH="$RUNTIME/site-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$REPO"

"$PYTHON" -m py_compile \
  benchmark/prepare_external_model_cohort.py \
  benchmark/cache_external_model_embeddings.py \
  benchmark/fit_external_model_bottleneck.py \
  benchmark/train_external_model_decoder.py \
  benchmark/train_external_model_pgen.py

"$PYTHON" - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import esm
import sceptr
import transformers
import benchmark.cache_external_model_embeddings
import benchmark.fit_external_model_bottleneck
import benchmark.train_external_model_decoder
import benchmark.train_external_model_pgen

root = Path("/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/cohort")
report = json.loads((root / "COHORT.json").read_text())
catalog = json.loads((root / "representations/representations.json").read_text())
assert report["status"] == "accepted" and report["common_rows"] == 99430
assert report["dropped_total"] == 0 and report["split_overlap"] == 0
expected = {"onehot": (99430, 1000), "tcremp": (99430, 9000), "rtp": (99430, 128)}
for name, shape in expected.items():
    matrix = np.load(catalog[name]["path"], mmap_mode="r")
    assert matrix.shape == shape, (name, matrix.shape, shape)
frame = pd.read_parquet(root / "dataset.parquet")
assert len(frame) == 99430 and frame.junction_aa.nunique() == 99430
print(json.dumps({
    "status": "accepted",
    "rows": len(frame),
    "split_counts": report["split_counts"],
    "representations": {name: list(shape) for name, shape in expected.items()},
    "torch": torch.__version__,
    "esm": esm.__version__,
    "transformers": transformers.__version__,
    "sceptr_import": True,
}, indent=2, sort_keys=True))
PY
echo "status=accepted"
