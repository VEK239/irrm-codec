#!/bin/bash
#SBATCH --job-name=irrm-ext-runtime
#SBATCH --partition=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/runtime-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/runtime-%j.err

set -euo pipefail

REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark/runtime-v2"
SITE="$ROOT/site-packages"
HF_HOME="$ROOT/huggingface"
TORCH_HOME="$ROOT/torch"
REPORT="$ROOT/RUNTIME.json"

mkdir -p "$REPO/experiments/trb/external-model-benchmark/logs" "$SITE" "$HF_HOME" "$TORCH_HOME"

# The student report named the model/checkpoint versions but did not pin the
# transformers/fair-esm library builds.  Pin compatible library versions here and record
# the complete installed distribution inventory below.  --target keeps the shared conda
# environment unchanged; its Python executable remains the runtime.
"$PYTHON" -m pip install \
  --disable-pip-version-check \
  --target "$SITE" \
  --upgrade \
  --no-deps \
  "sceptr==1.2.0" \
  "fair-esm==2.0.0" \
  "transformers==4.44.2" \
  "libtcrlm==1.1.3" \
  "tidytcells==2.2.1" \
  "blosum==2.2.0" \
  "tokenizers==0.19.1" \
  "huggingface-hub==0.36.2" \
  "safetensors==0.4.5" \
  "regex==2024.11.6" \
  "requests==2.32.3" \
  "filelock==3.16.1" \
  "packaging==24.2" \
  "PyYAML==6.0.2" \
  "tqdm==4.67.1" \
  "fsspec==2024.10.0" \
  "urllib3==2.2.3" \
  "charset-normalizer==3.4.0" \
  "chardet==5.2.0" \
  "idna==3.10" \
  "certifi==2024.8.30" \
  "rapidfuzz==3.10.1"

export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME TORCH_HOME

"$PYTHON" - <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
import os
import pathlib
import sys

root = pathlib.Path(os.environ["HF_HOME"]).parent
packages = [
    "sceptr", "fair-esm", "transformers", "tokenizers", "huggingface-hub",
    "safetensors", "rapidfuzz",
]
modules = {}
for name in ("torch", "sceptr", "esm", "transformers"):
    module = importlib.import_module(name)
    modules[name] = {
        "version": getattr(module, "__version__", None),
        "path": str(pathlib.Path(module.__file__).resolve()),
    }

distributions = {}
for package in packages:
    try:
        distributions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        distributions[package] = None

report = {
    "status": "accepted",
    "python": sys.executable,
    "site_packages": str(root / "site-packages"),
    "hf_home": os.environ["HF_HOME"],
    "torch_home": os.environ["TORCH_HOME"],
    "modules": modules,
    "distributions": distributions,
    "provenance_note": (
        "Student report pins SCEPTR 1.2.0, esm2_t6_8M_UR50D and "
        "wukevin/tcr-bert but not fair-esm/transformers library versions; "
        "this rerun pins compatible library builds explicitly."
    ),
}
if modules["torch"]["version"] != "2.4.1+cu121" or "/.conda/envs/irrm-codec/" not in modules["torch"]["path"]:
    raise RuntimeError(f"isolated packages shadowed validated conda torch: {modules['torch']}")
(root / "RUNTIME.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
PY

cd "$REPO"
sha256sum slurm/install_external_benchmark_runtime.sbatch "$REPORT" > "$ROOT/HASHES.sha256"
echo "status=accepted"
echo "runtime=$ROOT"
