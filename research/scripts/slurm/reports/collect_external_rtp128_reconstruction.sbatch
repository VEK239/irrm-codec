#!/bin/bash -l
#SBATCH --job-name=irrm-rtp128-sum
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=summary-%j.out
#SBATCH --error=summary-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/native-rtp128-v1"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/research/results/reconstruction"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.reconstruction.collect_external_reconstruction \
  --runs-dir "$ROOT/reconstruction" \
  --output-dir "$ROOT/research/results/reconstruction" \
  --bottleneck-dir "$ROOT/bottleneck" \
  --bottleneck-dim 128

"$PYTHON" - "$ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
runs = pd.read_csv(root / "research/results/legacy/reconstruction/reconstruction_runs.csv")
summary = pd.read_csv(root / "research/results/legacy/reconstruction/reconstruction_summary.csv")
assert len(runs) == 3 and set(runs.seed) == {42, 43, 44}
assert runs.representation.eq("rtp").all() and runs.bottleneck_dim.eq(128).all()
assert len(summary) == 1 and summary.loc[0, "representation"] == "rtp"
assert int(summary.loc[0, "seeds"]) == 3

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

artifacts = {}
for path in sorted((root / "research/results/reconstruction").glob("*")):
    if path.is_file():
        artifacts[str(path)] = sha256(path)
for path in sorted((root / "reconstruction").glob("*/metrics.json")):
    artifacts[str(path)] = sha256(path)
report = {
    "status": "accepted",
    "interface": "native_rtp_128d",
    "seeds": [42, 43, 44],
    "pgen_reference": {
        "note": "not rerun; previously completed native RTP-128 external MLP",
        "rmse_mean": 0.2375036700,
        "rmse_std": 0.0007641427,
    },
    "reconstruction": summary.iloc[0].to_dict(),
    "artifacts": artifacts,
}
(root / "RESULTS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY

sha256sum "$ROOT/RESULTS.json" "$ROOT/research/results/reconstruction"/* > "$ROOT/RESULT_HASHES.sha256"
if grep -R -E -i 'traceback|out of memory|nan|inf' "$ROOT/logs"/*.err; then
  echo "error/nonfinite signature found" >&2
  exit 1
fi
echo status=accepted
