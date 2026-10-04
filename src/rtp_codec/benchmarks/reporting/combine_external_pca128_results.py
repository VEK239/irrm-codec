"""Combine the all-model PCA-128 sensitivity tables without recomputing metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--sceptr-root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    main_root, sceptr_root = Path(args.main_root), Path(args.sceptr_root)
    output = main_root / "results" / "all_baselines"
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for task, filename in (("reconstruction", "reconstruction_summary.csv"), ("pgen", "pgen_summary.csv")):
        paths = [
            main_root / "results" / task / filename,
            sceptr_root / "results" / task / filename,
        ]
        frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        identity_column = "representation" if task == "reconstruction" else "arm"
        if len(frame) != 3 or frame[identity_column].nunique() != 3:
            raise ValueError(f"Expected three distinct PCA-128 baseline rows for {task}")
        destination = output / filename
        frame.to_csv(destination, index=False)
        sources[task] = {
            "inputs": [{"path": str(path), "sha256": sha256(path)} for path in paths],
            "output": str(destination),
            "output_sha256": sha256(destination),
        }
    (output / "MANIFEST.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print("status=accepted baselines=3 tasks=2")


if __name__ == "__main__":
    main()
