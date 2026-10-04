"""Fail-closed preflight for the locked external-model benchmark.

This script performs metadata/import/model-cache checks on a Slurm CPU node.  It does
not fit a model or produce benchmark scores.  Its JSON output is the gate used before
external representation extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "junction_aa",
    "v_call",
    "j_call",
    "log10_pgen_1mm",
}
SPECIAL_REPRESENTATIONS = ("onehot", "tcremp", "rtp")
EXTERNAL_REPRESENTATIONS = ("esm2_8m", "tcr_bert", "sceptr")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest_rows(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")
    if "row_index" not in frame:
        raise ValueError(f"{path} has no row_index column")
    rows = frame["row_index"].to_numpy(dtype=np.int64)
    if len(rows) != len(np.unique(rows)):
        raise ValueError(f"{path} contains duplicate row_index values")
    return rows


def validate_splits(dataset_dir: Path, n_rows: int) -> dict[str, Any]:
    manifests = dataset_dir / "manifests"
    split_rows = {
        split: read_manifest_rows(manifests / f"{split}.tsv")
        for split in ("train", "val", "test")
    }
    for split, rows in split_rows.items():
        if np.any(rows < 0) or np.any(rows >= n_rows):
            raise ValueError(f"{split} has out-of-range row_index")
    sets = {name: set(rows.tolist()) for name, rows in split_rows.items()}
    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"split overlap detected: {overlaps}")
    if len(set.union(*sets.values())) != n_rows:
        raise ValueError("train/val/test manifests do not cover the dataset exactly")
    return {
        "counts": {name: int(len(rows)) for name, rows in split_rows.items()},
        "overlaps": overlaps,
        "manifest_sha256": {
            name: sha256_file(manifests / f"{name}.tsv") for name in split_rows
        },
    }


def module_report(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    report: dict[str, Any] = {"available": spec is not None}
    if spec is None:
        return report
    try:
        module = importlib.import_module(name)
        report.update(
            {
                "import_ok": True,
                "version": getattr(module, "__version__", None),
                "path": getattr(module, "__file__", None),
            }
        )
    except Exception as exc:  # preflight records the exact import blocker
        report.update({"import_ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return report


def inspect_hf_cache() -> dict[str, Any]:
    cache_root = Path(
        os.environ.get(
            "HF_HOME",
            os.environ.get("HUGGINGFACE_HUB_CACHE", str(Path.home() / ".cache" / "huggingface")),
        )
    )
    hub_root = cache_root if cache_root.name == "hub" else cache_root / "hub"
    candidates = {
        "tcr_bert": hub_root / "models--wukevin--tcr-bert",
    }
    return {
        "root": str(hub_root),
        "models": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "snapshots": sorted(p.name for p in (path / "snapshots").glob("*"))
                if (path / "snapshots").exists()
                else [],
            }
            for name, path in candidates.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--rtp-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=99_430)
    parser.add_argument("--expected-latent-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset_dir / "dataset.parquet"
    ready_path = args.dataset_dir / "READY.json"
    if not dataset_path.exists() or not ready_path.exists():
        raise FileNotFoundError("locked dataset.parquet/READY.json is missing")

    frame = pd.read_parquet(dataset_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    if len(frame) != args.expected_rows:
        raise ValueError(f"dataset has {len(frame)} rows, expected {args.expected_rows}")
    if frame["junction_aa"].duplicated().any():
        raise ValueError("locked dataset contains duplicate CDR3 amino-acid sequences")
    if not np.isfinite(frame["log10_pgen_1mm"].to_numpy(dtype=float)).all():
        raise ValueError("log10_pgen_1mm contains non-finite targets")

    split_report = validate_splits(args.dataset_dir, len(frame))
    rtp = np.load(args.rtp_embeddings, mmap_mode="r")
    if rtp.shape != (args.expected_rows, args.expected_latent_dim):
        raise ValueError(
            f"RTP embedding shape is {rtp.shape}, expected "
            f"({args.expected_rows}, {args.expected_latent_dim})"
        )
    # Streamed finiteness check avoids making an unnecessary 50 MB copy.
    for start in range(0, len(rtp), 8192):
        if not np.isfinite(np.asarray(rtp[start : start + 8192])).all():
            raise ValueError(f"RTP embeddings non-finite at block beginning {start}")

    v = frame["v_call"].fillna("").astype(str).str.strip()
    j = frame["j_call"].fillna("").astype(str).str.strip()
    sceptr_candidate = v.ne("") & j.ne("")
    report = {
        "status": "accepted_metadata_gate",
        "note": "Imports/caches are inventoried; encoder execution is a separate Slurm gate.",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "dataset": {
            "path": str(dataset_path),
            "rows": int(len(frame)),
            "dataset_sha256": sha256_file(dataset_path),
            "ready_sha256": sha256_file(ready_path),
            "unique_cdr3": int(frame["junction_aa"].nunique()),
            "finite_log10_pgen_1mm": True,
            "split": split_report,
        },
        "rtp": {
            "path": str(args.rtp_embeddings),
            "shape": list(rtp.shape),
            "dtype": str(rtp.dtype),
            "finite": True,
            "sha256": sha256_file(args.rtp_embeddings),
        },
        "sceptr_metadata": {
            "candidate_rows_with_v_and_j": int(sceptr_candidate.sum()),
            "missing_v": int(v.eq("").sum()),
            "missing_j": int(j.eq("").sum()),
            "candidate_by_split": {},
            "warning": "SCEPTR uses V/J calls and is annotation-aware, unlike the sequence-only encoders.",
        },
        "modules": {
            name: module_report(name)
            for name in ("torch", "sklearn", "transformers", "esm", "sceptr")
        },
        "hf_cache": inspect_hf_cache(),
        "planned_representations": [*SPECIAL_REPRESENTATIONS, *EXTERNAL_REPRESENTATIONS],
    }
    for split in ("train", "val", "test"):
        rows = read_manifest_rows(args.dataset_dir / "manifests" / f"{split}.tsv")
        report["sceptr_metadata"]["candidate_by_split"][split] = int(sceptr_candidate.iloc[rows].sum())

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
