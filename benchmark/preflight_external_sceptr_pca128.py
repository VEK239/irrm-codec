"""Standalone fail-closed smoke gate for the SCEPTR residue-PCA 128D study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark.fit_external_residue_pca256 import fit_randomized_pca
from benchmark.train_external_model_pgen import FROZEN_ARMS, RegressionHead
from irrm_codec.inverse_model import InverseModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--residue-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    residue_dir = Path(args.residue_dir)
    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    splits = {
        name: pd.read_csv(dataset_dir / "manifests" / f"{name}.tsv", sep="\t")[
            "row_index"
        ].to_numpy(np.int64)
        for name in ("train", "val", "test")
    }
    if tuple(map(len, splits.values())) != (79544, 9943, 9943):
        raise ValueError("Locked split counts changed")
    split_sets = {name: set(rows.tolist()) for name, rows in splits.items()}
    if any(
        split_sets[a] & split_sets[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Split row overlap")
    source = residue_dir / "sceptr_residue.npy"
    matrix = np.load(source, mmap_mode="r")
    if tuple(matrix.shape) != (len(frame), 40, 64):
        raise ValueError(f"Unexpected SCEPTR residue shape {matrix.shape}")
    for start in range(0, len(matrix), 4096):
        if not np.isfinite(matrix[start : start + 4096]).all():
            raise ValueError("Non-finite SCEPTR residue states")

    rng = np.random.default_rng(7)
    toy = rng.normal(size=(30, 4, 5)).astype(np.float32)
    mean, scale, components, explained = fit_randomized_pca(
        toy,
        np.arange(20),
        n_components=6,
        oversample=2,
        niter=2,
        seed=42,
        device=torch.device("cpu"),
    )
    scores = ((toy.reshape(30, -1) - mean) / scale) @ components
    if scores.shape != (30, 6) or not np.isfinite(scores).all() or not 0 < explained <= 1:
        raise ValueError("Randomized PCA smoke failed")
    if FROZEN_ARMS.get("sceptr_residue_pca128_mlp") != "sceptr_residue":
        raise ValueError("Pgen arm mapping is missing or incorrect")
    with torch.no_grad():
        features = torch.randn(4, 128)
        prediction = RegressionHead(128)(features)
        logits = InverseModel(embedding_dim=128, max_len=40)(features)
    if prediction.shape != (4,) or logits.shape != (4, 40, 25):
        raise ValueError("Head shape smoke failed")
    if not torch.isfinite(prediction).all() or not torch.isfinite(logits).all():
        raise ValueError("Non-finite head smoke")

    report = {
        "status": "accepted",
        "rows": len(frame),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_overlap": 0,
        "source_shape": list(matrix.shape),
        "source_sha256": sha256(source),
        "target_dim": 128,
        "pgen_arm": "sceptr_residue_pca128_mlp",
        "finite_smoke": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
