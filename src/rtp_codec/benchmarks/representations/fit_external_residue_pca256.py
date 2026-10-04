"""Fit a train-only PCA from flattened residue states to one compact clonotype vector.

The public encoder states are already frozen.  We flatten the zero-padded [40,D]
tensor in residue order, standardize each flattened coordinate using the locked
training split only, fit a seeded randomized PCA, and apply that fixed projection to
all rows.  Validation and test rows never participate in fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPRESENTATIONS = ("esm2_8m", "tcr_bert", "sceptr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def fit_randomized_pca(matrix, train_rows, n_components, oversample, niter, seed, device):
    """Return train-only scaler, components and explained-variance diagnostics."""
    train = np.asarray(matrix[train_rows], dtype=np.float32).reshape(len(train_rows), -1)
    if not np.isfinite(train).all():
        raise ValueError("Non-finite training residue states.")
    x = torch.from_numpy(train).to(device)
    del train
    mean = x.mean(dim=0)
    scale = x.std(dim=0, correction=0).clamp_min(1e-6)
    x.sub_(mean).div_(scale)
    q = min(n_components + oversample, x.shape[0], x.shape[1])
    if q < n_components:
        raise ValueError(f"Cannot fit {n_components} components to shape {tuple(x.shape)}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    _, singular_values, vectors = torch.pca_lowrank(
        x, q=q, center=False, niter=niter
    )
    components = vectors[:, :n_components].contiguous()
    kept = singular_values[:n_components].square().sum()
    total = x.square().sum()
    explained = float((kept / total).item()) if total.item() > 0 else 0.0
    return (
        mean.cpu().numpy().astype(np.float32),
        scale.cpu().numpy().astype(np.float32),
        components.cpu().numpy().astype(np.float32),
        explained,
    )


def transform_to_memmap(matrix, mean, scale, components, output, batch_size, device):
    projected = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(matrix.shape[0], components.shape[1])
    )
    mean_t = torch.from_numpy(mean).to(device)
    scale_t = torch.from_numpy(scale).to(device)
    components_t = torch.from_numpy(components).to(device)
    for start in range(0, matrix.shape[0], batch_size):
        stop = min(start + batch_size, matrix.shape[0])
        block = np.asarray(matrix[start:stop], dtype=np.float32).reshape(stop - start, -1)
        x = torch.from_numpy(block).to(device)
        x.sub_(mean_t).div_(scale_t)
        projected[start:stop] = (x @ components_t).cpu().numpy()
    projected.flush()
    return projected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--residue-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--components", type=int, default=256)
    parser.add_argument("--oversample", type=int, default=32)
    parser.add_argument("--power-iterations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Residue PCA fitting must run in a GPU Slurm allocation.")
    device = torch.device("cuda")
    dataset_dir, residue_dir, output_dir = map(
        Path, (args.dataset_dir, args.residue_dir, args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    train_manifest = dataset_dir / "manifests" / "train.tsv"
    train_rows = pd.read_csv(train_manifest, sep="\t")["row_index"].to_numpy(np.int64)
    source = residue_dir / f"{args.representation}_residue.npy"
    source_mask = residue_dir / f"{args.representation}_residue_mask.npy"
    matrix = np.load(source, mmap_mode="r")
    mask = np.load(source_mask, mmap_mode="r")
    if matrix.ndim != 3 or matrix.shape[:2] != mask.shape or matrix.shape[0] != len(frame):
        raise ValueError(f"Unexpected residue/mask shapes: {matrix.shape}, {mask.shape}")
    started = time.perf_counter()
    mean, scale, components, explained = fit_randomized_pca(
        matrix,
        train_rows,
        args.components,
        args.oversample,
        args.power_iterations,
        args.seed,
        device,
    )
    output = output_dir / f"{args.representation}_residue_{args.components}.npy"
    projected = transform_to_memmap(
        matrix, mean, scale, components, output, args.batch_size, device
    )
    for start in range(0, len(projected), args.batch_size):
        if not np.isfinite(projected[start : start + args.batch_size]).all():
            raise ValueError("Non-finite projected values.")
    projection_path = output_dir / f"{args.representation}_residue_projection.npz"
    np.savez(
        projection_path,
        mean=mean,
        scale=scale,
        components=components,
        fit_rows=train_rows,
    )
    report = {
        "status": "accepted",
        "representation": f"{args.representation}_residue",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_shape": list(matrix.shape),
        "flattened_dim": int(matrix.shape[1] * matrix.shape[2]),
        "output": str(output),
        "output_sha256": sha256(output),
        "output_shape": list(projected.shape),
        "projection": str(projection_path),
        "projection_sha256": sha256(projection_path),
        "components": args.components,
        "explained_variance_ratio": explained,
        "fit_split": "train",
        "fit_rows": int(len(train_rows)),
        "fit_rows_sha256": rows_sha256(train_rows),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": sha256(train_manifest),
        "validation_or_test_used_for_fit": False,
        "standardization": "per_flattened_coordinate_train_only",
        "solver": "torch_randomized_pca_lowrank",
        "oversample": args.oversample,
        "power_iterations": args.power_iterations,
        "seed": args.seed,
        "finite": True,
        "seconds": round(time.perf_counter() - started, 2),
    }
    report_path = output_dir / f"{args.representation}_residue_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
