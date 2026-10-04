"""Issue 2: reduce every representation to a common bottleneck dimension with PCA.

The scaler and the PCA are fitted on the training split only and then applied to
validation and test, so no information from the held-out rows leaks into the
projection. Representations arrive on very different scales (TCRemP holds alignment
distances in the hundreds, transformer activations sit near zero), so each one is
standardized before the PCA.

One PCA is fitted at the largest requested width. Principal components are ordered by
explained variance, so the narrower bottlenecks are prefixes of the same fit and cost
nothing extra.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

from irrm_codec.utils import setup_logging

# Above this input width the full SVD needs more memory than it is worth, so the
# incremental solver streams the training split in batches instead.
INCREMENTAL_PCA_MIN_DIM = 2000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="data/benchmark/trb")
    p.add_argument("--representations-dir", default="data/benchmark/trb/representations")
    p.add_argument("--output-dir", default="data/benchmark/trb/bottleneck")
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--representations", nargs="+")
    p.add_argument("--batch-size", type=int, default=4096, help="IncrementalPCA batch size.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def fit_projection(matrix, train_rows, n_components, batch_size, seed, log):
    """Standardize and fit a PCA on the training rows only."""
    scaler = StandardScaler()
    incremental = matrix.shape[1] >= INCREMENTAL_PCA_MIN_DIM

    if incremental:
        # Two streamed passes: one to learn the scaling, one to fit the components.
        for start in range(0, len(train_rows), batch_size):
            scaler.partial_fit(np.asarray(matrix[train_rows[start : start + batch_size]]))
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        for start in range(0, len(train_rows), batch_size):
            block = np.asarray(matrix[train_rows[start : start + batch_size]], dtype=np.float64)
            if len(block) < n_components:
                break  # A trailing batch smaller than the component count cannot be fitted.
            pca.partial_fit(scaler.transform(block))
    else:
        train = np.asarray(matrix[train_rows], dtype=np.float64)
        scaler.fit(train)
        pca = PCA(n_components=n_components, random_state=seed)
        pca.fit(scaler.transform(train))

    log.info(
        "fitted %s pca n_components=%d",
        "incremental" if incremental else "full",
        n_components,
    )
    return scaler, pca


def transform_all(matrix, scaler, pca, batch_size):
    """Project every row, streaming so the full float64 copy never materializes."""
    out = np.empty((matrix.shape[0], pca.n_components_), dtype=np.float32)
    for start in range(0, matrix.shape[0], batch_size):
        block = np.asarray(matrix[start : start + batch_size], dtype=np.float64)
        out[start : start + batch_size] = pca.transform(scaler.transform(block)).astype(np.float32)
    return out


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    reps_dir = Path(args.representations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "fit_bottleneck.log")

    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    train_rows = pd.read_csv(dataset_dir / "manifests" / "train.tsv", sep="\t")["row_index"].to_numpy()
    log.info("dataset rows=%d train rows=%d", len(frame), len(train_rows))

    catalog = json.loads((reps_dir / "representations.json").read_text(encoding="utf-8"))
    names = args.representations or list(catalog)
    max_dim = max(args.dims)

    summary_path = output_dir / "bottleneck_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    for name in names:
        source = Path(catalog[name]["path"])
        matrix = np.load(source, mmap_mode="r")
        if matrix.shape[0] != len(frame):
            raise ValueError(f"{name} has {matrix.shape[0]} rows, expected {len(frame)}.")

        # A representation narrower than the bottleneck cannot be reduced to it; SCEPTR
        # is already 64-dimensional, so its widest projection is capped at its own width.
        n_components = min(max_dim, matrix.shape[1], len(train_rows))
        if name in summary and not args.overwrite:
            log.info("%s already projected, skipping", name)
            continue

        log.info("fitting %s dim=%d -> %d", name, matrix.shape[1], n_components)
        started = time.perf_counter()
        scaler, pca = fit_projection(
            matrix, train_rows, n_components, args.batch_size, args.seed, log
        )
        projected = transform_all(matrix, scaler, pca, args.batch_size)
        elapsed = time.perf_counter() - started

        dump({"scaler": scaler, "pca": pca}, output_dir / f"{name}_projection.joblib")

        ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        explained = {}
        for dim in sorted(args.dims):
            if dim > n_components:
                explained[str(dim)] = None
                continue
            np.save(output_dir / f"{name}_{dim}.npy", projected[:, :dim])
            explained[str(dim)] = round(float(ratios[:dim].sum()), 6)

        summary[name] = {
            "source_dim": int(matrix.shape[1]),
            "components_fitted": int(n_components),
            "explained_variance": explained,
            "seconds": round(elapsed, 2),
            "solver": "incremental" if matrix.shape[1] >= INCREMENTAL_PCA_MIN_DIM else "full",
            "capped_by_source_dim": bool(n_components < max_dim),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("%s done in %.1f s explained=%s", name, elapsed, explained)

    log.info("=" * 68)
    header = "  ".join(f"{d:>8d}" for d in sorted(args.dims))
    log.info("%-10s %6s  %s", "repr", "dim", header)
    for name, info in summary.items():
        cells = "  ".join(
            f"{info['explained_variance'][str(d)]:8.4f}"
            if info["explained_variance"].get(str(d)) is not None
            else "       -"
            for d in sorted(args.dims)
        )
        log.info("%-10s %6d  %s", name, info["source_dim"], cells)
    log.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
