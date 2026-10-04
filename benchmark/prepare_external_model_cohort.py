"""Build the locked, common-row cohort used by every external representation.

The source benchmark and its split labels remain immutable.  Rows with V/J calls that
SCEPTR cannot standardize as functional human genes are removed *before* model scoring;
the same filtered rows are then used for every representation.  Original row identity is
retained in ``source_row_index`` and all emitted matrices are checked by that mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_rows(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t")["row_index"].to_numpy(dtype=np.int64)


def deterministic_rank(sequence: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{sequence}".encode("utf-8")).hexdigest()


def nested_training_subsets(
    frame: pd.DataFrame, train_rows: np.ndarray, sizes=(1_000, 10_000), seed=42
) -> dict[str, np.ndarray]:
    ranked = sorted(
        train_rows.tolist(),
        key=lambda row: (deterministic_rank(str(frame.at[row, "junction_aa"]), seed), row),
    )
    subsets: dict[str, np.ndarray] = {}
    previous: list[int] = []
    for size in sizes:
        if len(ranked) < size:
            raise ValueError(f"only {len(ranked)} filtered train rows, cannot emit {size}")
        previous_set = set(previous)
        chosen = previous + [row for row in ranked if row not in previous_set][: size - len(previous)]
        subsets[str(size // 1000) + "k"] = np.asarray(chosen, dtype=np.int64)
        previous = chosen
    subsets["all"] = np.asarray(ranked, dtype=np.int64)
    return subsets


def subset_matrix(source: Path, source_rows: np.ndarray, output: Path) -> dict[str, object]:
    matrix = np.load(source, mmap_mode="r")
    if np.any(source_rows < 0) or np.any(source_rows >= matrix.shape[0]):
        raise ValueError(f"{source}: row mapping exceeds matrix shape {matrix.shape}")
    target = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(len(source_rows), matrix.shape[1])
    )
    for start in range(0, len(source_rows), 4096):
        block = np.asarray(matrix[source_rows[start : start + 4096]], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"{source}: non-finite block at filtered row {start}")
        target[start : start + len(block)] = block
    target.flush()
    del target
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "shape": [int(len(source_rows)), int(matrix.shape[1])],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--rtp-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import tidytcells as tt

    source_dataset = args.source_dataset_dir / "dataset.parquet"
    source_frame = pd.read_parquet(source_dataset).reset_index(drop=True)
    standardized: dict[str, pd.Series] = {}
    usable = np.ones(len(source_frame), dtype=bool)
    dropped: dict[str, int] = {}
    for column, gene_type in (("v_call", "v"), ("j_call", "j")):
        values = source_frame[column].map(
            lambda value: tt.tr.standardize(
                value, species="homosapiens", enforce_functional=True, log_failures=False
            )
        )
        standardized[column] = values
        invalid = values.isna().to_numpy()
        dropped[f"nonfunctional_or_invalid_{gene_type}"] = int(invalid.sum())
        usable &= ~invalid

    source_rows = np.flatnonzero(usable).astype(np.int64)
    output = args.output_dir
    manifests = output / "manifests"
    reps = output / "representations"
    manifests.mkdir(parents=True, exist_ok=True)
    reps.mkdir(parents=True, exist_ok=True)

    filtered = source_frame.iloc[source_rows].copy()
    filtered["source_row_index"] = source_rows
    filtered["v_call"] = standardized["v_call"].iloc[source_rows].to_numpy()
    filtered["j_call"] = standardized["j_call"].iloc[source_rows].to_numpy()
    filtered["row_index"] = np.arange(len(filtered), dtype=np.int64)
    filtered.to_parquet(output / "dataset.parquet", index=False)

    source_to_new = {int(source): int(new) for new, source in enumerate(source_rows)}
    split_new: dict[str, np.ndarray] = {}
    source_manifest_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = args.source_dataset_dir / "manifests" / f"{split}.tsv"
        original = manifest_rows(path)
        kept = [source_to_new[int(row)] for row in original if int(row) in source_to_new]
        split_new[split] = np.asarray(kept, dtype=np.int64)
        filtered.iloc[split_new[split]].to_csv(manifests / f"{split}.tsv", sep="\t", index=False)
        source_manifest_hashes[split] = sha256_file(path)

    sets = {name: set(rows.tolist()) for name, rows in split_new.items()}
    if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("filtered split overlap")
    if len(set.union(*sets.values())) != len(filtered):
        raise ValueError("filtered splits do not cover the common cohort")

    subsets = nested_training_subsets(filtered, split_new["train"], seed=args.seed)
    for name, rows in subsets.items():
        filtered.iloc[rows].to_csv(manifests / f"train_{name}.tsv", sep="\t", index=False)

    matrix_reports = {
        "tcremp": subset_matrix(
            args.source_dataset_dir / "embeddings.npy", source_rows, reps / "tcremp.npy"
        ),
        "rtp": subset_matrix(args.rtp_embeddings, source_rows, reps / "rtp.npy"),
    }
    catalog = {
        name: {
            "path": report["output"],
            "rows": report["shape"][0],
            "dim": report["shape"][1],
            "sha256": report["output_sha256"],
            "source": report["source"],
            "source_sha256": report["source_sha256"],
        }
        for name, report in matrix_reports.items()
    }
    (reps / "representations.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8"
    )

    report = {
        "status": "accepted",
        "definition": "common functional-human-V/J intersection fixed before model evaluation",
        "source_dataset": str(source_dataset),
        "source_dataset_sha256": sha256_file(source_dataset),
        "source_rows": int(len(source_frame)),
        "common_rows": int(len(filtered)),
        "dropped_total": int(len(source_frame) - len(filtered)),
        "dropped": dropped,
        "split_counts": {name: int(len(rows)) for name, rows in split_new.items()},
        "split_overlap": 0,
        "source_manifest_sha256": source_manifest_hashes,
        "sequence_sha256": hashlib.sha256(
            "\n".join(filtered["junction_aa"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "nested_train_counts": {name: int(len(rows)) for name, rows in subsets.items()},
        "matrices": matrix_reports,
        "sceptr_modality": "sequence plus V/J annotations; reported separately from sequence-only models",
    }
    (output / "COHORT.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
