from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtp_codec.benchmarks.representations.preflight_external_model_benchmark import read_manifest_rows, validate_splits
from rtp_codec.benchmarks.datasets.prepare_external_model_cohort import nested_training_subsets


def _write_manifest(root: Path, name: str, rows: list[int]) -> None:
    pd.DataFrame({"row_index": rows}).to_csv(root / f"{name}.tsv", sep="\t", index=False)


def test_validate_splits_requires_exact_nonoverlapping_cover(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    _write_manifest(manifests, "train", [0, 2, 4])
    _write_manifest(manifests, "val", [1])
    _write_manifest(manifests, "test", [3, 5])
    report = validate_splits(tmp_path, 6)
    assert report["counts"] == {"train": 3, "val": 1, "test": 2}
    assert report["overlaps"] == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_validate_splits_rejects_overlap(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    _write_manifest(manifests, "train", [0, 1])
    _write_manifest(manifests, "val", [1, 2])
    _write_manifest(manifests, "test", [3])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(tmp_path, 4)


def test_read_manifest_rejects_duplicate_identity(tmp_path: Path) -> None:
    path = tmp_path / "rows.tsv"
    pd.DataFrame({"row_index": np.array([2, 2])}).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest_rows(path)


def test_nested_training_subsets_are_exact_deterministic_and_nested() -> None:
    frame = pd.DataFrame({"junction_aa": [f"CASS{i:05d}F" for i in range(12_000)]})
    train = np.arange(len(frame), dtype=np.int64)
    first = nested_training_subsets(frame, train, seed=42)
    second = nested_training_subsets(frame, train[::-1], seed=42)
    assert len(first["1k"]) == 1_000
    assert len(first["10k"]) == 10_000
    assert set(first["1k"]) < set(first["10k"])
    assert np.array_equal(first["1k"], second["1k"])
    assert np.array_equal(first["10k"], second["10k"])
