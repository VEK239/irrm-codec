"""Audit row/mask provenance and direct residue-token retention for frozen encoders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


NAMES = ("esm2_8m", "tcr_bert", "sceptr")
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i for i, aa in enumerate(AA)}


def sequence_sha256(sequences) -> str:
    return hashlib.sha256("\n".join(sequences).encode("ascii")).hexdigest()


def length_matched_permutation(lengths: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.arange(len(lengths))
    for length in np.unique(lengths):
        positions = np.flatnonzero(lengths == length)
        if len(positions) > 1:
            order = rng.permutation(positions)
            result[order] = np.roll(order, 1)
    return result


def labels_for_rows(sequences: np.ndarray, rows: np.ndarray, max_len: int = 40):
    labels = np.full((len(rows), max_len), -1, dtype=np.int16)
    mask = np.zeros((len(rows), max_len), dtype=bool)
    for out_row, source_row in enumerate(rows):
        sequence = sequences[source_row]
        encoded = [AA_TO_ID[aa] for aa in sequence]
        labels[out_row, : len(encoded)] = encoded
        mask[out_row, : len(encoded)] = True
    return labels, mask


def centroid_probe(matrix, train_rows, test_rows, train_labels, test_labels, train_mask, test_mask):
    dim = matrix.shape[-1]
    sums = np.zeros((len(AA), dim), dtype=np.float64)
    counts = np.zeros(len(AA), dtype=np.int64)
    for local_start in range(0, len(train_rows), 1024):
        local_stop = min(local_start + 1024, len(train_rows))
        block = np.asarray(matrix[train_rows[local_start:local_stop]], dtype=np.float32)
        labels = train_labels[local_start:local_stop]
        mask = train_mask[local_start:local_stop]
        for aa_id in range(len(AA)):
            selected = block[labels == aa_id]
            if len(selected):
                sums[aa_id] += selected.sum(axis=0, dtype=np.float64)
                counts[aa_id] += len(selected)
    if (counts == 0).any():
        raise ValueError("Missing amino acid in training-centroid audit")
    centroids = (sums / counts[:, None]).astype(np.float32)
    centroid_norm = np.square(centroids).sum(axis=1)
    correct = total = 0
    predictions_by_row = np.full_like(test_labels, -1)
    for local_start in range(0, len(test_rows), 512):
        local_stop = min(local_start + 512, len(test_rows))
        block = np.asarray(matrix[test_rows[local_start:local_stop]], dtype=np.float32)
        flat = block.reshape(-1, dim)
        distances = (
            np.square(flat).sum(axis=1, keepdims=True)
            - 2 * flat @ centroids.T
            + centroid_norm[None, :]
        )
        predicted = distances.argmin(axis=1).reshape(block.shape[:2])
        predictions_by_row[local_start:local_stop] = predicted
        labels = test_labels[local_start:local_stop]
        mask = test_mask[local_start:local_stop]
        correct += int((predicted[mask] == labels[mask]).sum())
        total += int(mask.sum())
    return correct / total, predictions_by_row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--residue-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir, residue_dir = Path(args.dataset_dir), Path(args.residue_dir)
    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    sequences = frame["junction_aa"].astype(str).to_numpy()
    train_rows = pd.read_csv(dataset_dir / "manifests/train.tsv", sep="\t")["row_index"].to_numpy(np.int64)
    test_rows = pd.read_csv(dataset_dir / "manifests/test.tsv", sep="\t")["row_index"].to_numpy(np.int64)
    train_labels, train_mask = labels_for_rows(sequences, train_rows)
    test_labels, test_mask = labels_for_rows(sequences, test_rows)
    test_lengths = test_mask.sum(axis=1)
    perm = length_matched_permutation(test_lengths, args.seed)
    manifest = json.loads((residue_dir / "residue_representations.json").read_text(encoding="utf-8"))
    expected_sequence_sha = sequence_sha256(sequences)
    report = {
        "status": "accepted",
        "sequence_sha256": expected_sequence_sha,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "models": {},
        "interpretation": "Centroid accuracy measures direct amino-acid identity retained by raw residue states; it is not an epitope or biological endpoint.",
    }
    for name in NAMES:
        entry = manifest[name]
        if entry["sequence_sha256"] != expected_sequence_sha or entry["pooling"] != "none":
            raise ValueError(f"Sequence hash/pooling provenance mismatch for {name}")
        matrix = np.load(entry["features"], mmap_mode="r")
        mask = np.load(entry["mask"], mmap_mode="r")
        expected_shape = tuple(entry["shape"])
        if tuple(matrix.shape) != expected_shape or mask.shape != matrix.shape[:2]:
            raise ValueError(f"Shape mismatch for {name}")
        for start in range(0, len(matrix), 4096):
            if not np.isfinite(matrix[start:start + 4096]).all():
                raise ValueError(f"Non-finite residue states for {name}")
        expected_mask = np.zeros_like(mask, dtype=bool)
        for row, sequence in enumerate(sequences):
            expected_mask[row, : len(sequence)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError(f"Residue mask/sequence alignment mismatch for {name}")
        accuracy, predicted = centroid_probe(
            matrix, train_rows, test_rows, train_labels, test_labels, train_mask, test_mask
        )
        permuted_labels = test_labels[perm]
        permuted_mask = test_mask[perm]
        permuted_accuracy = float(
            (predicted[permuted_mask] == permuted_labels[permuted_mask]).mean()
        )
        report["models"][name] = {
            "model": entry["model"],
            "shape": entry["shape"],
            "features_sha256": entry["features_sha256"],
            "mask_sha256": entry["mask_sha256"],
            "mask_matches_sequence_lengths": True,
            "train_centroid_test_residue_accuracy": accuracy,
            "length_matched_permuted_row_accuracy": permuted_accuracy,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
