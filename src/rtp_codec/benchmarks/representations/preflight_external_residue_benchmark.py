"""Fail-closed gate for the unpooled frozen-residue benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtp_codec.benchmarks.representations.external_residue_models import (
    FrozenResidueDecoder,
    FrozenResiduePgenHead,
    parameter_count,
)
from rtp_codec.training.losses import inverse_loss, pgen_loss
from rtp_codec.tokenization.character import encode


REPRESENTATIONS = ("esm2_8m", "tcr_bert", "sceptr")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--smoke-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(cohort, split):
    path = cohort / "manifests" / f"{split}.tsv"
    rows = pd.read_csv(path, sep="\t")["row_index"].to_numpy(dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"Duplicate row identity in {split}")
    return rows, sha256_file(path)


def main():
    args = parse_args()
    cohort, smoke = Path(args.cohort_dir), Path(args.smoke_dir)
    cohort_report = json.loads((cohort / "COHORT.json").read_text(encoding="utf-8"))
    if cohort_report.get("status") != "accepted" or cohort_report.get("split_overlap") != 0:
        raise ValueError("Locked cohort is not accepted/non-overlapping")
    frame = pd.read_parquet(cohort / "dataset.parquet")
    expected_counts = {"train": 79544, "val": 9943, "test": 9943}
    split_rows, manifest_hashes = {}, {}
    for split, expected in expected_counts.items():
        split_rows[split], manifest_hashes[split] = load_rows(cohort, split)
        if len(split_rows[split]) != expected:
            raise ValueError(f"{split} count changed")
    if len(set(split_rows["train"]) & set(split_rows["val"])):
        raise ValueError("train/val overlap")
    if len(set(split_rows["train"]) & set(split_rows["test"])):
        raise ValueError("train/test overlap")
    if len(set(split_rows["val"]) & set(split_rows["test"])):
        raise ValueError("val/test overlap")

    catalog = json.loads((smoke / "residue_representations.json").read_text(encoding="utf-8"))
    targets = torch.tensor(
        [encode(sequence, max_len=40) for sequence in frame.junction_aa.iloc[:2]], dtype=torch.long
    )
    pgen = torch.tensor(frame.log10_pgen_1mm.iloc[:2].to_numpy(), dtype=torch.float32)
    checks = {}
    for name in REPRESENTATIONS:
        entry = catalog[name]
        features = np.load(entry["features"])
        mask = np.load(entry["mask"])
        if features.shape[0] < 2 or features.shape[1] != 40:
            raise ValueError(f"{name} smoke shape invalid: {features.shape}")
        lengths = frame.junction_aa.iloc[: len(features)].str.len().to_numpy()
        if not np.array_equal(mask.sum(axis=1), lengths):
            raise ValueError(f"{name} mask/length mismatch")
        if not np.isfinite(features).all():
            raise ValueError(f"{name} smoke features non-finite")
        x = torch.from_numpy(features[:2].astype(np.float32))
        m = torch.from_numpy(mask[:2])
        decoder = FrozenResidueDecoder(features.shape[-1]).eval()
        regressor = FrozenResiduePgenHead(features.shape[-1]).eval()
        with torch.no_grad():
            logits = decoder(x, m)
            reconstruction_smoke = inverse_loss(logits, targets)
            prediction = regressor(x, m)
            pgen_smoke = pgen_loss(prediction, pgen)
        if not torch.isfinite(reconstruction_smoke) or not torch.isfinite(pgen_smoke):
            raise ValueError(f"{name} smoke loss is non-finite")
        checks[name] = {
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "mask_counts_equal_lengths": True,
            "features_sha256": sha256_file(entry["features"]),
            "mask_sha256": sha256_file(entry["mask"]),
            "decoder_parameters": parameter_count(decoder),
            "pgen_head_parameters": parameter_count(regressor),
            "finite_reconstruction_smoke_loss": float(reconstruction_smoke),
            "finite_pgen_smoke_loss": float(pgen_smoke),
        }
    report = {
        "status": "accepted",
        "protocol": "frozen per-residue states; no global pooling before trainable task head",
        "cohort": {
            "rows": len(frame),
            "dataset_sha256": sha256_file(cohort / "dataset.parquet"),
            "split_counts": expected_counts,
            "manifest_sha256": manifest_hashes,
            "overlap": 0,
        },
        "representations": checks,
        "controls": {
            "encoder_frozen": True,
            "decoder_trained_separately_per_representation": True,
            "pgen_head_trained_separately_per_representation": True,
            "checkpoint_selection": "validation loss only",
            "test_used_for_selection": False,
            "pgen_normalization": "train-only",
            "important_scope": "best-available frozen interface, not a matched compact-vector comparison",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
