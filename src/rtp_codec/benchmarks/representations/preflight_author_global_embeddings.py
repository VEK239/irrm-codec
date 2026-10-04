"""Fail-closed CPU preflight for author-native global embedding benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rtp_codec.benchmarks.representations.cache_external_model_embeddings import encode_esm2, encode_sceptr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-dataset", type=Path, required=True)
    parser.add_argument("--redcea-cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--locked-sha256", required=True)
    parser.add_argument("--redcea-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    actual_hashes = {
        "locked_dataset": sha256_file(args.locked_dataset),
        "redcea_cohort": sha256_file(args.redcea_cohort),
    }
    expected_hashes = {
        "locked_dataset": args.locked_sha256,
        "redcea_cohort": args.redcea_sha256,
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            f"Input hash mismatch: expected={expected_hashes}, actual={actual_hashes}"
        )

    locked = read_table(args.locked_dataset)
    redcea = read_table(args.redcea_cohort)
    for name, frame in (("locked", locked), ("redcea", redcea)):
        if "junction_aa" not in frame and "cdr3" not in frame:
            raise ValueError(f"{name} input lacks junction_aa/cdr3")
        seq_col = "junction_aa" if "junction_aa" in frame else "cdr3"
        if frame[seq_col].astype(str).duplicated().any() and name == "redcea":
            raise ValueError("REDCEA cohort must contain unique clonotypes")

    locked_sequences = locked["junction_aa"].astype(str).head(2).tolist()
    redcea_seq_col = "junction_aa" if "junction_aa" in redcea else "cdr3"
    redcea_sequences = redcea[redcea_seq_col].astype(str).head(2).tolist()

    class SmokeArgs:
        device = "cpu"
        batch_size = 2

    log = logging.getLogger("author_global_preflight")
    esm35 = encode_esm2(locked_sequences, SmokeArgs(), log, model_name="esm2_35m")
    if esm35.shape != (2, 480) or not np.isfinite(esm35).all():
        raise RuntimeError(f"ESM2-35M smoke failed: shape={esm35.shape}")

    sceptr_cdr3 = encode_sceptr(
        locked_sequences,
        SmokeArgs(),
        log,
        frame=locked.head(2).copy(),
        cdr3_only=True,
    )
    if sceptr_cdr3.shape != (2, 64) or not np.isfinite(sceptr_cdr3).all():
        raise RuntimeError(f"SCEPTR-CDR3 smoke failed: shape={sceptr_cdr3.shape}")

    if not {"v_call", "j_call"}.issubset(redcea.columns):
        raise ValueError("REDCEA cohort lacks v_call/j_call for SCEPTR-default")
    sceptr_default = encode_sceptr(
        redcea_sequences,
        SmokeArgs(),
        log,
        frame=redcea.head(2).rename(columns={"cdr3": "junction_aa"}).copy(),
        cdr3_only=False,
    )
    if sceptr_default.shape != (2, 64) or not np.isfinite(sceptr_default).all():
        raise RuntimeError(f"SCEPTR-default smoke failed: shape={sceptr_default.shape}")

    report = {
        "status": "accepted",
        "input_hashes": actual_hashes,
        "row_counts": {"locked": len(locked), "redcea": len(redcea)},
        "representations": {
            "esm2_35m": {
                "model": "esm2_t12_35M_UR50D",
                "pooling": "final-layer residue mean excluding special/pad tokens",
                "modality": "sequence-only",
                "smoke_shape": list(esm35.shape),
            },
            "sceptr_cdr3": {
                "model": "SCEPTR variant.cdr3_only()",
                "pooling": "native contextualized CLS64",
                "modality": "sequence-only",
                "smoke_shape": list(sceptr_cdr3.shape),
            },
            "sceptr": {
                "model": "SCEPTR default",
                "pooling": "native contextualized CLS64",
                "modality": "annotation-aware TRBV+CDR3B+TRBJ",
                "smoke_shape": list(sceptr_default.shape),
            },
        },
    }
    output = args.output_dir / "PREFLIGHT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))



if __name__ == "__main__":
    main()
