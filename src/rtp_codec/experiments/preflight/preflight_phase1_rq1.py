"""Validate the locked TRB benchmark and Phase-1 runtime on a Slurm CPU node."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from rtp_codec.data.multitask import (
    TargetStandardizer,
    load_prepared_benchmark,
    select_split_indices,
)
from rtp_codec.training.objectives import RTPCodecMultiTaskLoss
from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer
from rtp_codec.tokenization.character import AA_VOCAB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    ready = json.loads((data_dir / "READY.json").read_text(encoding="utf-8"))
    if ready.get("status") != "ready" or not all(ready.get("checks", {}).values()):
        raise ValueError("Locked benchmark READY checks are not all true.")

    table, embeddings = load_prepared_benchmark(data_dir)
    split_indices = {
        split: select_split_indices(table, data_dir, split)
        for split in ("train", "val", "test")
    }
    standardizer = TargetStandardizer.load(data_dir / "target_standardizer.npz")
    if standardizer.pgen_target != "log10_pgen_1mm":
        raise ValueError("Unexpected locked pgen target.")
    if standardizer.train_rows != len(split_indices["train"]):
        raise ValueError("Locked standardizer was not fit on the full train split.")
    if len(standardizer.tcremp_mean) != embeddings.shape[1]:
        raise ValueError("Locked TCRemP standardizer dimension mismatch.")

    torch.manual_seed(42)
    model_config = RTPCodecConfig(
        input_vocab_size=len(AA_VOCAB),
        output_vocab_size=len(AA_VOCAB),
        max_sequence_len=8,
        d_model=32,
        latent_dim=24,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        ff_dim=64,
        dropout=0.0,
        tcremp_dim=12,
        tcremp_head_dim=24,
        pgen_head_dim=12,
        decoder_memory_tokens=2,
    )
    model = RTPCodecTransformer(model_config)
    tokens = torch.randint(5, len(AA_VOCAB), (2, 6))
    decoder_input = torch.randint(5, len(AA_VOCAB), (2, 7))
    outputs = model(tokens, tokens.ne(0), decoder_input)
    losses = RTPCodecMultiTaskLoss()(
        outputs,
        tcremp_target=torch.randn(2, 12),
        pgen_target=torch.randn(2),
        reconstruction_target=torch.randint(5, len(AA_VOCAB), (2, 7)),
        tcremp_mean=torch.zeros(12),
        tcremp_std=torch.ones(12),
        pgen_mean=torch.tensor(0.0),
        pgen_std=torch.tensor(1.0),
    )
    losses["loss"].backward()
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError("Runtime smoke-test loss is non-finite.")

    artifact_paths = [
        data_dir / "READY.json",
        data_dir / "dataset.parquet",
        data_dir / "embeddings.npy",
        data_dir / "target_standardizer.npz",
        *(data_dir / "manifests" / f"{name}.tsv" for name in ("train", "val", "test")),
    ]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    report = {
        "status": "ready",
        "git_revision": revision,
        "benchmark_path": str(data_dir),
        "benchmark_ready_summary": ready,
        "prepared_artifact_sha256": {
            str(path.relative_to(data_dir)): sha256(path) for path in artifact_paths
        },
        "dataset_rows": len(table),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "split_rows": {name: len(values) for name, values in split_indices.items()},
        "standardizer": {
            "pgen_target": standardizer.pgen_target,
            "train_rows": standardizer.train_rows,
            "tcremp_dim": len(standardizer.tcremp_mean),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "smoke_loss_finite": True,
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output_path)


if __name__ == "__main__":
    main()
