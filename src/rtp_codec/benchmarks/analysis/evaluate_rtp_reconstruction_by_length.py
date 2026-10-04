#!/usr/bin/env python
"""Quantify frozen RTP/DATA-ANCHOR reconstruction errors by target CDR3 length."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rtp_codec.data.multitask import MultiTaskBenchmarkDataset, build_multitask_dataloader, load_prepared_benchmark, resolve_encoder_tokenizer, select_split_indices
from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer
from rtp_codec.tokenization.character import decode


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in (args.data_dir / "READY.json", args.tokenizer, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    ready = json.loads((args.data_dir / "READY.json").read_text(encoding="utf-8"))
    if ready.get("status") != "ready" or not all(ready.get("checks", {}).values()):
        raise ValueError("Benchmark READY.json is not accepting")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = RTPCodecConfig(**payload["model_config"])
    if payload["tokenizer"]["type"] != "data_anchor":
        raise ValueError("This evaluation is registered for DATA-ANCHOR checkpoints only")
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(args.tokenizer))
    if tokenizer.vocab_size != config.input_vocab_size:
        raise ValueError("Tokenizer vocabulary differs from checkpoint input embedding")
    table, embeddings = load_prepared_benchmark(args.data_dir)
    indices = select_split_indices(table, args.data_dir, "test")
    dataset = MultiTaskBenchmarkDataset(table, embeddings, indices, tokenizer, "log10_pgen_1mm", config.max_sequence_len)
    device = torch.device(args.device)
    loader = build_multitask_dataloader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = RTPCodecTransformer(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    counts = defaultdict(lambda: {"sequences": 0, "exact": 0, "length_mismatch": 0, "aligned_positions": 0, "wrong_aligned_positions": 0})
    with torch.inference_mode():
        for batch in loader:
            tokens = batch["encoder_tokens"].to(device, non_blocking=True)
            mask = batch["encoder_mask"].to(device, non_blocking=True)
            predictions = [decode(row.tolist()) for row in model.reconstruct(tokens, mask).cpu().numpy()]
            for target, prediction in zip(batch["sequence"], predictions):
                item = counts[len(target)]
                item["sequences"] += 1
                item["exact"] += int(target == prediction)
                if len(target) != len(prediction):
                    item["length_mismatch"] += 1
                    continue
                item["aligned_positions"] += len(target)
                item["wrong_aligned_positions"] += sum(a != b for a, b in zip(target, prediction))

    rows = []
    for length, item in sorted(counts.items()):
        sequences = item["sequences"]
        rows.append({
            "target_length": length,
            **item,
            "sequence_error_rate": 1 - item["exact"] / sequences,
            "aligned_position_error_rate": item["wrong_aligned_positions"] / item["aligned_positions"] if item["aligned_positions"] else np.nan,
        })
    result = pd.DataFrame(rows)
    if int(result["sequences"].sum()) != len(indices):
        raise RuntimeError("Processed sequence count differs from locked test split")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output_dir / "error_rates_by_target_length.tsv", sep="\t", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(result["target_length"], result["sequence_error_rate"] * 100, marker="o", color="#332288")
    axes[0].set_xlabel("Target CDR3 length (aa)")
    axes[0].set_ylabel("Sequences not reconstructed exactly (%)")
    axes[0].set_title("Sequence-level reconstruction error")
    axes[1].plot(result["target_length"], result["aligned_position_error_rate"] * 100, marker="o", color="#EE6677")
    axes[1].set_xlabel("Target CDR3 length (aa)")
    axes[1].set_ylabel("Wrong aligned positions (%)")
    axes[1].set_title("Position-level reconstruction error")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RTP / DATA-ANCHOR reconstruction error by target CDR3 length", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig((args.output_dir / "reconstruction_error_by_length").with_suffix(f".{suffix}"), dpi=240 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "status": "complete",
        "analysis": "locked_test_autoregressive_error_by_target_cdr3_length",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "tokenizer_sha256": sha256(args.tokenizer),
        "benchmark_ready_sha256": sha256(args.data_dir / "READY.json"),
        "model_config": asdict(config),
        "test_sequences": len(indices),
        "length_range": [int(result["target_length"].min()), int(result["target_length"].max())],
    }
    (args.output_dir / "RESULTS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
