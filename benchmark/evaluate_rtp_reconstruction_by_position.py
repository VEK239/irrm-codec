#!/usr/bin/env python
"""Map autoregressive RTP/DATA-ANCHOR reconstruction errors along CDR3 position."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from irrm_codec.multitask_data import (
    MultiTaskBenchmarkDataset,
    build_multitask_dataloader,
    load_prepared_benchmark,
    resolve_encoder_tokenizer,
    select_split_indices,
)
from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer
from irrm_codec.tokenization import decode


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


def rate_table(counts: dict[int, list[int]], label: str) -> pd.DataFrame:
    rows = []
    for position, (eligible, wrong) in sorted(counts.items()):
        rows.append(
            {
                label: position,
                "eligible_positions": eligible,
                "wrong_positions": wrong,
                "error_rate": wrong / eligible if eligible else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_figure(absolute: pd.DataFrame, edge: pd.DataFrame, regions: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(absolute["position_1based"], absolute["error_rate"] * 100, marker="o", color="#332288")
    axes[0].set_xlabel("Position from N terminus")
    axes[0].set_ylabel("Autoregressive error rate (%)")
    axes[0].set_title("Absolute CDR3 position")

    axes[1].plot(edge["distance_from_nearest_edge"], edge["error_rate"] * 100, marker="o", color="#EE6677")
    axes[1].set_xlabel("Distance from nearest CDR3 edge")
    axes[1].set_ylabel("Autoregressive error rate (%)")
    axes[1].set_title("Distance from either edge")

    axes[2].bar(regions["region"], regions["error_rate"] * 100, color=["#4477AA", "#CCBB44", "#66CCEE"])
    axes[2].set_ylabel("Autoregressive error rate (%)")
    axes[2].set_title("First/last 3 residues versus middle")
    for idx, row in regions.iterrows():
        axes[2].text(idx, row["error_rate"] * 100, f"{row['wrong_positions']}/{row['eligible_positions']}", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RTP / DATA-ANCHOR reconstruction errors along locked-test CDR3 sequences", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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
    config = IRRMCodecConfig(**payload["model_config"])
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
    model = IRRMCodecTransformer(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    absolute: dict[int, list[int]] = {}
    edge: dict[int, list[int]] = {}
    region = {name: [0, 0] for name in ("N_edge_1_3", "middle", "C_edge_1_3")}
    matched_sequences = 0
    mismatched_sequences = 0

    with torch.inference_mode():
        for batch in loader:
            tokens = batch["encoder_tokens"].to(device, non_blocking=True)
            mask = batch["encoder_mask"].to(device, non_blocking=True)
            predictions = [decode(row.tolist()) for row in model.reconstruct(tokens, mask).cpu().numpy()]
            for target, prediction in zip(batch["sequence"], predictions):
                if len(target) != len(prediction):
                    mismatched_sequences += 1
                    continue
                matched_sequences += 1
                for position, (target_aa, prediction_aa) in enumerate(zip(target, prediction)):
                    wrong = int(target_aa != prediction_aa)
                    absolute.setdefault(position + 1, [0, 0])
                    absolute[position + 1][0] += 1
                    absolute[position + 1][1] += wrong
                    distance = min(position, len(target) - position - 1)
                    edge.setdefault(distance, [0, 0])
                    edge[distance][0] += 1
                    edge[distance][1] += wrong
                    region_name = "N_edge_1_3" if position < 3 else "C_edge_1_3" if position >= len(target) - 3 else "middle"
                    region[region_name][0] += 1
                    region[region_name][1] += wrong

    if matched_sequences + mismatched_sequences != len(indices):
        raise RuntimeError("Processed sequence count differs from locked test split")
    absolute_table = rate_table(absolute, "position_1based")
    edge_table = rate_table(edge, "distance_from_nearest_edge")
    region_table = pd.DataFrame(
        [{"region": name, "eligible_positions": values[0], "wrong_positions": values[1], "error_rate": values[1] / values[0]} for name, values in region.items()]
    )
    if int(region_table["wrong_positions"].sum()) != int(absolute_table["wrong_positions"].sum()):
        raise RuntimeError("Position and region error totals differ")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    absolute_table.to_csv(args.output_dir / "absolute_position_error_rates.tsv", sep="\t", index=False)
    edge_table.to_csv(args.output_dir / "edge_distance_error_rates.tsv", sep="\t", index=False)
    region_table.to_csv(args.output_dir / "region_error_rates.tsv", sep="\t", index=False)
    save_figure(absolute_table, edge_table, region_table, args.output_dir / "reconstruction_error_by_position")
    result = {
        "status": "complete",
        "analysis": "locked_test_autoregressive_error_by_cdr3_position",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "tokenizer_sha256": sha256(args.tokenizer),
        "benchmark_ready_sha256": sha256(args.data_dir / "READY.json"),
        "model_config": asdict(config),
        "matched_length_sequences": matched_sequences,
        "mismatched_length_sequences": mismatched_sequences,
        "regions": region_table.to_dict(orient="records"),
        "note": "Position analysis includes only target/prediction pairs with equal sequence length.",
    }
    (args.output_dir / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
