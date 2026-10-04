#!/usr/bin/env python
"""Evaluate structured reconstruction substitutions for one frozen RTP-CODEC run.

The analysis follows the earlier inverse-model protocol, but runs autoregressive
reconstruction from a specified final multi-task checkpoint on its locked test split.
It counts only wrong amino-acid positions among target/prediction pairs of equal
length and summarizes substitutions within predefined physicochemical groups.
"""

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

from rtp_codec.data.multitask import (
    MultiTaskBenchmarkDataset,
    build_multitask_dataloader,
    load_prepared_benchmark,
    resolve_encoder_tokenizer,
    select_split_indices,
)
from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer
from rtp_codec.tokenization.character import decode


AA_ORDER = list("CSTAGPDEQNHRKMILVWYF")
AA_GROUPS = {
    "C": set("C"),
    "STAGP": set("STAGP"),
    "DEQN": set("DEQN"),
    "HRK": set("HRK"),
    "MILV": set("MILV"),
    "WYF": set("WYF"),
}
AA_TO_GROUP = {aa: name for name, aas in AA_GROUPS.items() for aa in aas}
GROUP_BOUNDARIES = np.cumsum([len(group) for group in AA_GROUPS.values()])[:-1]


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


def save_figure(matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(matrix.to_numpy(), cmap="Greys")
    fig.colorbar(image, ax=ax, label="Wrong aligned positions")
    ax.set_xticks(np.arange(len(AA_ORDER)), AA_ORDER)
    ax.set_yticks(np.arange(len(AA_ORDER)), AA_ORDER)
    for boundary in GROUP_BOUNDARIES:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.5)
        ax.axvline(boundary - 0.5, color="black", linewidth=1.5)
    ax.set_xlabel("Predicted amino acid")
    ax.set_ylabel("Target amino acid")
    ax.set_title("RTP / DATA-ANCHOR: substitution errors on locked TRB test split")
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = RTPCodecConfig(**checkpoint["model_config"])
    tokenizer_info = checkpoint["tokenizer"]
    if tokenizer_info["type"] != "data_anchor":
        raise ValueError(f"Expected a DATA-ANCHOR checkpoint, got {tokenizer_info['type']!r}")
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(args.tokenizer))
    if tokenizer.vocab_size != config.input_vocab_size:
        raise ValueError("Tokenizer vocabulary differs from checkpoint input embedding")

    table, embeddings = load_prepared_benchmark(args.data_dir)
    test_indices = select_split_indices(table, args.data_dir, "test")
    dataset = MultiTaskBenchmarkDataset(
        table, embeddings, test_indices, tokenizer, "log10_pgen_1mm", config.max_sequence_len
    )
    device = torch.device(args.device)
    loader = build_multitask_dataloader(
        dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    model = RTPCodecTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    matrix = pd.DataFrame(0, index=AA_ORDER, columns=AA_ORDER, dtype=np.int64)
    total_sequences = 0
    exact_sequences = 0
    length_matched_sequences = 0
    length_mismatched_sequences = 0
    wrong_positions = 0
    within_group = 0
    between_group = 0
    unexpected_residues = 0

    with torch.inference_mode():
        for batch in loader:
            tokens = batch["encoder_tokens"].to(device, non_blocking=True)
            mask = batch["encoder_mask"].to(device, non_blocking=True)
            generated = model.reconstruct(tokens, mask).cpu().numpy()
            predictions = [decode(row.tolist()) for row in generated]
            for target, prediction in zip(batch["sequence"], predictions):
                total_sequences += 1
                if target == prediction:
                    exact_sequences += 1
                if len(target) != len(prediction):
                    length_mismatched_sequences += 1
                    continue
                length_matched_sequences += 1
                for target_aa, predicted_aa in zip(target, prediction):
                    if target_aa == predicted_aa:
                        continue
                    if target_aa not in AA_TO_GROUP or predicted_aa not in AA_TO_GROUP:
                        unexpected_residues += 1
                        continue
                    matrix.loc[target_aa, predicted_aa] += 1
                    wrong_positions += 1
                    if AA_TO_GROUP[target_aa] == AA_TO_GROUP[predicted_aa]:
                        within_group += 1
                    else:
                        between_group += 1

    if total_sequences != len(test_indices):
        raise RuntimeError(f"Observed {total_sequences} sequences, expected {len(test_indices)}")
    if wrong_positions != within_group + between_group:
        raise RuntimeError("Substitution counts are inconsistent")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    matrix.to_csv(args.output_dir / "substitution_matrix.tsv", sep="\t")
    summary = {
        "status": "complete",
        "analysis": "locked_test_autoregressive_substitution_errors",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_dir": str(args.data_dir),
        "benchmark_ready_sha256": sha256(args.data_dir / "READY.json"),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256(args.tokenizer),
        "model_config": asdict(config),
        "test_sequences": total_sequences,
        "exact_sequences": exact_sequences,
        "exact_match": exact_sequences / total_sequences,
        "length_matched_sequences": length_matched_sequences,
        "length_mismatched_sequences": length_mismatched_sequences,
        "wrong_aligned_positions": wrong_positions,
        "within_group_errors": within_group,
        "between_group_errors": between_group,
        "within_group_fraction": within_group / wrong_positions if wrong_positions else None,
        "between_group_fraction": between_group / wrong_positions if wrong_positions else None,
        "unexpected_residue_errors": unexpected_residues,
        "groups": {name: "".join(sorted(aas)) for name, aas in AA_GROUPS.items()},
        "note": "Only incorrect aligned positions from equal-length target/prediction pairs enter the matrix.",
    }
    (args.output_dir / "RESULTS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_figure(matrix, args.output_dir / "substitution_confusion")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
