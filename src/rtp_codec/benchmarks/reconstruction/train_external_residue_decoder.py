"""Train one decoder on one frozen, unpooled residue-state representation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rtp_codec.benchmarks.representations.external_residue_models import (
    FrozenResidueDecoder,
    ResidueArrayDataset,
    parameter_count,
)
from rtp_codec.training.losses import inverse_loss
from rtp_codec.tokenization.character import AA_VOCAB, ID2AA, encode
from rtp_codec.utils import choose_device, save_checkpoint, set_seed, setup_logging


GAP_ID = AA_VOCAB["-"]
REPRESENTATIONS = ("esm2_8m", "tcr_bert", "sceptr")


def levenshtein_distance(left: str, right: str) -> int:
    """Return exact edit distance without an optional third-party dependency."""
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_char in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_char in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--residue-dir", required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_rows(dataset_dir, split):
    return pd.read_csv(Path(dataset_dir) / "manifests" / f"{split}.tsv", sep="\t")[
        "row_index"
    ].to_numpy()


def decode_tokens(token_ids):
    return "".join(ID2AA[int(token)] for token in token_ids if int(token) > GAP_ID)


def make_loader(features_path, mask_path, rows, targets, args, shuffle):
    dataset = ResidueArrayDataset(features_path, mask_path, rows, targets[rows])
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(model, loader, device, optimizer):
    model.train()
    total, steps = 0.0, 0
    for features, mask, target in loader:
        features, mask, target = features.to(device), mask.to(device), target.long().to(device)
        loss = inverse_loss(model(features, mask), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, device, sequences, rows, collect=False):
    model.eval()
    total, steps, token_hits, token_total = 0.0, 0, 0, 0
    predictions = []
    for features, mask, target in loader:
        features, mask, target = features.to(device), mask.to(device), target.long().to(device)
        logits = model(features, mask)
        total += inverse_loss(logits, target).item()
        steps += 1
        predicted = logits.argmax(dim=-1)
        residues = target.gt(GAP_ID)
        token_hits += predicted.eq(target).logical_and(residues).sum().item()
        token_total += residues.sum().item()
        predictions.extend(decode_tokens(row) for row in predicted.cpu().numpy())
    truth = [sequences[index] for index in rows]
    distances = np.asarray([levenshtein_distance(pred, true) for pred, true in zip(predictions, truth)])
    lengths = np.asarray([max(len(sequence), 1) for sequence in truth])
    metrics = {
        "loss": total / max(steps, 1),
        "token_accuracy": token_hits / max(token_total, 1),
        "exact_match": float(np.mean(distances == 0)),
        "normalized_levenshtein": float(np.mean(distances / lengths)),
        "within_edit_distance_1": float(np.mean(distances <= 1)),
        "mean_edit_distance": float(distances.mean()),
    }
    return metrics, (predictions if collect else None), (truth if collect else None)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = choose_device()
    dataset_dir, residue_dir = Path(args.dataset_dir), Path(args.residue_dir)
    run_name = f"{args.representation}_residue_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "train.log")

    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    sequences = frame.junction_aa.tolist()
    targets = np.asarray([encode(sequence, max_len=args.max_len) for sequence in sequences], dtype=np.int64)
    rows = {split: load_rows(dataset_dir, split) for split in ("train", "val", "test")}
    feature_path = residue_dir / f"{args.representation}_residue.npy"
    mask_path = residue_dir / f"{args.representation}_residue_mask.npy"
    feature_matrix = np.load(feature_path, mmap_mode="r")
    if feature_matrix.shape[0] != len(frame) or feature_matrix.shape[1] != args.max_len:
        raise ValueError(f"Frozen feature shape mismatch: {feature_matrix.shape}")
    loaders = {
        split: make_loader(feature_path, mask_path, rows[split], targets, args, split == "train")
        for split in rows
    }
    model = FrozenResidueDecoder(
        input_dim=feature_matrix.shape[-1], max_len=args.max_len
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log.info(
        "run=%s device=%s frozen_shape=%s params=%d train=%d val=%d test=%d",
        run_name,
        device,
        feature_matrix.shape,
        parameter_count(model),
        *(len(rows[split]) for split in ("train", "val", "test")),
    )

    best_val, best_epoch, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], device, optimizer)
        val_metrics, _, _ = evaluate(model, loaders["val"], device, sequences, rows["val"])
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        log.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_exact=%.4f val_token=%.4f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["exact_match"],
            val_metrics["token_accuracy"],
        )
        if val_metrics["loss"] < best_val - 1e-5:
            best_val, best_epoch = val_metrics["loss"], epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)
        elif epoch - best_epoch >= args.patience:
            break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test, predictions, truth = evaluate(
        model, loaders["test"], device, sequences, rows["test"], collect=True
    )
    pd.DataFrame(
        {
            "row_index": rows["test"],
            "true_cdr3": truth,
            "predicted_cdr3": predictions,
            "edit_distance": [levenshtein_distance(pred, true) for pred, true in zip(predictions, truth)],
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    result = {
        "run_name": run_name,
        "representation": args.representation,
        "interface": "frozen_residue_states_no_global_pooling",
        "seed": args.seed,
        "train_size": int(len(rows["train"])),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "training_seconds": round(training_seconds, 2),
        "parameters": parameter_count(model),
        "frozen_shape": list(feature_matrix.shape),
        "test": test,
        "config": vars(args),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("status=complete test=%s", test)


if __name__ == "__main__":
    main()
