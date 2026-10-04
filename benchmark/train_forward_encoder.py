"""Issue 3, step 1: pretrain the sequence-to-TCRemP encoder on the benchmark split.

One of the Pgen arms is initialized from a pretrained sequence-to-TCRemP encoder. The
published checkpoints were trained with their own random split, so reusing one would leak
benchmark test sequences into pretraining and inflate exactly the comparison the issue
asks about. This trains the encoder on the benchmark training split only.

PgenModel is ForwardModel with output_dim=1, so everything except the final output layer
transfers directly into the Pgen arm.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from irrm_codec.forward_model import ForwardModel
from irrm_codec.losses import forward_loss, forward_metrics
from irrm_codec.tokenization import PAD_ID, encode
from irrm_codec.utils import choose_device, save_checkpoint, set_seed, setup_logging


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="data/benchmark/trb")
    p.add_argument("--output-dir", default="artifacts/benchmark/forward_encoder")
    p.add_argument("--train-subset", default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


class SequenceToEmbedding(Dataset):
    """Tokenized CDR3 in, normalized TCRemP vector out.

    Shuffled training reads the 9000-wide rows in random order, which is slow against a
    memory-mapped file on network storage. The split is materialized once instead: the
    full training slice is about 2.9 GB resident, which is cheaper than re-reading it
    from NFS every epoch.
    """

    def __init__(self, matrix, rows, sequences, mean, std, max_len):
        self.tokens = torch.tensor(
            [encode(sequences[i], max_len=max_len) for i in rows], dtype=torch.long
        )
        targets = np.empty((len(rows), matrix.shape[1]), dtype=np.float32)
        for start in range(0, len(rows), 4096):
            block = rows[start : start + 4096]
            targets[start : start + len(block)] = np.asarray(matrix[block], dtype=np.float32)
        self.targets = torch.from_numpy((targets - mean) / std)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        return tokens, tokens.ne(PAD_ID), self.targets[index]


def load_rows(dataset_dir, name):
    return pd.read_csv(Path(dataset_dir) / "manifests" / f"{name}.tsv", sep="\t")["row_index"].to_numpy()


def train_statistics(matrix, rows, batch_size=4096):
    """Streaming mean and standard deviation over the training rows only."""
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    total_sq = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        block = np.asarray(matrix[rows[start : start + batch_size]], dtype=np.float64)
        total += block.sum(axis=0)
        total_sq += (block ** 2).sum(axis=0)
    mean = total / len(rows)
    variance = np.maximum(total_sq / len(rows) - mean ** 2, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).clip(1e-6).astype(np.float32)


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(mode=is_train)
    sums = {"loss": 0.0, "mse": 0.0, "cosine": 0.0}
    steps = 0

    for tokens, mask, target in loader:
        tokens, mask, target = tokens.to(device), mask.to(device), target.to(device)
        with torch.set_grad_enabled(is_train):
            prediction = model(tokens, mask)
            loss = forward_loss(prediction, target)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = forward_metrics(prediction.detach(), target)
        sums["loss"] += loss.item()
        sums["mse"] += metrics["mse"]
        sums["cosine"] += metrics["cosine"]
        steps += 1

    return {name: value / max(steps, 1) for name, value in sums.items()}


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = choose_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "train.log")

    dataset_dir = Path(args.dataset_dir)
    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    sequences = frame["junction_aa"].tolist()
    matrix = np.load(dataset_dir / "embeddings.npy", mmap_mode="r")

    train_rows = load_rows(dataset_dir, f"train_{args.train_subset}")
    val_rows = load_rows(dataset_dir, "val")
    log.info("device=%s embeddings=%s train=%d val=%d", device, matrix.shape, len(train_rows), len(val_rows))

    mean, std = train_statistics(matrix, train_rows)
    np.save(output_dir / "mean.npy", mean)
    np.save(output_dir / "std.npy", std)

    loaders = {
        name: DataLoader(
            SequenceToEmbedding(matrix, rows, sequences, mean, std, args.max_len),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
        )
        for name, rows in (("train", train_rows), ("val", val_rows))
    }

    model = ForwardModel(output_dim=matrix.shape[1], max_len=args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # A constant learning rate leaves the model bouncing around the minimum, which shows
    # up as an oscillating validation loss and, downstream, as large seed-to-seed spread.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log.info("model params=%.2fM output_dim=%d", sum(p.numel() for p in model.parameters()) / 1e6, matrix.shape[1])

    best_val = float("inf")
    best_epoch = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        log.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_cosine=%.4f",
            epoch, train_metrics["loss"], val_metrics["loss"], val_metrics["cosine"],
        )

        if val_metrics["loss"] < best_val - 1e-5:
            best_val, best_epoch = val_metrics["loss"], epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)
        elif epoch - best_epoch >= args.patience:
            log.info("early stopping at epoch=%d (best epoch=%d)", epoch, best_epoch)
            break

    elapsed = time.perf_counter() - started
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "epochs_ran": len(history),
                "training_seconds": round(elapsed, 2),
                "embedding_dim": int(matrix.shape[1]),
                "train_size": int(len(train_rows)),
                "final": history[-1],
                "config": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info("=" * 60)
    log.info("best_val_loss=%.4f at epoch=%d in %.0f s", best_val, best_epoch, elapsed)
    log.info("checkpoint for Pgen transfer: %s", output_dir / "best.pt")


if __name__ == "__main__":
    main()
