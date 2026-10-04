"""Issue 2: train the IRRM-CODEC decoder on one representation and score reconstruction.

The decoder architecture, optimizer, schedule and early stopping are identical for every
representation, so the only thing that varies between runs is the bottleneck vector the
decoder reads. Inputs are standardized with train-split statistics: the projections
arrive on very different scales, and a shared learning rate on unequal scales would make
"identical training settings" true on paper only. The repo's forward/inverse trainers
normalize their embeddings the same way.

Reported metrics: exact sequence match, amino-acid token accuracy, normalized
Levenshtein distance, fraction of predictions within edit distance 1, inference speed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader, TensorDataset

from irrm_codec.inverse_model import InverseModel
from irrm_codec.losses import inverse_loss
from irrm_codec.tokenization import AA_VOCAB, ID2AA, encode

GAP_ID = AA_VOCAB["-"]
from irrm_codec.utils import choose_device, save_checkpoint, set_seed, setup_logging


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="data/benchmark/trb")
    p.add_argument("--bottleneck-dir", default="data/benchmark/trb/bottleneck")
    p.add_argument("--representation", required=True)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    p.add_argument("--output-dir", default="artifacts/benchmark/reconstruction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="Early-stopping patience in epochs.")
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--train-subset", default="all", help="Training subset manifest: 1k, 10k or all.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def load_split_rows(dataset_dir, name):
    return pd.read_csv(Path(dataset_dir) / "manifests" / f"{name}.tsv", sep="\t")["row_index"].to_numpy()


def decode_tokens(token_ids):
    """Turn predicted token ids back into a gap-free amino-acid string."""
    return "".join(ID2AA[int(t)] for t in token_ids if int(t) > GAP_ID)


def build_tensors(matrix, rows, sequences, max_len):
    features = torch.from_numpy(np.asarray(matrix[rows], dtype=np.float32))
    targets = torch.tensor(
        [encode(sequences[i], max_len=max_len) for i in rows], dtype=torch.long
    )
    return features, targets


@torch.no_grad()
def evaluate(model, loader, device, sequences, rows, max_len, collect_predictions=False):
    """Run the decoder over a split and score it at the sequence level."""
    model.eval()
    loss_sum, steps = 0.0, 0
    token_hits, token_total = 0, 0
    predictions = []

    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        loss_sum += inverse_loss(logits, targets).item()
        steps += 1

        predicted = logits.argmax(dim=-1)
        # encode() gap-pads to a fixed width, so every position is filled and most of
        # them are gaps. Scoring those would let a model inflate token accuracy by
        # predicting padding, so accuracy counts real residues only.
        residues = targets.gt(GAP_ID)
        token_hits += predicted.eq(targets).logical_and(residues).sum().item()
        token_total += residues.sum().item()
        predictions.extend(decode_tokens(row) for row in predicted.cpu().numpy())

    truth = [sequences[i] for i in rows]
    distances = np.array([Levenshtein.distance(p, t) for p, t in zip(predictions, truth)])
    lengths = np.array([max(len(t), 1) for t in truth])

    metrics = {
        "loss": loss_sum / max(steps, 1),
        "token_accuracy": token_hits / max(token_total, 1),
        "exact_match": float(np.mean(distances == 0)),
        "normalized_levenshtein": float(np.mean(distances / lengths)),
        "within_edit_distance_1": float(np.mean(distances <= 1)),
        "mean_edit_distance": float(distances.mean()),
    }
    return (metrics, predictions, truth) if collect_predictions else (metrics, None, None)


@torch.no_grad()
def measure_inference_speed(model, features, device, batch_size):
    """Throughput on the test split, excluding metric computation."""
    model.eval()
    warmup = features[: min(batch_size, len(features))].to(device)
    model(warmup)

    started = time.perf_counter()
    for start in range(0, len(features), batch_size):
        model(features[start : start + batch_size].to(device))
    elapsed = time.perf_counter() - started
    return {
        "sequences_per_second": len(features) / elapsed,
        "ms_per_sequence": elapsed / len(features) * 1000,
        "batch_size": batch_size,
        "device": str(device),
    }


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = choose_device()

    run_name = f"{args.representation}_d{args.bottleneck_dim}_{args.train_subset}_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "train.log")

    dataset_dir = Path(args.dataset_dir)
    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    sequences = frame["junction_aa"].tolist()

    source = Path(args.bottleneck_dir) / f"{args.representation}_{args.bottleneck_dim}.npy"
    matrix = np.load(source, mmap_mode="r")
    log.info("run=%s device=%s features=%s", run_name, device, matrix.shape)

    train_rows = load_split_rows(dataset_dir, f"train_{args.train_subset}")
    val_rows = load_split_rows(dataset_dir, "val")
    test_rows = load_split_rows(dataset_dir, "test")

    train_x, train_y = build_tensors(matrix, train_rows, sequences, args.max_len)
    val_x, val_y = build_tensors(matrix, val_rows, sequences, args.max_len)
    test_x, test_y = build_tensors(matrix, test_rows, sequences, args.max_len)

    # Train-split statistics only, so no held-out information reaches the projection.
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x, val_x, test_x = ((t - mean) / std for t in (train_x, val_x, test_x))
    np.save(output_dir / "mean.npy", mean.numpy())
    np.save(output_dir / "std.npy", std.numpy())

    loaders = {
        name: DataLoader(
            TensorDataset(x, y),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
        )
        for name, (x, y) in (
            ("train", (train_x, train_y)),
            ("val", (val_x, val_y)),
            ("test", (test_x, test_y)),
        )
    }

    model = InverseModel(embedding_dim=args.bottleneck_dim, max_len=args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log.info("model params=%.2fM train=%d val=%d test=%d",
             sum(p.numel() for p in model.parameters()) / 1e6,
             len(train_rows), len(val_rows), len(test_rows))

    best_val = float("inf")
    best_epoch = 0
    history = []
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, steps = 0.0, 0
        for features, targets in loaders["train"]:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss = inverse_loss(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        train_loss = epoch_loss / max(steps, 1)
        val_metrics, _, _ = evaluate(model, loaders["val"], device, sequences, val_rows, args.max_len)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        log.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_exact=%.4f val_tok=%.4f",
            epoch, train_loss, val_metrics["loss"], val_metrics["exact_match"], val_metrics["token_accuracy"],
        )

        if val_metrics["loss"] < best_val - 1e-5:
            best_val, best_epoch = val_metrics["loss"], epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)
        elif epoch - best_epoch >= args.patience:
            log.info("early stopping at epoch=%d (best epoch=%d)", epoch, best_epoch)
            break

    training_seconds = time.perf_counter() - training_started
    save_checkpoint(output_dir / "last.pt", model, optimizer, len(history), history[-1])

    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device)["model_state"])
    test_metrics, predictions, truth = evaluate(
        model, loaders["test"], device, sequences, test_rows, args.max_len, collect_predictions=True
    )
    speed = measure_inference_speed(model, test_x, device, args.batch_size)

    pd.DataFrame(
        {
            "row_index": test_rows,
            "true_cdr3": truth,
            "predicted_cdr3": predictions,
            "edit_distance": [Levenshtein.distance(p, t) for p, t in zip(predictions, truth)],
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    result = {
        "run_name": run_name,
        "representation": args.representation,
        "bottleneck_dim": args.bottleneck_dim,
        "train_subset": args.train_subset,
        "train_size": int(len(train_rows)),
        "seed": args.seed,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "training_seconds": round(training_seconds, 2),
        "test": test_metrics,
        "inference": speed,
        "config": vars(args),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info("test exact_match=%.4f token_accuracy=%.4f", test_metrics["exact_match"], test_metrics["token_accuracy"])
    log.info("test norm_levenshtein=%.4f within_ed1=%.4f", test_metrics["normalized_levenshtein"], test_metrics["within_edit_distance_1"])
    log.info("training_seconds=%.1f inference=%.0f seq/s", training_seconds, speed["sequences_per_second"])
    log.info("wrote %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
