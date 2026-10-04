"""Train a position-aware pgen probe on frozen, unpooled residue states."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from benchmark.external_residue_models import (
    FrozenResiduePgenHead,
    ResidueArrayDataset,
    parameter_count,
)
from irrm_codec.losses import pgen_loss
from irrm_codec.utils import choose_device, save_checkpoint, set_seed, setup_logging


REPRESENTATIONS = ("esm2_8m", "tcr_bert", "sceptr")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--residue-dir", required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="log10_pgen_1mm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_rows(dataset_dir, split):
    return pd.read_csv(Path(dataset_dir) / "manifests" / f"{split}.tsv", sep="\t")[
        "row_index"
    ].to_numpy()


def regression_metrics(prediction, truth):
    residual = prediction - truth
    variance = np.var(truth)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1 - np.mean(residual**2) / variance),
        "pearson_r": float(stats.pearsonr(prediction, truth)[0]),
        "spearman_rho": float(stats.spearmanr(prediction, truth)[0]),
    }


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, steps = 0.0, 0
    for features, mask, target in loader:
        features, mask, target = features.to(device), mask.to(device), target.float().to(device)
        with torch.set_grad_enabled(training):
            loss = pgen_loss(model(features, mask), target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    output = []
    for features, mask, _ in loader:
        output.append(model(features.to(device), mask.to(device)).cpu().numpy())
    return np.concatenate(output)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = choose_device()
    dataset_dir, residue_dir = Path(args.dataset_dir), Path(args.residue_dir)
    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    rows = {split: load_rows(dataset_dir, split) for split in ("train", "val", "test")}
    raw_target = frame[args.target].to_numpy(dtype=np.float64)
    target_mean = float(raw_target[rows["train"]].mean())
    target_std = float(raw_target[rows["train"]].std())
    scaled = ((raw_target - target_mean) / target_std).astype(np.float32)
    feature_path = residue_dir / f"{args.representation}_residue.npy"
    mask_path = residue_dir / f"{args.representation}_residue_mask.npy"
    feature_matrix = np.load(feature_path, mmap_mode="r")
    loaders = {
        split: DataLoader(
            ResidueArrayDataset(feature_path, mask_path, rows[split], scaled[rows[split]]),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split in rows
    }
    model = FrozenResiduePgenHead(input_dim=feature_matrix.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_name = f"{args.representation}_residue_{args.target}_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "train.log")
    log.info(
        "run=%s device=%s frozen_shape=%s params=%d",
        run_name,
        device,
        feature_matrix.shape,
        parameter_count(model),
    )
    best_val, best_epoch, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], device, optimizer)
        val_loss = run_epoch(model, loaders["val"], device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch % 5 == 0 or epoch == 1:
            log.info("epoch=%d train_loss=%.5f val_loss=%.5f", epoch, train_loss, val_loss)
        if val_loss < best_val - 1e-6:
            best_val, best_epoch = val_loss, epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, {"val_loss": val_loss})
        elif epoch - best_epoch >= args.patience:
            break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device)["model_state"])
    scaled_prediction = predict(model, loaders["test"], device)
    prediction = scaled_prediction * target_std + target_mean
    truth = raw_target[rows["test"]]
    metrics = regression_metrics(prediction, truth)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(
        {
            "row_index": rows["test"],
            "cdr3": frame.junction_aa.iloc[rows["test"]].to_numpy(),
            "true": truth,
            "predicted": prediction,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    result = {
        "run_name": run_name,
        "representation": args.representation,
        "interface": "frozen_residue_states_no_global_pooling",
        "target": args.target,
        "seed": args.seed,
        "train_size": int(len(rows["train"])),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "training_seconds": round(training_seconds, 2),
        "parameters": parameter_count(model),
        "frozen_shape": list(feature_matrix.shape),
        "target_scaling": {"fit_split": "train", "mean": target_mean, "std": target_std},
        "test": metrics,
        "config": vars(args),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("status=complete test=%s", metrics)


if __name__ == "__main__":
    main()
