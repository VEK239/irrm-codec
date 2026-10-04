"""Issue 3: train one Pgen-prediction arm and score it on the benchmark test split.

Arms:
  irrm_scratch      PgenModel from a random initialization
  irrm_pretrained   PgenModel initialized from the sequence-to-TCRemP encoder
  tfidf_ridge       1-3-mer character TF-IDF + Ridge regression
  sceptr_mlp        frozen SCEPTR embeddings + the shared regression head
  tcr_bert_mlp      frozen TCR-BERT embeddings + the same head
  esm2_8m_mlp       frozen ESM-2 8M embeddings + the same head

The three frozen-embedding arms share one head and one training configuration, so the
only thing that varies between them is the representation. The two IRRM arms differ only
in initialization, which is the comparison the issue asks about.

Targets stay in raw log10 units, matching the repo's pgen trainer, so RMSE, MAE and bias
are directly interpretable as log10 error.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from irrm_codec.losses import pgen_loss
from irrm_codec.pgen_model import PgenModel
from irrm_codec.tokenization import PAD_ID, encode
from irrm_codec.utils import choose_device, save_checkpoint, set_seed, setup_logging

SEQUENCE_ARMS = ("irrm_scratch", "irrm_pretrained")
FROZEN_ARMS = {"sceptr_mlp": "sceptr", "tcr_bert_mlp": "tcr_bert", "esm2_8m_mlp": "esm2_8m"}
ARMS = (*SEQUENCE_ARMS, "tfidf_ridge", *FROZEN_ARMS)

# Ridge has no training loop, so the penalty is selected on validation instead. This is
# the baseline's equivalent of early stopping; fixing it arbitrarily would handicap it.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--target", default="log10_pgen_1mm", choices=("log10_pgen", "log10_pgen_1mm"))
    p.add_argument("--dataset-dir", default="data/benchmark/trb")
    p.add_argument("--representations-dir", default="data/benchmark/trb/representations")
    p.add_argument("--pretrained-checkpoint", default="artifacts/benchmark/forward_encoder/best.pt")
    p.add_argument("--output-dir", default="artifacts/benchmark/pgen")
    p.add_argument("--train-subset", default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Wider than the reconstruction runs use: the cosine schedule only pays off if training
    # reaches the low-learning-rate tail, and stopping at epoch 17 of 60 would decay the
    # rate by under a fifth.
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


class RegressionHead(nn.Module):
    """One head shared by every frozen representation, so only the input differs."""

    def __init__(self, input_dim, hidden_dims=(512, 256), dropout=0.2):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(previous, width), nn.GELU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features).squeeze(-1)


def load_rows(dataset_dir, name):
    return pd.read_csv(Path(dataset_dir) / "manifests" / f"{name}.tsv", sep="\t")["row_index"].to_numpy()


def peak_memory_mb():
    """True peak resident set size, plus GPU peak when a device was used."""
    result = {}
    try:
        import resource

        # ru_maxrss is in kilobytes on Linux.
        result["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, AttributeError):
        try:
            import psutil

            result["peak_rss_mb"] = psutil.Process().memory_info().rss / 1024**2
        except ImportError:
            result["peak_rss_mb"] = None
    if torch.cuda.is_available():
        result["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    return result


def regression_metrics(predictions, truth):
    residual = predictions - truth
    total_variance = np.var(truth)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - np.mean(residual**2) / total_variance) if total_variance > 0 else float("nan"),
        "pearson_r": float(stats.pearsonr(predictions, truth)[0]),
        "spearman_rho": float(stats.spearmanr(predictions, truth)[0]),
    }


def build_features(args, frame, rows_by_split, log):
    """Return per-split model inputs for the requested arm."""
    if args.arm in SEQUENCE_ARMS:
        tokens = {
            name: torch.tensor(
                [encode(frame.junction_aa.iloc[i], max_len=args.max_len) for i in rows],
                dtype=torch.long,
            )
            for name, rows in rows_by_split.items()
        }
        return tokens, None

    representation = FROZEN_ARMS[args.arm]
    matrix = np.load(Path(args.representations_dir) / f"{representation}.npy", mmap_mode="r")
    features = {
        name: torch.from_numpy(np.asarray(matrix[rows], dtype=np.float32))
        for name, rows in rows_by_split.items()
    }
    # Train-split statistics only: a shared learning rate across representations with
    # different scales would not be the identical setting the issue requires.
    mean = features["train"].mean(dim=0, keepdim=True)
    std = features["train"].std(dim=0, keepdim=True).clamp_min(1e-6)
    for name in features:
        features[name] = (features[name] - mean) / std
    log.info("%s features dim=%d", representation, features["train"].shape[1])
    return features, (mean, std)


def build_model(args, input_dim, log):
    if args.arm in SEQUENCE_ARMS:
        model = PgenModel(max_len=args.max_len)
        if args.arm == "irrm_pretrained":
            state = torch.load(args.pretrained_checkpoint, map_location="cpu")["model_state"]
            own = model.state_dict()
            # Everything but the output layer matches: PgenModel is ForwardModel with
            # output_dim=1, so only mlp.<last> differs in shape.
            transferable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
            model.load_state_dict(transferable, strict=False)
            log.info(
                "transferred %d/%d tensors from %s",
                len(transferable), len(own), args.pretrained_checkpoint,
            )
        return model
    return RegressionHead(input_dim)


def run_epoch(model, loader, device, is_sequence, optimizer=None):
    is_train = optimizer is not None
    model.train(mode=is_train)
    loss_sum, steps = 0.0, 0
    for batch in loader:
        inputs, target = batch[0].to(device), batch[-1].to(device)
        with torch.set_grad_enabled(is_train):
            prediction = model(inputs, inputs.ne(PAD_ID)) if is_sequence else model(inputs)
            loss = pgen_loss(prediction, target)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        loss_sum += loss.item()
        steps += 1
    return loss_sum / max(steps, 1)


@torch.no_grad()
def predict_all(model, inputs, device, is_sequence, batch_size):
    model.eval()
    chunks = []
    for start in range(0, len(inputs), batch_size):
        block = inputs[start : start + batch_size].to(device)
        prediction = model(block, block.ne(PAD_ID)) if is_sequence else model(block)
        chunks.append(prediction.cpu().numpy())
    return np.concatenate(chunks)


def fit_ridge(args, frame, rows_by_split, targets, log):
    """1-3-mer character TF-IDF into Ridge, with the penalty chosen on validation."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    started = time.perf_counter()
    sequences = {
        name: frame.junction_aa.iloc[rows].tolist() for name, rows in rows_by_split.items()
    }
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    train_features = vectorizer.fit_transform(sequences["train"])
    val_features = vectorizer.transform(sequences["val"])

    best = (None, float("inf"), None)
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(train_features, targets["train"])
        rmse = float(np.sqrt(np.mean((model.predict(val_features) - targets["val"]) ** 2)))
        log.info("ridge alpha=%.3g val_rmse=%.4f", alpha, rmse)
        if rmse < best[1]:
            best = (model, rmse, alpha)
    model, val_rmse, alpha = best
    training_seconds = time.perf_counter() - started
    log.info("selected alpha=%.3g val_rmse=%.4f features=%d", alpha, val_rmse, train_features.shape[1])

    # Vectorization is part of this arm's inference path, so it belongs inside the timed
    # section; excluding it would report the Ridge solve alone and overstate throughput
    # by orders of magnitude.
    inference_started = time.perf_counter()
    test_features = vectorizer.transform(sequences["test"])
    predictions = model.predict(test_features)
    elapsed = time.perf_counter() - inference_started

    return (
        predictions,
        training_seconds,
        {
            "sequences_per_second": len(predictions) / elapsed,
            "ms_per_sequence": elapsed / len(predictions) * 1000,
            "batch_size": len(predictions),
            "device": "cpu",
        },
        {"alpha": alpha, "val_rmse": val_rmse, "n_features": int(train_features.shape[1])},
        [],
    )


def fit_neural(args, features, targets, log):
    device = choose_device()
    is_sequence = args.arm in SEQUENCE_ARMS
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loaders = {
        name: DataLoader(
            TensorDataset(features[name], torch.from_numpy(targets[name].astype(np.float32))),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
        )
        for name in ("train", "val")
    }

    model = build_model(args, features["train"].shape[-1], log).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # A constant learning rate leaves the model bouncing around the minimum: validation
    # loss oscillates, early stopping fires at an arbitrary epoch, and the IRRM arms end up
    # spreading roughly tenfold wider across seeds than the frozen-embedding arms.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log.info("device=%s params=%.2fM", device, sum(p.numel() for p in model.parameters()) / 1e6)

    best_val, best_epoch = float("inf"), 0
    best_state = None
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], device, is_sequence, optimizer)
        val_loss = run_epoch(model, loaders["val"], device, is_sequence)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch % 5 == 0 or epoch == 1:
            log.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            log.info("early stopping at epoch=%d (best epoch=%d)", epoch, best_epoch)
            break

    training_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)

    inference_started = time.perf_counter()
    predictions = predict_all(model, features["test"], device, is_sequence, args.batch_size)
    elapsed = time.perf_counter() - inference_started

    return (
        predictions,
        training_seconds,
        {
            "sequences_per_second": len(predictions) / elapsed,
            "ms_per_sequence": elapsed / len(predictions) * 1000,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        {"best_val_loss": best_val, "best_epoch": best_epoch, "epochs_ran": len(history)},
        history,
        model,
        optimizer,
    )


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    run_name = f"{args.arm}_{args.target}_{args.train_subset}_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "train.log")

    dataset_dir = Path(args.dataset_dir)
    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    rows_by_split = {
        "train": load_rows(dataset_dir, f"train_{args.train_subset}"),
        "val": load_rows(dataset_dir, "val"),
        "test": load_rows(dataset_dir, "test"),
    }
    targets = {
        name: frame[args.target].to_numpy()[rows] for name, rows in rows_by_split.items()
    }
    # log10 Pgen sits near -6.5, so a freshly initialized network spends its first epochs
    # just finding the offset while Ridge fits an intercept for free. Standardizing the
    # target on train statistics removes that asymmetry; predictions are mapped back
    # before any metric is computed, so the reported errors stay in log10 units.
    target_mean = float(targets["train"].mean())
    target_std = max(float(targets["train"].std()), 1e-6)
    scaled_targets = {name: (values - target_mean) / target_std for name, values in targets.items()}
    log.info(
        "run=%s train=%d val=%d test=%d target_mean=%.3f",
        run_name, *(len(r) for r in rows_by_split.values()), targets["train"].mean(),
    )

    history, model, optimizer = [], None, None
    if args.arm == "tfidf_ridge":
        scaled_predictions, training_seconds, speed, extra, history = fit_ridge(
            args, frame, rows_by_split, scaled_targets, log
        )
    else:
        features, _ = build_features(args, frame, rows_by_split, log)
        scaled_predictions, training_seconds, speed, extra, history, model, optimizer = fit_neural(
            args, features, scaled_targets, log
        )
        save_checkpoint(output_dir / "best.pt", model, optimizer, extra["best_epoch"], extra)

    predictions = scaled_predictions * target_std + target_mean
    metrics = regression_metrics(predictions, targets["test"])
    pd.DataFrame(
        {
            "row_index": rows_by_split["test"],
            "cdr3": frame.junction_aa.iloc[rows_by_split["test"]].to_numpy(),
            "true": targets["test"],
            "predicted": predictions,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    if history:
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    result = {
        "run_name": run_name,
        "arm": args.arm,
        "target": args.target,
        "train_subset": args.train_subset,
        "train_size": int(len(rows_by_split["train"])),
        "seed": args.seed,
        "training_seconds": round(training_seconds, 2),
        "test": metrics,
        "inference": speed,
        "memory": peak_memory_mb(),
        "target_scaling": {"mean": target_mean, "std": target_std},
        "arm_details": extra,
        "config": vars(args),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info(
        "test rmse=%.4f mae=%.4f r2=%.4f pearson=%.4f spearman=%.4f bias=%.4f",
        metrics["rmse"], metrics["mae"], metrics["r2"],
        metrics["pearson_r"], metrics["spearman_rho"], metrics["bias"],
    )
    log.info(
        "training=%.0f s inference=%.0f seq/s peak_rss=%.0f MB",
        training_seconds, speed["sequences_per_second"], result["memory"].get("peak_rss_mb") or 0,
    )
    log.info("wrote %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
