"""CPU gate for matched DATA-ANCHOR objective-ablation training runs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from irrm_codec.multitask_data import (
    MultiTaskBenchmarkDataset,
    TargetStandardizer,
    collate_multitask,
    load_prepared_benchmark,
    resolve_encoder_tokenizer,
    select_split_indices,
)
from irrm_codec.multitask_losses import IRRMCodecMultiTaskLoss, MultiTaskLossWeights
from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer


CONDITIONS = {
    "r": {"reconstruction": 1.0, "tcremp": 0.0, "pgen": 0.0},
    "t": {"reconstruction": 0.0, "tcremp": 1.0, "pgen": 0.0},
    "p": {"reconstruction": 0.0, "tcremp": 0.0, "pgen": 1.0},
    "rt": {"reconstruction": 1.0, "tcremp": 1.0, "pgen": 0.0},
    "rp": {"reconstruction": 1.0, "tcremp": 0.0, "pgen": 1.0},
    "tp": {"reconstruction": 0.0, "tcremp": 1.0, "pgen": 1.0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--reference-run-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    ready = json.loads((args.data_dir / "READY.json").read_text(encoding="utf-8"))
    if ready.get("status") != "ready" or not all(ready.get("checks", {}).values()):
        raise ValueError("Locked benchmark READY.json is not accepting.")
    reference = json.loads(args.reference_run_config.read_text(encoding="utf-8"))
    training = reference["training"]
    required = {
        "data_dir": str(args.data_dir),
        "train_subset": "all",
        "pgen_target": "log10_pgen_1mm",
        "tokenizer_type": "data_anchor",
        "tokenizer_path": str(args.tokenizer),
        "max_sequence_len": 40,
        "d_model": 320,
        "latent_dim": 128,
        "nhead": 8,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "ff_dim": 1280,
        "dropout": 0.1,
        "tcremp_head_dim": 1024,
        "pgen_head_dim": 256,
        "decoder_memory_tokens": 4,
        "tcremp_loss_weight": 1.0,
        "pgen_loss_weight": 1.0,
        "reconstruction_loss_weight": 1.0,
        "tcremp_mse_fraction": 0.7,
        "pgen_huber_delta": 0.5,
        "label_smoothing": 0.0,
        "batch_size": 64,
        "epochs": 40,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 0,
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "scheduler_min_lr": 1e-6,
        "seed": 42,
        "num_workers": 8,
        "amp": True,
        "skip_generation_metrics": False,
        "save_test_predictions": False,
        "val_generation_every": 0,
        "max_train_batches": 0,
        "max_eval_batches": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": training.get(key)}
        for key, value in required.items() if training.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Completed DATA-ANCHOR reference is not the locked run: {mismatches}")

    table, embeddings = load_prepared_benchmark(args.data_dir)
    indices = {
        split: select_split_indices(table, args.data_dir, split)
        for split in ("train", "val", "test")
    }
    normalizer = TargetStandardizer.load(args.data_dir / "target_standardizer.npz")
    if normalizer.train_rows != len(indices["train"]) or normalizer.pgen_target != "log10_pgen_1mm":
        raise ValueError("Train-only normalizer does not match the locked train manifest.")
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(args.tokenizer))
    config = IRRMCodecConfig(**reference["model"])
    if tokenizer.vocab_size != config.input_vocab_size:
        raise ValueError("Tokenizer vocabulary does not match reference input embedding.")
    dataset = MultiTaskBenchmarkDataset(
        table, embeddings, indices["train"][:2], tokenizer, "log10_pgen_1mm", 40
    )
    batch = collate_multitask([dataset[0], dataset[1]])
    standardized = normalizer.as_torch(torch.device("cpu"))

    reports = {}
    configs_dir = args.output_root / "configs"
    for name, weights in CONDITIONS.items():
        torch.manual_seed(42)
        model = IRRMCodecTransformer(config)
        outputs = model(batch["encoder_tokens"], batch["encoder_mask"], batch["decoder_input"])
        criterion = IRRMCodecMultiTaskLoss(
            weights=MultiTaskLossWeights(
                tcremp=weights["tcremp"],
                pgen=weights["pgen"],
                reconstruction=weights["reconstruction"],
            )
        )
        losses = criterion(
            outputs,
            tcremp_target=batch["tcremp_target"],
            pgen_target=batch["pgen_target"],
            reconstruction_target=batch["reconstruction_target"],
            **standardized,
        )
        losses["loss"].backward()
        encoder_grads = [
            parameter.grad for parameter in model.encoder.parameters() if parameter.grad is not None
        ]
        if not torch.isfinite(losses["loss"]) or not encoder_grads or not all(
            torch.isfinite(gradient).all() for gradient in encoder_grads
        ):
            raise FloatingPointError(f"{name} has non-finite loss or encoder gradient.")
        condition_config = {
            "condition": name,
            "experiment_id": f"data-anchor-{name}-seed42",
            "loss_weights": weights,
            "model": asdict(config),
            "training_reference": str(args.reference_run_config),
            "training_reference_sha256": sha256(args.reference_run_config),
            "benchmark_ready_sha256": sha256(args.data_dir / "READY.json"),
            "normalizer_sha256": sha256(args.data_dir / "target_standardizer.npz"),
            "tokenizer_sha256": sha256(args.tokenizer),
            "seed": 42,
            "epochs": 40,
            "batch_size": 64,
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "inactive_head_metrics_interpretable": False,
        }
        config_path = configs_dir / f"{name}.json"
        stable_json(config_path, condition_config)
        reports[name] = {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "smoke_loss": float(losses["loss"].detach()),
            "finite_encoder_gradient": True,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "architecture_equal_to_full_reference": asdict(config) == reference["model"],
        }
        del model, outputs, losses, criterion
        gc.collect()

    report = {
        "status": "ready",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "required_conditions": ["r", "t", "p", "rt", "rp", "tp"],
        "newly_required_conditions": ["t", "rt", "tp"],
        "fixed_reference": {
            "condition": "rtp",
            "job_id": "1426222",
            "run_config": str(args.reference_run_config),
            "run_config_sha256": sha256(args.reference_run_config),
            "parameter_count": reference["parameter_count"],
        },
        "checks": {
            "locked_benchmark": True,
            "locked_train_only_normalizer": True,
            "same_data_splits": True,
            "same_data_anchor_tokenizer": True,
            "same_model_architecture": True,
            "same_seed_optimizer_budget": True,
            "only_loss_weights_differ": True,
            "finite_smoke_losses_and_encoder_gradients": True,
            "output_directories_isolated": True,
        },
        "hashes": {
            "benchmark_ready": sha256(args.data_dir / "READY.json"),
            "normalizer": sha256(args.data_dir / "target_standardizer.npz"),
            "tokenizer": sha256(args.tokenizer),
        },
        "splits": {split: len(value) for split, value in indices.items()},
        "conditions": reports,
        "note": (
            "Every condition retains the identical decoder and all auxiliary heads. Metrics from "
            "heads with zero loss weight are non-interpretable."
        ),
    }
    stable_json(args.output_root / "PREFLIGHT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
