"""Validate RQ5 Stage-2 encoder-depth candidates against the selected d128 run."""

import argparse
import gc
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from irrm_codec.multitask_data import TargetStandardizer, load_prepared_benchmark, select_split_indices
from irrm_codec.multitask_losses import IRRMCodecMultiTaskLoss, MultiTaskLossWeights
from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer
from irrm_codec.tokenization import AA_VOCAB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-config", required=True)
    parser.add_argument("--d128-test-metrics", required=True)
    parser.add_argument("--d320-test-metrics", required=True)
    parser.add_argument("--d512-test-metrics", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_run_config)
    baseline = json.loads(baseline_path.read_text())
    width_metrics = {
        "128": json.loads(Path(args.d128_test_metrics).read_text()),
        "320": json.loads(Path(args.d320_test_metrics).read_text()),
        "512": json.loads(Path(args.d512_test_metrics).read_text()),
    }
    validation_losses = {
        width: metrics["best_checkpoint_val_loss"] for width, metrics in width_metrics.items()
    }
    if min(validation_losses, key=validation_losses.get) != "128":
        raise ValueError(f"latent_dim=128 was not selected by validation: {validation_losses}")

    data_dir = Path(args.data_dir)
    ready = json.loads((data_dir / "READY.json").read_text())
    if ready.get("status") != "ready" or not all(ready.get("checks", {}).values()):
        raise ValueError("Locked benchmark READY checks failed.")
    table, embeddings = load_prepared_benchmark(data_dir)
    splits = {name: select_split_indices(table, data_dir, name) for name in ("train", "val", "test")}
    standardizer = TargetStandardizer.load(data_dir / "target_standardizer.npz")
    if standardizer.train_rows != len(splits["train"]) or standardizer.pgen_target != "log10_pgen_1mm":
        raise ValueError("Locked train-only normalizer does not match the selected d128 run.")

    expected_training = {
        "train_subset": "all", "pgen_target": "log10_pgen_1mm", "tokenizer_type": "char",
        "tokenizer_path": None, "max_sequence_len": 40, "d_model": 320, "latent_dim": 128,
        "nhead": 8, "encoder_layers": 4, "decoder_layers": 4, "ff_dim": 1280,
        "dropout": 0.1, "tcremp_head_dim": 1024, "pgen_head_dim": 256,
        "decoder_memory_tokens": 4, "tcremp_loss_weight": 1.0, "pgen_loss_weight": 1.0,
        "reconstruction_loss_weight": 1.0, "tcremp_mse_fraction": 0.7,
        "pgen_huber_delta": 0.5, "label_smoothing": 0.0, "batch_size": 64,
        "epochs": 40, "lr": 0.0003, "weight_decay": 0.0001,
        "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
        "early_stopping_patience": 0, "scheduler_factor": 0.5, "scheduler_patience": 2,
        "scheduler_min_lr": 1e-6, "seed": 42, "num_workers": 8, "amp": True,
        "skip_generation_metrics": False, "save_test_predictions": False,
        "val_generation_every": 0, "max_train_batches": 0, "max_eval_batches": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": baseline["training"].get(key)}
        for key, expected in expected_training.items()
        if baseline["training"].get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Selected d128 run differs from locked RQ5 settings: {mismatches}")
    if baseline["model"]["latent_dim"] != 128 or baseline["model"]["encoder_layers"] != 4:
        raise ValueError("Selected baseline is not the completed d128, 4+4-layer run.")
    if baseline["parameter_count"] != width_metrics["128"]["parameter_count"]:
        raise ValueError("Selected d128 run/metric parameter counts disagree.")

    results = {}
    for depth in (2, 6):
        torch.manual_seed(42)
        model_payload = dict(baseline["model"])
        model_payload["encoder_layers"] = depth
        config = IRRMCodecConfig(**model_payload)
        config.validate()
        changed_model_fields = sorted(
            key for key, value in asdict(config).items() if value != baseline["model"][key]
        )
        if changed_model_fields != ["encoder_layers"]:
            raise ValueError(f"encoder-depth {depth} changed unexpected model fields: {changed_model_fields}")

        model = IRRMCodecTransformer(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        tokens = torch.randint(5, len(AA_VOCAB), (2, 12))
        decoder_input = torch.randint(5, len(AA_VOCAB), (2, 13))
        outputs = model(tokens, tokens.ne(0), decoder_input)
        losses = IRRMCodecMultiTaskLoss(weights=MultiTaskLossWeights(1.0, 1.0, 1.0))(
            outputs,
            tcremp_target=torch.randn(2, embeddings.shape[1]),
            pgen_target=torch.randn(2),
            reconstruction_target=torch.randint(5, len(AA_VOCAB), (2, 13)),
            tcremp_mean=torch.zeros(embeddings.shape[1]),
            tcremp_std=torch.ones(embeddings.shape[1]),
            pgen_mean=torch.tensor(0.0),
            pgen_std=torch.tensor(1.0),
        )
        losses["loss"].backward()
        finite = bool(torch.isfinite(losses["loss"]))
        if not finite:
            raise FloatingPointError(f"encoder-depth {depth} smoke loss is non-finite.")
        results[str(depth)] = {
            "encoder_layers": depth,
            "decoder_layers": config.decoder_layers,
            "latent_dim": config.latent_dim,
            "parameter_count": parameter_count,
            "parameter_delta_vs_e4": parameter_count - baseline["parameter_count"],
            "changed_model_fields_vs_e4": changed_model_fields,
            "changed_training_fields_vs_e4": [],
            "finite_full_multitask_smoke_loss": finite,
            "smoke_loss": float(losses["loss"].detach()),
            "model_config": asdict(config),
        }
        del model, outputs, losses
        gc.collect()

    report = {
        "status": "ready",
        "interpretation": "RQ5 Stage 2 varies encoder_layers only; latent_dim=128 and decoder_layers=4 remain fixed.",
        "selection": {
            "criterion": "minimum best-checkpoint validation loss only",
            "validation_losses": validation_losses,
            "selected_latent_dim": 128,
            "test_metrics_used_for_selection": False,
        },
        "baseline": {
            "job_id": "1425949", "run_config_path": str(baseline_path.resolve()),
            "run_config_sha256": sha256(baseline_path), "latent_dim": 128,
            "encoder_layers": 4, "decoder_layers": 4,
            "parameter_count": baseline["parameter_count"], "non_depth_training_fields_match": True,
        },
        "benchmark": {
            "path": str(data_dir.resolve()), "ready_sha256": sha256(data_dir / "READY.json"),
            "rows": len(table), "embedding_shape": list(embeddings.shape),
            "split_rows": {key: len(value) for key, value in splits.items()},
            "normalizer_fit_split": "train", "normalizer_train_rows": standardizer.train_rows,
        },
        "candidates": results,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
