"""CPU gate for the leakage-safe TRB WordPiece vocabulary sweep."""

import argparse
import gc
import hashlib
import json
import platform
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
from irrm_codec.tokenization import AA_VOCAB


VOCAB_SIZES = (44, 64, 128, 256, 512, 1024, 2048, 4096)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-root", required=True)
    parser.add_argument("--baseline-run-config", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer_root = Path(args.tokenizer_root)
    baseline_path = Path(args.baseline_run_config)
    baseline = json.loads(baseline_path.read_text())
    if (
        baseline["model"]["latent_dim"] != 128
        or baseline["model"]["encoder_layers"] != 4
        or baseline["model"]["decoder_layers"] != 4
    ):
        raise ValueError("Baseline is not the validation-selected char d128 4+4 model.")

    ready = json.loads((data_dir / "READY.json").read_text())
    if ready.get("status") != "ready" or not all(ready.get("checks", {}).values()):
        raise ValueError("Locked benchmark is not ready.")
    table, embeddings = load_prepared_benchmark(data_dir)
    split_indices = {
        split: select_split_indices(table, data_dir, split) for split in ("train", "val", "test")
    }
    standardizer = TargetStandardizer.load(data_dir / "target_standardizer.npz")
    if standardizer.train_rows != len(split_indices["train"]):
        raise ValueError("Locked standardizer is not train-only for the locked manifest.")
    if standardizer.pgen_target != "log10_pgen_1mm":
        raise ValueError("Locked standardizer uses the wrong pgen target.")

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
        raise ValueError(f"Selected char baseline differs from locked Stage-A settings: {mismatches}")

    train_manifest = data_dir / "manifests" / "train.tsv"
    standardizer_tensors = standardizer.as_torch(torch.device("cpu"))
    candidates = {}
    for vocab_size in VOCAB_SIZES:
        token_dir = tokenizer_root / f"v{vocab_size}-seed42"
        config = json.loads((token_dir / "tokenizer_config.json").read_text())
        validation = json.loads((token_dir / "validation_report.json").read_text())
        tokenizer_path = token_dir / "tokenizer.json"
        if config["actual_vocab_size"] != vocab_size or config["requested_vocab_size"] != vocab_size:
            raise ValueError(f"WordPiece v{vocab_size} actual/requested size mismatch.")
        if config["training_split"] != "train" or config["training_rows"] != 79_544:
            raise ValueError(f"WordPiece v{vocab_size} lacks train-only provenance.")
        if config["train_manifest_sha256"] != sha256(train_manifest):
            raise ValueError(f"WordPiece v{vocab_size} train manifest checksum changed.")
        if config["tokenizer_sha256"] != sha256(tokenizer_path):
            raise ValueError(f"WordPiece v{vocab_size} tokenizer checksum changed.")
        if not all(validation["checks"].values()):
            raise ValueError(f"WordPiece v{vocab_size} validation checks are not all true.")

        tokenizer = resolve_encoder_tokenizer("wordpiece", str(tokenizer_path))
        dataset = MultiTaskBenchmarkDataset(
            table, embeddings, split_indices["train"][:2], tokenizer, "log10_pgen_1mm", 40
        )
        batch = collate_multitask([dataset[0], dataset[1]])
        model_payload = dict(baseline["model"])
        model_payload["input_vocab_size"] = vocab_size
        model_payload["output_vocab_size"] = len(AA_VOCAB)
        model_payload["share_input_output_embeddings"] = False
        model_config = IRRMCodecConfig(**model_payload)
        model_config.validate()
        changed_model_fields = sorted(
            key for key, value in asdict(model_config).items() if value != baseline["model"][key]
        )
        if changed_model_fields != ["input_vocab_size", "share_input_output_embeddings"]:
            raise ValueError(f"WordPiece v{vocab_size} changed unexpected model fields: {changed_model_fields}")
        torch.manual_seed(42)
        model = IRRMCodecTransformer(model_config)
        outputs = model(batch["encoder_tokens"], batch["encoder_mask"], batch["decoder_input"])
        losses = IRRMCodecMultiTaskLoss(weights=MultiTaskLossWeights(1.0, 1.0, 1.0))(
            outputs,
            tcremp_target=batch["tcremp_target"],
            pgen_target=batch["pgen_target"],
            reconstruction_target=batch["reconstruction_target"],
            **standardizer_tensors,
        )
        losses["loss"].backward()
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"WordPiece v{vocab_size} smoke loss is non-finite.")
        candidates[str(vocab_size)] = {
            "requested_vocab_size": vocab_size,
            "actual_vocab_size": tokenizer.vocab_size,
            "tokenizer_sha256": config["tokenizer_sha256"],
            "vocab_sha256": config["vocab_sha256"],
            "token_statistics_sha256": config["token_statistics_sha256"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_delta_vs_char": sum(parameter.numel() for parameter in model.parameters())
            - baseline["parameter_count"],
            "changed_model_fields_vs_char": changed_model_fields,
            "changed_training_fields_vs_char": ["tokenizer_type", "tokenizer_path"],
            "separate_wordpiece_encoder_char_decoder": True,
            "finite_full_multitask_smoke_loss": True,
            "smoke_loss": float(losses["loss"].detach()),
        }
        del model, outputs, losses, dataset, batch
        gc.collect()

    report = {
        "status": "ready",
        "selection_rule": {
            "primary": "minimum best-checkpoint full validation objective",
            "secondary": ["validation pgen RMSE", "validation reconstruction", "validation TCRemP"],
            "test_metrics_used_for_selection": False,
        },
        "baseline": {
            "job_id": "1425949",
            "run_config_sha256": sha256(baseline_path),
            "parameter_count": baseline["parameter_count"],
            "non_tokenizer_training_fields_match": True,
        },
        "benchmark": {
            "path": str(data_dir.resolve()),
            "ready_sha256": sha256(data_dir / "READY.json"),
            "rows": len(table),
            "embedding_shape": list(embeddings.shape),
            "split_rows": {split: len(indices) for split, indices in split_indices.items()},
            "normalizer_fit_split": "train",
            "normalizer_train_rows": standardizer.train_rows,
        },
        "candidates": candidates,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
