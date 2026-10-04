"""CPU preflight for the locked RQ4 WordPiece benchmark runs."""

import argparse
import hashlib
import json
import platform
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
    validation = json.loads((tokenizer_dir / "validation_report.json").read_text())
    if config["training_split"] != "train" or not validation["train_only_fit"]:
        raise ValueError("Tokenizer provenance does not prove train-only fitting.")
    if not all(validation["checks"].values()):
        raise ValueError("Tokenizer validation checks are not all true.")
    if sha256(tokenizer_path) != config["tokenizer_sha256"]:
        raise ValueError("Tokenizer checksum does not match immutable config.")
    train_manifest = data_dir / "manifests" / "train.tsv"
    if sha256(train_manifest) != config["train_manifest_sha256"]:
        raise ValueError("Locked train-manifest checksum changed.")

    ready = json.loads((data_dir / "READY.json").read_text())
    if ready.get("status") != "ready" or not all(ready.get("checks", {}).values()):
        raise ValueError("Locked benchmark is not ready.")
    table, embeddings = load_prepared_benchmark(data_dir)
    train_indices = select_split_indices(table, data_dir, "train")
    val_indices = select_split_indices(table, data_dir, "val")
    test_indices = select_split_indices(table, data_dir, "test")
    standardizer = TargetStandardizer.load(data_dir / "target_standardizer.npz")
    if standardizer.train_rows != len(train_indices) or standardizer.pgen_target != "log10_pgen_1mm":
        raise ValueError("Locked train-only target standardizer does not match RQ4.")
    if len(standardizer.tcremp_mean) != embeddings.shape[1]:
        raise ValueError("Locked TCRemP standardizer has the wrong dimension.")

    tokenizer = resolve_encoder_tokenizer("wordpiece", str(tokenizer_path))
    dataset = MultiTaskBenchmarkDataset(
        table, embeddings, train_indices[:2], tokenizer, "log10_pgen_1mm", 40
    )
    batch = collate_multitask([dataset[0], dataset[1]])
    torch.manual_seed(42)
    model_config = IRRMCodecConfig(
        input_vocab_size=tokenizer.vocab_size,
        output_vocab_size=len(AA_VOCAB),
        share_input_output_embeddings=False,
        max_sequence_len=40,
        d_model=32,
        latent_dim=24,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        ff_dim=64,
        dropout=0.0,
        tcremp_dim=embeddings.shape[1],
        tcremp_head_dim=24,
        pgen_head_dim=12,
        decoder_memory_tokens=2,
    )
    model = IRRMCodecTransformer(model_config)
    outputs = model(batch["encoder_tokens"], batch["encoder_mask"], batch["decoder_input"])
    standardizer_tensors = standardizer.as_torch(torch.device("cpu"))
    losses = IRRMCodecMultiTaskLoss(weights=MultiTaskLossWeights(1.0, 1.0, 1.0))(
        outputs,
        tcremp_target=batch["tcremp_target"],
        pgen_target=batch["pgen_target"],
        reconstruction_target=batch["reconstruction_target"],
        **standardizer_tensors,
    )
    losses["loss"].backward()
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError("WordPiece smoke loss is non-finite.")

    report = {
        "status": "ready",
        "tokenizer": {
            "path": str(tokenizer_path.resolve()),
            "sha256": sha256(tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
            "special_token_ids": config["special_token_ids"],
            "train_manifest_sha256": config["train_manifest_sha256"],
            "train_only_fit": True,
        },
        "benchmark": {
            "path": str(data_dir.resolve()),
            "ready_sha256": sha256(data_dir / "READY.json"),
            "rows": len(table),
            "embedding_shape": list(embeddings.shape),
            "split_rows": {
                "train": len(train_indices), "val": len(val_indices), "test": len(test_indices)
            },
            "normalizer_fit_split": "train",
            "normalizer_train_rows": standardizer.train_rows,
        },
        "model_smoke": {
            "input_vocab_size": tokenizer.vocab_size,
            "output_vocab_size": len(AA_VOCAB),
            "shared_input_output_embeddings": False,
            "finite_loss": True,
            "loss": float(losses["loss"].detach()),
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
