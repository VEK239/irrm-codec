"""CPU gate for the three leakage-safe anchored tokenizers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from rtp_codec.tokenization.anchored import AnchoredTokenizer
from rtp_codec.data.multitask import (
    MultiTaskBenchmarkDataset,
    TargetStandardizer,
    collate_multitask,
    load_prepared_benchmark,
    resolve_encoder_tokenizer,
    select_split_indices,
)
from rtp_codec.training.objectives import RTPCodecMultiTaskLoss, MultiTaskLossWeights
from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer
from rtp_codec.tokenization.character import AA_VOCAB


CANDIDATES = ("edge_k", "data_anchor", "germline_anchor")


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
    root = Path(args.tokenizer_root)
    baseline_path = Path(args.baseline_run_config)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        baseline["model"]["latent_dim"] != 128
        or baseline["model"]["encoder_layers"] != 4
        or baseline["model"]["decoder_layers"] != 4
    ):
        raise ValueError("Baseline is not the validation-selected char d128 4+4 model.")
    ready = json.loads((data_dir / "READY.json").read_text(encoding="utf-8"))
    if ready.get("status") != "ready" or not all(ready.get("checks", {}).values()):
        raise ValueError("Locked benchmark is not ready.")
    preparation = json.loads((root / "preparation_manifest.json").read_text(encoding="utf-8"))
    if preparation["external_corpus"]["overlap_after_filter"] != 0:
        raise ValueError("External tokenizer corpus overlaps the locked benchmark.")
    if preparation["inference_modality"] != "sequence_only":
        raise ValueError("Anchored study is not sequence-only.")

    table, embeddings = load_prepared_benchmark(data_dir)
    split_indices = {split: select_split_indices(table, data_dir, split) for split in ("train", "val", "test")}
    standardizer = TargetStandardizer.load(data_dir / "target_standardizer.npz")
    if standardizer.train_rows != len(split_indices["train"]) or standardizer.pgen_target != "log10_pgen_1mm":
        raise ValueError("Locked train-only normalizer does not match the benchmark.")

    expected_training = {
        "train_subset":"all","pgen_target":"log10_pgen_1mm","tokenizer_type":"char",
        "tokenizer_path":None,"max_sequence_len":40,"d_model":320,"latent_dim":128,"nhead":8,
        "encoder_layers":4,"decoder_layers":4,"ff_dim":1280,"dropout":0.1,
        "tcremp_head_dim":1024,"pgen_head_dim":256,"decoder_memory_tokens":4,
        "tcremp_loss_weight":1.0,"pgen_loss_weight":1.0,"reconstruction_loss_weight":1.0,
        "tcremp_mse_fraction":0.7,"pgen_huber_delta":0.5,"label_smoothing":0.0,
        "batch_size":64,"epochs":40,"lr":0.0003,"weight_decay":0.0001,
        "gradient_accumulation_steps":1,"max_grad_norm":1.0,"early_stopping_patience":0,
        "scheduler_factor":0.5,"scheduler_patience":2,"scheduler_min_lr":1e-6,
        "seed":42,"num_workers":8,"amp":True,"skip_generation_metrics":False,
        "save_test_predictions":False,"val_generation_every":0,"max_train_batches":0,
        "max_eval_batches":0,
    }
    mismatches = {
        key: {"expected": value, "actual": baseline["training"].get(key)}
        for key, value in expected_training.items() if baseline["training"].get(key) != value
    }
    if mismatches:
        raise ValueError(f"Selected baseline differs from locked settings: {mismatches}")

    standardizer_tensors = standardizer.as_torch(torch.device("cpu"))
    candidate_reports = {}
    for kind in CANDIDATES:
        directory = root / kind
        bundle_path = directory / "anchored_tokenizer.json"
        config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
        validation = json.loads((directory / "validation_report.json").read_text(encoding="utf-8"))
        if config["bundle_sha256"] != sha256(bundle_path):
            raise ValueError(f"{kind} bundle checksum changed.")
        if config["inference_modality"] != "sequence_only" or config["accepts_v_call_or_j_call"]:
            raise ValueError(f"{kind} violates sequence-only inference.")
        if not all(validation["checks"].values()):
            raise ValueError(f"{kind} preparation checks are not all accepting.")
        runtime = AnchoredTokenizer(bundle_path)
        roundtrip_rows = 0
        max_encoded = 0
        for split, indices in split_indices.items():
            for sequence in table.iloc[indices]["junction_aa"].astype(str):
                encoded = runtime.encode_with_boundaries(sequence, 40)
                if runtime.decode(encoded.ids) != sequence:
                    raise ValueError(f"{kind} roundtrip failed in {split} for {sequence!r}.")
                if encoded.n_anchor != sequence[:len(encoded.n_anchor)] or encoded.c_anchor != sequence[-len(encoded.c_anchor):]:
                    raise ValueError(f"{kind} protected token does not match a literal edge.")
                if encoded.boundary_types[0] != "N_ANCHOR" or encoded.boundary_types[-1] != "C_ANCHOR":
                    raise ValueError(f"{kind} boundary typing failed.")
                max_encoded = max(max_encoded, len(encoded.ids))
                roundtrip_rows += 1
        if max_encoded > 40:
            raise ValueError(f"{kind} exceeds max input length.")

        tokenizer = resolve_encoder_tokenizer(kind, str(bundle_path))
        dataset = MultiTaskBenchmarkDataset(
            table, embeddings, split_indices["train"][:2], tokenizer, "log10_pgen_1mm", 40
        )
        batch = collate_multitask([dataset[0], dataset[1]])
        model_payload = dict(baseline["model"])
        model_payload["input_vocab_size"] = tokenizer.vocab_size
        model_payload["output_vocab_size"] = len(AA_VOCAB)
        model_payload["share_input_output_embeddings"] = False
        model_config = RTPCodecConfig(**model_payload)
        model_config.validate()
        changed_model_fields = sorted(
            key for key, value in asdict(model_config).items() if value != baseline["model"][key]
        )
        if changed_model_fields != ["input_vocab_size", "share_input_output_embeddings"]:
            raise ValueError(f"{kind} changed unexpected model fields: {changed_model_fields}")
        torch.manual_seed(42)
        model = RTPCodecTransformer(model_config)
        outputs = model(batch["encoder_tokens"], batch["encoder_mask"], batch["decoder_input"])
        losses = RTPCodecMultiTaskLoss(weights=MultiTaskLossWeights(1.0, 1.0, 1.0))(
            outputs,
            tcremp_target=batch["tcremp_target"],
            pgen_target=batch["pgen_target"],
            reconstruction_target=batch["reconstruction_target"],
            **standardizer_tensors,
        )
        losses["loss"].backward()
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"{kind} finite smoke gate failed.")
        candidate_reports[kind] = {
            "tokenizer_type": kind,
            "bundle_path": str(bundle_path),
            "bundle_sha256": sha256(bundle_path),
            "input_vocab_size": tokenizer.vocab_size,
            "roundtrip_rows": roundtrip_rows,
            "max_encoded_length": max_encoded,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_delta_vs_char": sum(parameter.numel() for parameter in model.parameters()) - baseline["parameter_count"],
            "changed_model_fields_vs_char": changed_model_fields,
            "changed_training_fields_vs_char": ["tokenizer_type", "tokenizer_path"],
            "finite_full_multitask_smoke_loss": True,
            "smoke_loss": float(losses["loss"].detach()),
        }
        del runtime, tokenizer, dataset, batch, model, outputs, losses
        gc.collect()

    report = {
        "status": "ready",
        "selection_rule": {
            "primary": "minimum best-checkpoint full validation objective",
            "secondary": ["validation pgen RMSE", "validation reconstruction", "validation TCRemP"],
            "test_metrics_used_for_selection": False,
        },
        "study_constraints": {
            "sequence_only_inference": True,
            "v_j_calls_are_model_inputs": False,
            "char_decoder_vocab_size": len(AA_VOCAB),
            "configs_differ_only_in_tokenizer_and_implied_input_embedding": True,
        },
        "baseline": {
            "job_id": "1425949",
            "run_config_sha256": sha256(baseline_path),
            "parameter_count": baseline["parameter_count"],
        },
        "benchmark": {
            "rows": len(table),
            "split_rows": {split: len(value) for split, value in split_indices.items()},
            "normalizer_fit_split": "train",
            "normalizer_train_rows": standardizer.train_rows,
        },
        "candidates": candidate_reports,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
