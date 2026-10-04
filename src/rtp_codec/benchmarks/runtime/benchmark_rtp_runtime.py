"""Measure frozen IRRM encoder runtime on the VDJdb uniform-100 cohort.

The measured interval starts after the checkpoint, tokenizer and cohort are in
memory and excludes warm-up.  It reports tokenization separately from the
sequence-to-latent encoder pass; the latter includes host-to-device transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtp_codec.data.multitask import resolve_encoder_tokenizer
from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer
from rtp_codec.tokenization.character import PAD_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_batch(encoded: list[list[int]], rows: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(encoded[row]) for row in rows)
    tokens = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    for local_index, row in enumerate(rows):
        ids = encoded[row]
        tokens[local_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    tokens = tokens.to(device, non_blocking=device.type == "cuda")
    return tokens, tokens.ne(PAD_ID)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sequence-column", default="cdr3")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-batches", type=int, default=3)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.batch_size < 1 or args.warmup_batches < 0:
        raise ValueError("batch size must be positive and warmup batches non-negative")

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    cohort = pd.read_csv(args.cohort, sep="\t")
    sequences = cohort[args.sequence_column].astype(str).tolist()
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Cohort has an empty sequence")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = RTPCodecConfig(**payload["model_config"])
    model = RTPCodecTransformer(model_config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(args.tokenizer))
    if tokenizer.vocab_size != model_config.input_vocab_size:
        raise ValueError("Tokenizer vocabulary does not match checkpoint")

    tokenization_started = time.perf_counter()
    encoded = [tokenizer.encode(sequence, model_config.max_sequence_len) for sequence in sequences]
    tokenization_seconds = time.perf_counter() - tokenization_started
    rows = [list(range(start, min(start + args.batch_size, len(encoded)))) for start in range(0, len(encoded), args.batch_size)]

    with torch.inference_mode():
        for batch_rows in rows[: args.warmup_batches]:
            tokens, mask = make_batch(encoded, batch_rows, device)
            latent = model.encode(tokens, mask)
            if latent.shape != (len(batch_rows), model_config.latent_dim):
                raise ValueError(f"Unexpected latent shape: {tuple(latent.shape)}")
        synchronize(device)

        encoded_rows = 0
        latent_sum = 0.0
        inference_started = time.perf_counter()
        for batch_rows in rows:
            tokens, mask = make_batch(encoded, batch_rows, device)
            latent = model.encode(tokens, mask)
            latent_sum += float(latent.sum())
            encoded_rows += len(batch_rows)
        synchronize(device)
        inference_seconds = time.perf_counter() - inference_started

    if encoded_rows != len(sequences) or not np.isfinite(latent_sum):
        raise RuntimeError("Encoder output validation failed")
    result = {
        "status": "complete",
        "cohort": {
            "path": str(args.cohort.resolve()),
            "sha256": sha256_file(args.cohort),
            "rows": len(sequences),
            "sequence_column": args.sequence_column,
        },
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint)},
        "tokenizer": {"path": str(args.tokenizer.resolve()), "sha256": sha256_file(args.tokenizer)},
        "model": asdict(model_config),
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "warmup_batches": args.warmup_batches,
            "tokenization_seconds": tokenization_seconds,
            "encoder_seconds": inference_seconds,
            "end_to_end_seconds": tokenization_seconds + inference_seconds,
            "encoder_sequences_per_second": len(sequences) / inference_seconds,
            "end_to_end_sequences_per_second": len(sequences) / (tokenization_seconds + inference_seconds),
            "ms_per_sequence_encoder": inference_seconds / len(sequences) * 1000,
            "ms_per_sequence_end_to_end": (tokenization_seconds + inference_seconds) / len(sequences) * 1000,
        },
        "validation": {"latent_dim": model_config.latent_dim, "latent_sum": latent_sum},
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    if device.type == "cuda":
        result["environment"]["cuda_device"] = torch.cuda.get_device_name(device)
        result["environment"]["cuda_runtime"] = torch.version.cuda
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["runtime"], sort_keys=True))


if __name__ == "__main__":
    main()
