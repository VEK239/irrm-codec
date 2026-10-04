"""Extract one frozen IRRM-CODEC latent matrix for an ordered VDJdb cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irrm_codec.multitask_data import resolve_encoder_tokenizer
from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer
from irrm_codec.tokenization import PAD_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    cohort = pd.read_csv(args.cohort, sep="\t")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = IRRMCodecConfig(**checkpoint["model_config"])
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(args.tokenizer))
    if tokenizer.vocab_size != config.input_vocab_size:
        raise ValueError("Tokenizer vocabulary differs from checkpoint architecture")
    encoded = [tokenizer.encode(sequence, config.max_sequence_len)
               for sequence in cohort["cdr3"].astype(str)]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = IRRMCodecTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    matrix = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float32,
        shape=(len(cohort), config.latent_dim),
    )
    with torch.inference_mode():
        for start in range(0, len(encoded), args.batch_size):
            rows = encoded[start:start + args.batch_size]
            width = max(map(len, rows))
            tokens = torch.full((len(rows), width), PAD_ID, dtype=torch.long, device=device)
            for index, ids in enumerate(rows):
                tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            matrix[start:start + len(rows)] = model.encode(tokens, tokens.ne(PAD_ID)).float().cpu().numpy()
    matrix.flush()
    if not np.isfinite(matrix).all():
        raise ValueError("Extracted latent contains non-finite values")
    metadata = {
        "status": "complete", "rows": len(cohort), "shape": list(matrix.shape),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "model_config": asdict(config), "cohort_sha256": sha256(args.cohort),
        "latent_sha256": sha256(args.output), "ordering_preserved": True,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
