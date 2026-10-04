"""Load a training checkpoint and encode ordered CDR3 sequences."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import pandas as pd
import torch
from rtp_codec.data.multitask import resolve_encoder_tokenizer
from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer
from rtp_codec.tokenization.character import PAD_ID

class SequenceEncoder:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    def encode_sequences(self, sequences: Sequence[str], batch_size: int = 512) -> np.ndarray:
        """Return float32 vectors in input order; reject invalid or overlong CDR3s."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        output = np.empty((len(sequences), self.model.config.latent_dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                rows = [self.tokenizer.encode(s, self.model.config.max_sequence_len)
                        for s in sequences[start:start + batch_size]]
                width = max(map(len, rows))
                tokens = torch.full((len(rows), width), PAD_ID, dtype=torch.long, device=self.device)
                for index, ids in enumerate(rows):
                    tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                output[start:start + len(rows)] = self.model.encode(tokens, tokens.ne(PAD_ID)).float().cpu().numpy()
        if not np.isfinite(output).all():
            raise ValueError("Encoder produced non-finite vectors.")
        return output

def load_encoder(checkpoint_path: str | Path, *, tokenizer_path: str | Path | None = None,
                 device: str = "cpu") -> SequenceEncoder:
    """Load the architecture and tokenizer saved by the multi-objective trainer."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = RTPCodecConfig(**checkpoint["model_config"])
    metadata = checkpoint["tokenizer"]
    kind = metadata["type"]
    bundle = tokenizer_path if tokenizer_path is not None else metadata.get("path")
    if bundle is not None:
        bundle = Path(bundle)
        if not bundle.is_absolute() and not bundle.is_file():
            bundle = path.parent / bundle
        if not bundle.is_file():
            raise FileNotFoundError(f"Tokenizer bundle not found: {bundle}. Supply tokenizer_path/--tokenizer.")
    tokenizer = resolve_encoder_tokenizer(kind, None if bundle is None else str(bundle))
    if tokenizer.vocab_size != config.input_vocab_size:
        raise ValueError("Tokenizer vocabulary differs from the checkpoint architecture.")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    model = RTPCodecTransformer(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(target)
    return SequenceEncoder(model, tokenizer, target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="Override the saved tokenizer bundle location")
    parser.add_argument("--input", type=Path, required=True, help="TSV table with CDR3 amino-acid sequences")
    parser.add_argument("--sequence-column", default="junction_aa")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy matrix in input row order")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("--output must have a .npy extension")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    table = pd.read_csv(args.input, sep="\t")
    if args.sequence_column not in table:
        raise ValueError(f"Input is missing column {args.sequence_column!r}.")
    if table[args.sequence_column].isna().any():
        raise ValueError("Input contains missing CDR3 sequences.")
    encoder = load_encoder(args.checkpoint, tokenizer_path=args.tokenizer, device=args.device)
    matrix = encoder.encode_sequences(table[args.sequence_column].tolist(), args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, matrix)
    args.output.with_suffix(".json").write_text(json.dumps({
        "input": str(args.input), "sequence_column": args.sequence_column,
        "checkpoint": str(args.checkpoint), "tokenizer_type": encoder.tokenizer.name,
        "shape": list(matrix.shape), "dtype": str(matrix.dtype), "ordering_preserved": True,
    }, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
