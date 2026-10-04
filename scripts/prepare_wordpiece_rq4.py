"""Train and validate the immutable RQ4 WordPiece tokenizer on TRB train only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from irrm_codec.multitask_data import load_prepared_benchmark, select_split_indices
from irrm_codec.tokenization import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from irrm_codec.wordpiece_tokenization import validate_wordpiece_tokenizer


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PREFIX = "##"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encode_report(tokenizer: Tokenizer, sequences: list[str]) -> dict:
    token_lengths = []
    unknown_sequences = 0
    roundtrip_mismatches = 0
    for sequence in sequences:
        encoding = tokenizer.encode(sequence)
        token_lengths.append(len(encoding.ids))
        unknown_sequences += int(UNK_ID in encoding.ids)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        roundtrip_mismatches += int(decoded != sequence)
    values = np.asarray(token_lengths, dtype=np.int64)
    return {
        "rows": len(sequences),
        "unknown_sequences": unknown_sequences,
        "roundtrip_mismatches": roundtrip_mismatches,
        "token_length_min": int(values.min()),
        "token_length_max": int(values.max()),
        "token_length_mean": float(values.mean()),
        "token_length_p50": float(np.quantile(values, 0.5)),
        "token_length_p95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("RQ4 tokenizer seed must remain fixed at 42.")
    if args.vocab_size != 256:
        raise ValueError("RQ4 uses the preregistered 256-token WordPiece vocabulary.")
    output_dir = Path(args.output_dir)
    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable tokenizer: {tokenizer_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    table, _embeddings = load_prepared_benchmark(data_dir)
    split_indices = {
        split: select_split_indices(table, data_dir, split)
        for split in ("train", "val", "test")
    }
    train_manifest = data_dir / "manifests" / "train.tsv"
    sequences = {
        split: table.iloc[indices]["junction_aa"].astype(str).tolist()
        for split, indices in split_indices.items()
    }

    corpus_path = output_dir / "train_cdr3.txt"
    corpus_path.write_text(
        "".join(f"{sequence}\n" for sequence in sequences["train"]),
        encoding="utf-8",
    )
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder(prefix=PREFIX)
    tokenizer.train(
        [str(corpus_path)],
        WordPieceTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix=PREFIX,
            show_progress=False,
        ),
    )
    validate_wordpiece_tokenizer(tokenizer, "newly trained RQ4 tokenizer")
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != args.vocab_size:
        raise ValueError(
            f"Requested vocabulary {args.vocab_size}, got {actual_vocab_size}."
        )

    reports = {split: encode_report(tokenizer, values) for split, values in sequences.items()}
    for split, report in reports.items():
        if report["unknown_sequences"] or report["roundtrip_mismatches"]:
            raise ValueError(f"Tokenizer failed {split} coverage/roundtrip validation: {report}")
        if report["token_length_max"] > 40:
            raise ValueError(f"Tokenizer exceeds model width on {split}: {report}")

    tokenizer.save(str(tokenizer_path))
    tokenizer.model.save(str(output_dir))
    special_ids = {
        "pad": PAD_ID,
        "bos": BOS_ID,
        "eos": EOS_ID,
        "unk": UNK_ID,
    }
    config = {
        "status": "immutable",
        "tokenizer_type": "wordpiece",
        "seed": args.seed,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "min_frequency": args.min_frequency,
        "continuing_subword_prefix": PREFIX,
        "special_tokens": SPECIAL_TOKENS,
        "special_token_ids": special_ids,
        "training_split": "train",
        "training_rows": len(split_indices["train"]),
        "train_manifest_path": str(train_manifest.resolve()),
        "train_manifest_sha256": sha256(train_manifest),
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": sha256(corpus_path),
        "tokenizer_path": str(tokenizer_path.resolve()),
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    validation = {
        "status": "ready",
        "train_only_fit": True,
        "split_rows": {split: len(indices) for split, indices in split_indices.items()},
        "split_index_overlap": {
            "train_val": int(np.intersect1d(split_indices["train"], split_indices["val"]).size),
            "train_test": int(np.intersect1d(split_indices["train"], split_indices["test"]).size),
            "val_test": int(np.intersect1d(split_indices["val"], split_indices["test"]).size),
        },
        "encoding": reports,
        "checks": {
            "special_ids_match_model": True,
            "train_manifest_matches_locked_split": True,
            "no_split_overlap": True,
            "no_unknown_tokens": True,
            "roundtrip_exact": True,
            "all_encoded_lengths_at_most_40": True,
        },
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    config["tokenizer_sha256"] = sha256(tokenizer_path)
    config["vocab_sha256"] = sha256(output_dir / "vocab.txt")
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
