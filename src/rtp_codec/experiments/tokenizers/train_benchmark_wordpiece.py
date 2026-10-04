"""Train leakage-free WordPiece tokenizers on a prepared benchmark subset."""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from rtp_codec.data.multitask import load_prepared_benchmark, select_split_indices
from rtp_codec.utils import setup_logging


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PREFIX = "##"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/benchmark/trb")
    parser.add_argument("--output-dir", default="artifacts/tokenizers/trb")
    parser.add_argument("--train-subset", choices=["1k", "10k", "all"], default="all")
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser.parse_args()


def train_tokenizer(corpus_path: Path, vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder(prefix=PREFIX)
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix=PREFIX,
    )
    tokenizer.train([str(corpus_path)], trainer)
    return tokenizer


def main():
    args = parse_args()
    if args.min_frequency < 1:
        raise ValueError("--min-frequency must be positive.")
    if any(size < len(SPECIAL_TOKENS) for size in args.vocab_sizes):
        raise ValueError("Every vocabulary must be larger than the special-token set.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "train_wordpiece.log")
    table, _embeddings = load_prepared_benchmark(args.data_dir)
    train_indices = select_split_indices(
        table,
        args.data_dir,
        "train",
        train_subset=args.train_subset,
    )
    sequences = table.iloc[train_indices]["junction_aa"].astype(str).tolist()
    corpus_path = output_dir / f"train_{args.train_subset}.txt"
    corpus_path.write_text("".join(f"{sequence}\n" for sequence in sequences), encoding="utf-8")
    corpus_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    summary = {}
    for requested_size in args.vocab_sizes:
        tokenizer = train_tokenizer(corpus_path, requested_size, args.min_frequency)
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            actual_id = tokenizer.token_to_id(token)
            if actual_id != expected_id:
                raise ValueError(f"{token} has id={actual_id}, expected {expected_id}.")

        mismatches = 0
        for sequence in sequences[:1000]:
            decoded = tokenizer.decode(tokenizer.encode(sequence).ids, skip_special_tokens=True)
            mismatches += int(decoded != sequence)
        if mismatches:
            raise ValueError(
                f"Tokenizer vocab={requested_size} failed {mismatches}/1000 round trips."
            )

        vocab_dir = output_dir / f"wordpiece_vocab_{requested_size}"
        vocab_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = vocab_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        tokenizer.model.save(str(vocab_dir))
        actual_size = tokenizer.get_vocab_size()
        summary[str(requested_size)] = {
            "requested_vocab_size": requested_size,
            "actual_vocab_size": actual_size,
            "min_frequency": args.min_frequency,
            "tokenizer_path": str(tokenizer_path),
            "train_subset": args.train_subset,
            "train_rows": len(train_indices),
            "corpus_sha256": corpus_sha256,
            "roundtrip_mismatches_first_1000": mismatches,
        }
        logger.info(
            "saved tokenizer requested_vocab=%d actual_vocab=%d path=%s",
            requested_size,
            actual_size,
            tokenizer_path,
        )

    (output_dir / "wordpiece_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
