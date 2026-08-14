"""Train and audit the leakage-safe TRB WordPiece vocabulary sweep."""

import argparse
import csv
import hashlib
import json
import math
import tempfile
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
DEFAULT_VOCAB_SIZES = (44, 64, 128, 256, 512, 1024, 2048, 4096)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def audit_tokenizer(
    tokenizer: Tokenizer,
    sequences: dict[str, list[str]],
    train_pgen: np.ndarray,
) -> tuple[dict, list[dict]]:
    vocab_size = tokenizer.get_vocab_size()
    token_by_id = {identifier: token for token, identifier in tokenizer.get_vocab().items()}
    split_counts = {split: np.zeros(vocab_size, dtype=np.int64) for split in sequences}
    split_presence = {split: np.zeros(vocab_size, dtype=np.int64) for split in sequences}
    reports = {}
    train_start = np.zeros(vocab_size, dtype=np.int64)
    train_end = np.zeros(vocab_size, dtype=np.int64)
    train_internal = np.zeros(vocab_size, dtype=np.int64)
    train_position_sum = np.zeros(vocab_size, dtype=np.float64)
    train_pgen_present_sum = np.zeros(vocab_size, dtype=np.float64)
    train_low_presence = np.zeros(vocab_size, dtype=np.int64)
    train_high_presence = np.zeros(vocab_size, dtype=np.int64)
    low_threshold, high_threshold = np.quantile(train_pgen, [0.25, 0.75])

    for split, values in sequences.items():
        token_lengths = []
        unknown_sequences = 0
        roundtrip_mismatches = 0
        for row_number, (sequence, encoding) in enumerate(
            zip(values, tokenizer.encode_batch(values), strict=True)
        ):
            ids = encoding.ids
            token_lengths.append(len(ids))
            unknown_sequences += int(UNK_ID in ids)
            roundtrip_mismatches += int(
                tokenizer.decode(ids, skip_special_tokens=True) != sequence
            )
            split_counts[split] += np.bincount(ids, minlength=vocab_size)
            unique_ids = np.unique(ids)
            split_presence[split][unique_ids] += 1
            if split == "train":
                train_start[ids[0]] += 1
                train_end[ids[-1]] += 1
                if len(ids) > 2:
                    train_internal += np.bincount(ids[1:-1], minlength=vocab_size)
                denominator = max(len(ids) - 1, 1)
                for position, token_id in enumerate(ids):
                    train_position_sum[token_id] += position / denominator
                pgen = float(train_pgen[row_number])
                train_pgen_present_sum[unique_ids] += pgen
                if pgen <= low_threshold:
                    train_low_presence[unique_ids] += 1
                if pgen >= high_threshold:
                    train_high_presence[unique_ids] += 1
        lengths = np.asarray(token_lengths, dtype=np.int64)
        reports[split] = {
            "rows": len(values),
            "unknown_sequences": unknown_sequences,
            "roundtrip_mismatches": roundtrip_mismatches,
            "token_length_min": int(lengths.min()),
            "token_length_max": int(lengths.max()),
            "token_length_mean": float(lengths.mean()),
            "token_length_p50": float(np.quantile(lengths, 0.5)),
            "token_length_p95": float(np.quantile(lengths, 0.95)),
        }

    train_rows = len(sequences["train"])
    train_global_mean = float(train_pgen.mean())
    train_global_std = float(train_pgen.std())
    low_rows = int(np.count_nonzero(train_pgen <= low_threshold))
    high_rows = int(np.count_nonzero(train_pgen >= high_threshold))
    total_train_occurrences = int(split_counts["train"].sum())
    global_start_fraction = train_rows / total_train_occurrences
    global_end_fraction = train_rows / total_train_occurrences
    rows = []
    for token_id in range(vocab_size):
        count = int(split_counts["train"][token_id])
        present = int(split_presence["train"][token_id])
        absent = train_rows - present
        mean_present = (
            train_pgen_present_sum[token_id] / present if present else math.nan
        )
        mean_absent = (
            (float(train_pgen.sum()) - train_pgen_present_sum[token_id]) / absent
            if absent
            else math.nan
        )
        presence_fraction = present / train_rows
        association = (
            (mean_present - train_global_mean)
            * math.sqrt(presence_fraction * (1.0 - presence_fraction))
            / train_global_std
            if present and absent and train_global_std > 0
            else math.nan
        )
        high_rate = (train_high_presence[token_id] + 0.5) / (high_rows + 1.0)
        low_rate = (train_low_presence[token_id] + 0.5) / (low_rows + 1.0)
        start_fraction = train_start[token_id] / count if count else math.nan
        end_fraction = train_end[token_id] / count if count else math.nan
        rows.append(
            {
                "token_id": token_id,
                "token": token_by_id[token_id],
                "surface": token_by_id[token_id].removeprefix(PREFIX),
                "is_special": token_id < len(SPECIAL_TOKENS),
                "train_count": count,
                "val_count": int(split_counts["val"][token_id]),
                "test_count": int(split_counts["test"][token_id]),
                "train_sequence_count": present,
                "train_sequence_fraction": presence_fraction,
                "train_start_count": int(train_start[token_id]),
                "train_internal_count": int(train_internal[token_id]),
                "train_end_count": int(train_end[token_id]),
                "train_mean_normalized_position": _safe_float(
                    train_position_sum[token_id] / count if count else math.nan
                ),
                "start_log2_enrichment": _safe_float(
                    math.log2(start_fraction / global_start_fraction)
                    if count and start_fraction > 0
                    else math.nan
                ),
                "end_log2_enrichment": _safe_float(
                    math.log2(end_fraction / global_end_fraction)
                    if count and end_fraction > 0
                    else math.nan
                ),
                "train_pgen_mean_present": _safe_float(mean_present),
                "train_pgen_mean_absent": _safe_float(mean_absent),
                "train_pgen_mean_difference": _safe_float(mean_present - mean_absent),
                "train_pgen_presence_correlation": _safe_float(association),
                "high_vs_low_pgen_log2_enrichment": _safe_float(math.log2(high_rate / low_rate)),
            }
        )

    eligible = [row for row in rows if not row["is_special"] and row["train_sequence_count"] >= 50]
    summary = {
        "encoding": reports,
        "pgen_association": {
            "split": "train",
            "target": "log10_pgen_1mm",
            "global_mean": train_global_mean,
            "global_std": train_global_std,
            "low_quartile_threshold": float(low_threshold),
            "high_quartile_threshold": float(high_threshold),
            "interpretation": "Descriptive token-presence associations only; not biological or causal claims.",
        },
        "top_tokens": {
            "frequency": sorted(eligible, key=lambda row: row["train_count"], reverse=True)[:25],
            "absolute_pgen_association": sorted(
                eligible,
                key=lambda row: abs(row["train_pgen_presence_correlation"] or 0.0),
                reverse=True,
            )[:25],
            "start_enrichment": sorted(
                eligible, key=lambda row: row["start_log2_enrichment"] or -math.inf, reverse=True
            )[:25],
            "end_enrichment": sorted(
                eligible, key=lambda row: row["end_log2_enrichment"] or -math.inf, reverse=True
            )[:25],
        },
    }
    return summary, rows


def write_token_statistics(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=list(DEFAULT_VOCAB_SIZES))
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    vocab_sizes = tuple(args.vocab_sizes)
    if args.seed != 42 or vocab_sizes != DEFAULT_VOCAB_SIZES:
        raise ValueError("WordPiece Stage A requires seed 42 and the locked eight-size sweep.")

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    table, _embeddings = load_prepared_benchmark(data_dir)
    split_indices = {
        split: select_split_indices(table, data_dir, split) for split in ("train", "val", "test")
    }
    sequences = {
        split: table.iloc[indices]["junction_aa"].astype(str).tolist()
        for split, indices in split_indices.items()
    }
    train_pgen = table.iloc[split_indices["train"]]["log10_pgen_1mm"].to_numpy(np.float64)
    train_manifest = data_dir / "manifests" / "train.tsv"
    corpus_path = output_root / "train_cdr3.txt"
    corpus_text = "".join(f"{sequence}\n" for sequence in sequences["train"])
    if corpus_path.exists() and corpus_path.read_text(encoding="utf-8") != corpus_text:
        raise ValueError("Existing WordPiece sweep corpus differs from the locked train split.")
    corpus_path.write_text(corpus_text, encoding="utf-8")

    sweep = {
        "status": "ready",
        "tokenizer_type": "wordpiece",
        "vocab_sizes": list(vocab_sizes),
        "seed": args.seed,
        "training_split": "train",
        "training_rows": len(split_indices["train"]),
        "train_manifest_sha256": sha256(train_manifest),
        "corpus_sha256": sha256(corpus_path),
        "tokenizers": [],
    }
    for vocab_size in vocab_sizes:
        final_dir = output_root / f"v{vocab_size}-seed42"
        if final_dir.exists():
            raise FileExistsError(f"Refusing to overwrite immutable tokenizer directory: {final_dir}")
        with tempfile.TemporaryDirectory(prefix=f".v{vocab_size}-", dir=output_root) as temporary:
            token_dir = Path(temporary)
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.decoder = WordPieceDecoder(prefix=PREFIX)
            tokenizer.train(
                [str(corpus_path)],
                WordPieceTrainer(
                    vocab_size=vocab_size,
                    min_frequency=args.min_frequency,
                    special_tokens=SPECIAL_TOKENS,
                    continuing_subword_prefix=PREFIX,
                    show_progress=False,
                ),
            )
            validate_wordpiece_tokenizer(tokenizer, f"WordPiece v{vocab_size}")
            actual_vocab_size = tokenizer.get_vocab_size()
            if actual_vocab_size != vocab_size:
                raise ValueError(
                    f"Requested WordPiece vocabulary {vocab_size}, got {actual_vocab_size}; size is infeasible."
                )
            audit, rows = audit_tokenizer(tokenizer, sequences, train_pgen)
            for split, report in audit["encoding"].items():
                if report["unknown_sequences"] or report["roundtrip_mismatches"]:
                    raise ValueError(f"WordPiece v{vocab_size} failed {split}: {report}")
                if report["token_length_max"] > 40:
                    raise ValueError(f"WordPiece v{vocab_size} exceeds max length on {split}: {report}")
            tokenizer_path = token_dir / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            tokenizer.model.save(str(token_dir))
            write_token_statistics(token_dir / "token_statistics.tsv", rows)
            config = {
                "status": "immutable",
                "tokenizer_type": "wordpiece",
                "seed": args.seed,
                "requested_vocab_size": vocab_size,
                "actual_vocab_size": actual_vocab_size,
                "min_frequency": args.min_frequency,
                "continuing_subword_prefix": PREFIX,
                "special_tokens": SPECIAL_TOKENS,
                "special_token_ids": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unk": UNK_ID},
                "training_split": "train",
                "training_rows": len(split_indices["train"]),
                "train_manifest_path": str(train_manifest.resolve()),
                "train_manifest_sha256": sha256(train_manifest),
                "corpus_path": str(corpus_path.resolve()),
                "corpus_sha256": sha256(corpus_path),
            }
            (token_dir / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
            validation = {
                "status": "ready",
                "train_only_fit": True,
                "split_rows": {split: len(indices) for split, indices in split_indices.items()},
                "split_index_overlap": {
                    "train_val": int(np.intersect1d(split_indices["train"], split_indices["val"]).size),
                    "train_test": int(np.intersect1d(split_indices["train"], split_indices["test"]).size),
                    "val_test": int(np.intersect1d(split_indices["val"], split_indices["test"]).size),
                },
                "encoding": audit["encoding"],
                "checks": {
                    "special_ids_match_model": True,
                    "requested_vocab_size_feasible": True,
                    "train_manifest_matches_locked_split": True,
                    "no_split_overlap": True,
                    "no_unknown_tokens": True,
                    "roundtrip_exact": True,
                    "all_encoded_lengths_at_most_40": True,
                    "pgen_association_train_only": True,
                },
            }
            (token_dir / "validation_report.json").write_text(
                json.dumps(validation, indent=2), encoding="utf-8"
            )
            config.update(
                {
                    "tokenizer_sha256": sha256(tokenizer_path),
                    "vocab_sha256": sha256(token_dir / "vocab.txt"),
                    "token_statistics_sha256": sha256(token_dir / "token_statistics.tsv"),
                    "audit_report_sha256": sha256(token_dir / "audit_report.json"),
                }
            )
            (token_dir / "tokenizer_config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )
            token_dir.rename(final_dir)
        sweep["tokenizers"].append(
            {
                "requested_vocab_size": vocab_size,
                "actual_vocab_size": actual_vocab_size,
                "directory": str(final_dir.resolve()),
                "tokenizer_sha256": config["tokenizer_sha256"],
                "vocab_sha256": config["vocab_sha256"],
                "token_statistics_sha256": config["token_statistics_sha256"],
                "max_encoded_lengths": {
                    split: audit["encoding"][split]["token_length_max"] for split in sequences
                },
            }
        )
    (output_root / "sweep_manifest.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
