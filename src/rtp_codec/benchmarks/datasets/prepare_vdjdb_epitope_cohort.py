"""Prepare an auditable, leakage-controlled VDJdb TRB cohort.

This command intentionally stops before full embedding extraction or evaluation.
It filters and hashes the source, removes exact benchmark overlaps, locks an
objective label rule, and smoke-loads one frozen encoder checkpoint on CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
MISSING = frozenset({"", "NA", "N/A", "NONE", "NULL", "UNKNOWN"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vdjdb-tsv", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--min-donors", type=int, default=2)
    parser.add_argument("--sensitivity-thresholds", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--min-vdjdb-score", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalized_context(value: object) -> str:
    text = clean_text(value).upper()
    return "" if text in MISSING else text


def parse_json_object(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def normalize_vdjdb_row(row: dict[str, str], min_score: float) -> tuple[dict | None, str]:
    if normalized_context(row.get("gene")) != "TRB":
        return None, "not_trb"
    if normalized_context(row.get("species")) not in {"HOMOSAPIENS", "HOMO SAPIENS"}:
        return None, "not_human"

    sequence = normalized_context(row.get("cdr3"))
    if not sequence or len(sequence) > 40 or set(sequence).difference(VALID_AA):
        return None, "invalid_cdr3"

    fix = parse_json_object(row.get("cdr3fix", ""))
    if fix.get("good") is not True:
        return None, "cdr3fix_not_good"
    fixed_sequence = normalized_context(fix.get("cdr3"))
    if fixed_sequence and fixed_sequence != sequence:
        return None, "cdr3fix_changes_sequence"

    epitope = normalized_context(row.get("antigen.epitope"))
    mhc_a = normalized_context(row.get("mhc.a"))
    mhc_b = normalized_context(row.get("mhc.b"))
    mhc_class = normalized_context(row.get("mhc.class"))
    if not epitope or not mhc_a or not mhc_class:
        return None, "missing_epitope_or_mhc"

    try:
        score = float(row.get("vdjdb.score", "nan"))
    except ValueError:
        score = math.nan
    if not math.isfinite(score) or score < min_score:
        return None, "low_or_invalid_score"

    meta = parse_json_object(row.get("meta", ""))
    subject = clean_text(meta.get("subject.id") or meta.get("subject_id"))
    reference = clean_text(row.get("reference.id"))
    study = clean_text(meta.get("study.id") or meta.get("study_id") or reference)
    if not subject:
        return None, "missing_donor"
    donor_key = f"{study}|{subject}"
    label = "|".join((epitope, mhc_a, mhc_b or "-", mhc_class))
    return {
        "cdr3": sequence,
        "label": label,
        "epitope": epitope,
        "mhc_a": mhc_a,
        "mhc_b": mhc_b,
        "mhc_class": mhc_class,
        "donor_key": donor_key,
        "reference_id": reference,
        "v_call": normalized_context(row.get("v.segm")),
        "j_call": normalized_context(row.get("j.segm")),
        "vdjdb_score": score,
        "length": len(sequence),
    }, "accepted_row"


def read_benchmark_sequences(benchmark_dir: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    split_sequences: dict[str, set[str]] = {}
    hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = benchmark_dir / "manifests" / f"{split}.tsv"
        hashes[str(path)] = sha256_file(path)
        sequences: set[str] = set()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "junction_aa" not in (reader.fieldnames or []):
                raise ValueError(f"Missing junction_aa in {path}")
            for row in reader:
                sequences.add(normalized_context(row["junction_aa"]))
        split_sequences[split] = sequences
    return split_sequences, hashes


def prepare_records(
    vdjdb_tsv: Path,
    benchmark_sequences: dict[str, set[str]],
    min_score: float,
) -> tuple[list[dict], dict]:
    counts: Counter[str] = Counter()
    by_sequence: dict[str, list[dict]] = defaultdict(list)
    with vdjdb_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "gene", "cdr3", "species", "mhc.a", "mhc.b", "mhc.class",
            "antigen.epitope", "reference.id", "vdjdb.score", "meta", "cdr3fix",
            "v.segm", "j.segm",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"VDJdb source is missing columns: {sorted(missing)}")
        for row in reader:
            counts["source_rows"] += 1
            normalized, reason = normalize_vdjdb_row(row, min_score=min_score)
            counts[reason] += 1
            if normalized is not None:
                by_sequence[normalized["cdr3"]].append(normalized)

    retained: list[dict] = []
    overlap_by_split: Counter[str] = Counter()
    for sequence, rows in sorted(by_sequence.items()):
        labels = {row["label"] for row in rows}
        if len(labels) != 1:
            counts["ambiguous_multilabel_clonotypes"] += 1
            continue
        donors = {row["donor_key"] for row in rows}
        if len(donors) != 1:
            counts["multi_donor_clonotypes"] += 1
            continue
        overlapping_splits = [
            split for split, sequences in benchmark_sequences.items() if sequence in sequences
        ]
        if overlapping_splits:
            counts["benchmark_overlap_clonotypes"] += 1
            for split in overlapping_splits:
                overlap_by_split[split] += 1
            continue
        representative = min(
            rows,
            key=lambda item: (
                item["label"], item["donor_key"], item["reference_id"], -item["vdjdb_score"]
            ),
        ).copy()
        representative["source_rows"] = len(rows)
        retained.append(representative)

    counts["deduplicated_nonoverlap_clonotypes"] = len(retained)
    report = {
        "filter_counts": dict(sorted(counts.items())),
        "benchmark_overlap_by_split": dict(sorted(overlap_by_split.items())),
    }
    return retained, report


def label_summary(records: Iterable[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["label"]].append(record)
    summaries = []
    for label, rows in sorted(groups.items()):
        lengths = sorted(row["length"] for row in rows)
        summaries.append({
            "label": label,
            "epitope": rows[0]["epitope"],
            "mhc_a": rows[0]["mhc_a"],
            "mhc_b": rows[0]["mhc_b"],
            "mhc_class": rows[0]["mhc_class"],
            "unique_cdr3": len(rows),
            "donors": len({row["donor_key"] for row in rows}),
            "length_min": lengths[0],
            "length_median": lengths[len(lengths) // 2],
            "length_max": lengths[-1],
        })
    return summaries


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def smoke_checkpoint(
    checkpoint_path: Path,
    run_config_path: Path,
    tokenizer_path: Path,
    sequences: list[str],
) -> dict:
    import torch

    from rtp_codec.data.multitask import resolve_encoder_tokenizer
    from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer
    from rtp_codec.tokenization.character import PAD_ID

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    training = run_config["training"]
    expected = {
        "tokenizer_type": "data_anchor",
        "latent_dim": 128,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "tcremp_loss_weight": 1.0,
        "pgen_loss_weight": 1.0,
        "reconstruction_loss_weight": 1.0,
        "seed": 42,
    }
    observed = {key: training[key] for key in expected}
    if observed != expected:
        raise ValueError(f"Frozen checkpoint run is not the locked DATA-ANCHOR R+T+P model: {observed}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = RTPCodecConfig(**payload["model_config"])
    if asdict(model_config) != run_config["model"]:
        raise ValueError("Checkpoint model_config differs from run_config.json")
    model = RTPCodecTransformer(model_config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(tokenizer_path))
    if tokenizer.vocab_size != model_config.input_vocab_size:
        raise ValueError("Tokenizer vocabulary does not match checkpoint input embedding")
    encoded = [tokenizer.encode(sequence, model_config.max_sequence_len) for sequence in sequences[:2]]
    width = max(map(len, encoded))
    tokens = torch.full((len(encoded), width), PAD_ID, dtype=torch.long)
    for index, ids in enumerate(encoded):
        tokens[index, : len(ids)] = torch.tensor(ids)
    with torch.no_grad():
        latent = model.encode(tokens, tokens.ne(PAD_ID))
    if latent.shape != (len(encoded), model_config.latent_dim) or not torch.isfinite(latent).all():
        raise ValueError(f"Frozen encoder smoke output is invalid: {tuple(latent.shape)}")
    return {
        "checkpoint_epoch_zero_based": int(payload["epoch"]),
        "checkpoint_best_val_loss": float(payload["best_val_loss"]),
        "model_config": asdict(model_config),
        "tokenizer_type": tokenizer.name,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "smoke_sequences": sequences[:2],
        "smoke_token_lengths": [len(ids) for ids in encoded],
        "smoke_latent_shape": list(latent.shape),
        "smoke_latent_finite": True,
    }


def main() -> None:
    args = parse_args()
    if args.min_support < 2 or args.min_donors < 2:
        raise ValueError("Primary support and donor thresholds must both be at least two.")
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight directory: {output}")
    output.mkdir(parents=True)

    benchmark_sequences, benchmark_hashes = read_benchmark_sequences(args.benchmark_dir)
    records, inventory = prepare_records(
        args.vdjdb_tsv, benchmark_sequences, min_score=args.min_vdjdb_score
    )
    summaries = label_summary(records)
    eligible_labels = {
        row["label"] for row in summaries
        if row["unique_cdr3"] >= args.min_support and row["donors"] >= args.min_donors
    }
    cohort = [row for row in records if row["label"] in eligible_labels]
    if len(eligible_labels) < 2:
        raise ValueError(
            f"Primary rule yielded {len(eligible_labels)} eligible labels; at least two are required."
        )

    fields = [
        "cdr3", "label", "epitope", "mhc_a", "mhc_b", "mhc_class", "donor_key",
        "reference_id", "vdjdb_score", "length", "source_rows",
    ]
    write_tsv(output / "cohort.tsv", cohort, fields)
    write_tsv(
        output / "label_summary.tsv",
        summaries,
        ["label", "epitope", "mhc_a", "mhc_b", "mhc_class", "unique_cdr3", "donors",
         "length_min", "length_median", "length_max"],
    )
    sensitivity = []
    for threshold in sorted(set(args.sensitivity_thresholds + [args.min_support])):
        selected = [
            row for row in summaries
            if row["unique_cdr3"] >= threshold and row["donors"] >= args.min_donors
        ]
        sensitivity.append({
            "min_unique_cdr3": threshold,
            "min_donors": args.min_donors,
            "eligible_labels": len(selected),
            "eligible_clonotypes": sum(row["unique_cdr3"] for row in selected),
        })
    write_tsv(
        output / "threshold_sensitivity.tsv", sensitivity,
        ["min_unique_cdr3", "min_donors", "eligible_labels", "eligible_clonotypes"],
    )

    checkpoint = smoke_checkpoint(
        args.checkpoint, args.run_config, args.tokenizer,
        [row["cdr3"] for row in cohort],
    )
    protocol = {
        "study_type": "single-cohort preflight; model comparison unresolved",
        "primary_label": "antigen.epitope|mhc.a|mhc.b-or-dash|mhc.class",
        "primary_selection": {
            "all_labels_meeting_rule": True,
            "min_unique_cdr3": args.min_support,
            "min_donors": args.min_donors,
            "vdjdb_score_min": args.min_vdjdb_score,
            "exact_cdr3_deduplication": True,
            "ambiguous_multilabel_cdr3_removed": True,
            "multi_donor_cdr3_removed_for_clustered_inference": True,
            "all_locked_benchmark_split_overlaps_removed": True,
        },
        "future_metrics_after_model_protocol_acceptance": {
            "primary": "per-label within/between cosine-distance ratio; lower is better",
            "secondary_distance": "Euclidean",
            "pairwise": ["same-label AUROC", "same-label AUPRC with sampled prevalence reported"],
            "retrieval": ["donor-excluded precision@1/5/10", "donor-excluded MAP@R"],
            "negative_control": "between-label pairs matched on ordered CDR3 lengths",
            "nulls": [
                "absolute length-difference score",
                "1000 label permutations within length bins and study/donor strata",
            ],
            "uncertainty": "2000 donor-cluster bootstrap replicates, fixed seed",
            "visualization": "descriptive only; no UMAP/scaler/metric learning fit on evaluation data",
        },
        "seed": args.seed,
        "embedding_extraction_authorized": False,
        "blocking_gate": "accept cohort and resolve matched model-comparison protocol",
    }
    stable_json(output / "evaluation_protocol.json", protocol)
    stable_json(output / "inventory.json", inventory)

    manifest = {
        "status": "ready_for_cohort_review_not_embedding_extraction",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source": {
            "path": str(args.vdjdb_tsv),
            "sha256": sha256_file(args.vdjdb_tsv),
            "bytes": args.vdjdb_tsv.stat().st_size,
        },
        "benchmark": {
            "path": str(args.benchmark_dir),
            "manifest_sha256": benchmark_hashes,
            "unique_sequences_by_split": {
                split: len(values) for split, values in benchmark_sequences.items()
            },
        },
        "frozen_checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "run_config_path": str(args.run_config),
            "run_config_sha256": sha256_file(args.run_config),
            "tokenizer_path": str(args.tokenizer),
            "tokenizer_sha256": sha256_file(args.tokenizer),
            **checkpoint,
        },
        "cohort": {
            "eligible_labels": len(eligible_labels),
            "unique_cdr3": len(cohort),
            "donors": len({row["donor_key"] for row in cohort}),
            "cohort_sha256": sha256_file(output / "cohort.tsv"),
            "label_summary_sha256": sha256_file(output / "label_summary.tsv"),
        },
        "embedding_extraction_performed": False,
        "evaluation_performed": False,
    }
    stable_json(output / "PREFLIGHT.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
