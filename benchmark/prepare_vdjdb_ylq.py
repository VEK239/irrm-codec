"""Prepare an exact-peptide, dominant-MHC YLQ VDJdb evaluation cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from benchmark.prepare_vdjdb_epitope_cohort import (
    prepare_records,
    read_benchmark_sequences,
    sha256_file,
    stable_json,
    write_tsv,
)


TARGET_PEPTIDE = "YLQPRTFLL"
AA = "ACDEFGHIKLMNPQRSTVWY"


def composition(sequence: str) -> np.ndarray:
    return np.asarray([sequence.count(aa) / len(sequence) for aa in AA], dtype=float)


def context(record: dict) -> tuple[str, str, str]:
    return record["mhc_a"], record["mhc_b"] or "-", record["mhc_class"]


def build_pairs(positives: list[dict], negatives: list[dict], seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    positive_pairs = [
        (left, right)
        for left in range(len(positives))
        for right in range(left + 1, len(positives))
        if positives[left]["donor_key"] != positives[right]["donor_key"]
    ]
    rng.shuffle(positive_pairs)
    negative_by_length: dict[int, list[int]] = defaultdict(list)
    negative_composition = []
    for index, record in enumerate(negatives):
        negative_by_length[int(record["length"])].append(index)
        negative_composition.append(composition(record["cdr3"]))
    negative_composition = np.asarray(negative_composition)
    usage: Counter[int] = Counter()
    rows = []
    for pair_number, (query, positive) in enumerate(positive_pairs):
        if rng.integers(2):
            query, positive = positive, query
        query_record = positives[query]
        positive_record = positives[positive]
        candidates = [
            index for index in negative_by_length[int(positive_record["length"])]
            if negatives[index]["donor_key"] not in {
                query_record["donor_key"], positive_record["donor_key"]
            }
        ]
        if not candidates:
            raise ValueError(
                f"No exact-length donor-disjoint negative for positive length {positive_record['length']}."
            )
        target_composition = composition(positive_record["cdr3"])
        def rank(index: int) -> tuple:
            candidate = negatives[index]
            v_missing = not positive_record["v_call"] or not candidate["v_call"]
            j_missing = not positive_record["j_call"] or not candidate["j_call"]
            v_mismatch = int(v_missing or candidate["v_call"] != positive_record["v_call"])
            j_mismatch = int(j_missing or candidate["j_call"] != positive_record["j_call"])
            comp_l1 = float(np.abs(negative_composition[index] - target_composition).sum())
            return v_mismatch + j_mismatch, usage[index], comp_l1, candidate["cdr3"]
        negative = min(candidates, key=rank)
        usage[negative] += 1
        negative_record = negatives[negative]
        rows.append({
            "pair_id": pair_number,
            "query_index": query,
            "positive_index": positive,
            "negative_index": len(positives) + negative,
            "query_donor": query_record["donor_key"],
            "positive_donor": positive_record["donor_key"],
            "negative_donor": negative_record["donor_key"],
            "query_length": query_record["length"],
            "positive_length": positive_record["length"],
            "negative_length": negative_record["length"],
            "v_exact_match": int(
                bool(positive_record["v_call"])
                and positive_record["v_call"] == negative_record["v_call"]
            ),
            "j_exact_match": int(
                bool(positive_record["j_call"])
                and positive_record["j_call"] == negative_record["j_call"]
            ),
            "vj_exact_match": int(
                bool(positive_record["v_call"] and positive_record["j_call"])
                and positive_record["v_call"] == negative_record["v_call"]
                and positive_record["j_call"] == negative_record["j_call"]
            ),
            "composition_l1": float(
                np.abs(negative_composition[negative] - target_composition).sum()
            ),
        })
    if not rows:
        raise ValueError("No donor-disjoint positive YLQ pairs are available.")
    report = {
        "positive_pairs": len(rows),
        "exact_length_fraction": 1.0,
        "v_exact_match_fraction": float(np.mean([row["v_exact_match"] for row in rows])),
        "j_exact_match_fraction": float(np.mean([row["j_exact_match"] for row in rows])),
        "vj_exact_match_fraction": float(np.mean([row["vj_exact_match"] for row in rows])),
        "mean_composition_l1": float(np.mean([row["composition_l1"] for row in rows])),
        "unique_negatives_used": len(usage),
        "max_negative_reuse": max(usage.values()),
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vdjdb-tsv", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-positive-cdr3", type=int, default=50)
    parser.add_argument("--min-positive-donors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    benchmark, benchmark_hashes = read_benchmark_sequences(args.benchmark_dir)
    records, inventory = prepare_records(args.vdjdb_tsv, benchmark, min_score=1.0)
    exact_peptide = [record for record in records if record["epitope"] == TARGET_PEPTIDE]
    distribution = []
    by_context: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in exact_peptide:
        by_context[context(record)].append(record)
    for key, rows in sorted(by_context.items(), key=lambda item: (-len(item[1]), item[0])):
        distribution.append({
            "mhc_a": key[0], "mhc_b": key[1], "mhc_class": key[2],
            "unique_cdr3": len(rows), "donors": len({row["donor_key"] for row in rows}),
        })
    if not distribution:
        raise ValueError("No exact YLQPRTFLL records remain after leakage filtering.")
    dominant = distribution[0]
    dominant_context = (dominant["mhc_a"], dominant["mhc_b"], dominant["mhc_class"])
    if dominant["unique_cdr3"] < args.min_positive_cdr3 or dominant["donors"] < args.min_positive_donors:
        stable_json(args.output_dir / "BLOCKED.json", {
            "status": "blocked_insufficient_support", "mhc_distribution": distribution,
        })
        raise ValueError("Dominant exact YLQ peptide+MHC context has insufficient support.")

    positives = [
        record for record in exact_peptide if context(record) == dominant_context
    ]
    negatives = [
        record for record in records
        if record["epitope"] != TARGET_PEPTIDE and context(record) == dominant_context
    ]
    if len(negatives) < 50:
        raise ValueError(f"Only {len(negatives)} same-context non-YLQ negatives remain.")
    candidates = []
    candidate_fields = [
        "candidate_index", "is_ylq", "cdr3", "epitope", "mhc_a", "mhc_b", "mhc_class",
        "donor_key", "reference_id", "v_call", "j_call", "vdjdb_score", "length", "source_rows",
    ]
    for index, record in enumerate([*positives, *negatives]):
        candidate = dict(record)
        candidate["candidate_index"] = index
        candidate["is_ylq"] = int(index < len(positives))
        candidates.append(candidate)
    write_tsv(args.output_dir / "candidates.tsv", candidates, candidate_fields)
    write_tsv(
        args.output_dir / "mhc_distribution.tsv", distribution,
        ["mhc_a", "mhc_b", "mhc_class", "unique_cdr3", "donors"],
    )
    pairs, match_report = build_pairs(positives, negatives, seed=args.seed)
    pair_fields = list(pairs[0])
    write_tsv(args.output_dir / "pair_manifest.tsv", pairs, pair_fields)
    stable_json(args.output_dir / "inventory.json", inventory)
    manifest = {
        "status": "ready",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "target_peptide_definition": "exact equality to YLQPRTFLL",
        "primary_context_rule": "dominant exact peptide+MHC context after all filters",
        "primary_context": {
            **dominant,
            "label": "|".join(dominant_context),
        },
        "mhc_distribution": distribution,
        "positive_cdr3": len(positives),
        "positive_donors": len({row["donor_key"] for row in positives}),
        "same_context_non_ylq_negative_cdr3": len(negatives),
        "same_context_non_ylq_negative_donors": len({row["donor_key"] for row in negatives}),
        "matching": match_report,
        "leakage": inventory,
        "source": {"path": str(args.vdjdb_tsv), "sha256": sha256_file(args.vdjdb_tsv)},
        "benchmark_manifest_sha256": benchmark_hashes,
        "artifacts": {
            "candidates_sha256": sha256_file(args.output_dir / "candidates.tsv"),
            "pairs_sha256": sha256_file(args.output_dir / "pair_manifest.tsv"),
            "mhc_distribution_sha256": sha256_file(args.output_dir / "mhc_distribution.tsv"),
        },
        "motif_expansion_performed": False,
        "embedding_extraction_performed": False,
    }
    stable_json(args.output_dir / "PREFLIGHT.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
