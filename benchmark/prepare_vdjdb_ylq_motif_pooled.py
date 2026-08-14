"""Prepare pooled exact-YLQ evaluation data from official VDJdb motif members."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from benchmark.prepare_vdjdb_epitope_cohort import (
    VALID_AA,
    clean_text,
    normalized_context,
    parse_json_object,
    read_benchmark_sequences,
    sha256_file,
    stable_json,
    write_tsv,
)


TARGET_PEPTIDE = "YLQPRTFLL"
TARGET_CONTEXT = ("HLA-A*02:01", "B2M", "MHCI")
AA = "ACDEFGHIKLMNPQRSTVWY"


def composition(sequence: str) -> np.ndarray:
    return np.asarray([sequence.count(aa) / len(sequence) for aa in AA], dtype=float)


def read_official_members(path: Path) -> tuple[dict[tuple[str, str], set[str]], dict]:
    counts: Counter[str] = Counter()
    membership: dict[tuple[str, str], set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "species", "antigen.epitope", "mhc.a", "mhc.b", "mhc.class", "gene",
            "cdr3aa", "cid", "csz", "v.segm", "j.segm",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Official cluster-members file lacks columns: {sorted(missing)}")
        for row in reader:
            counts["source_rows"] += 1
            if normalized_context(row["species"]) != "HOMOSAPIENS":
                continue
            if normalized_context(row["gene"]) != "TRB":
                continue
            context = (
                normalized_context(row["mhc.a"]), normalized_context(row["mhc.b"]),
                normalized_context(row["mhc.class"]),
            )
            if context != TARGET_CONTEXT:
                continue
            counts["exact_context_trb_rows"] += 1
            sequence = normalized_context(row["cdr3aa"])
            peptide = normalized_context(row["antigen.epitope"])
            cid = clean_text(row["cid"])
            if not sequence or len(sequence) > 40 or set(sequence).difference(VALID_AA):
                counts["invalid_member_cdr3"] += 1
                continue
            if not peptide or not cid:
                counts["missing_member_label_or_cid"] += 1
                continue
            membership[(sequence, peptide)].add(cid)
            counts["valid_exact_context_membership_rows"] += 1
            if peptide == TARGET_PEPTIDE:
                counts["exact_ylq_membership_rows"] += 1
    report = {
        "counts": dict(sorted(counts.items())),
        "unique_sequence_peptide_memberships": len(membership),
        "unique_ylq_member_cdr3": len({key[0] for key in membership if key[1] == TARGET_PEPTIDE}),
    }
    return membership, report


def read_joined_records(
    vdjdb_path: Path,
    membership: dict[tuple[str, str], set[str]],
    benchmark: dict[str, set[str]],
    min_score: float,
) -> tuple[list[dict], dict]:
    counts: Counter[str] = Counter()
    overlap: Counter[str] = Counter()
    grouped: dict[str, list[dict]] = defaultdict(list)
    with vdjdb_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "gene", "cdr3", "v.segm", "j.segm", "species", "mhc.a", "mhc.b",
            "mhc.class", "antigen.epitope", "reference.id", "vdjdb.score", "meta", "cdr3fix",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pinned VDJdb table lacks columns: {sorted(missing)}")
        for row in reader:
            counts["source_rows"] += 1
            if normalized_context(row["gene"]) != "TRB":
                continue
            if normalized_context(row["species"]) != "HOMOSAPIENS":
                continue
            context = (
                normalized_context(row["mhc.a"]), normalized_context(row["mhc.b"]),
                normalized_context(row["mhc.class"]),
            )
            if context != TARGET_CONTEXT:
                continue
            counts["exact_context_trb_rows"] += 1
            sequence = normalized_context(row["cdr3"])
            if not sequence or len(sequence) > 40 or set(sequence).difference(VALID_AA):
                counts["invalid_cdr3"] += 1
                continue
            fix = parse_json_object(row.get("cdr3fix", ""))
            if fix.get("good") is not True:
                counts["cdr3fix_not_good"] += 1
                continue
            fixed = normalized_context(fix.get("cdr3"))
            if fixed and fixed != sequence:
                counts["cdr3fix_changes_sequence"] += 1
                continue
            try:
                score = float(row["vdjdb.score"])
            except ValueError:
                score = math.nan
            if not math.isfinite(score) or score < min_score:
                counts["low_or_invalid_score"] += 1
                continue
            peptide = normalized_context(row["antigen.epitope"])
            key = (sequence, peptide)
            if key not in membership:
                counts["not_official_motif_member"] += 1
                continue
            counts["exact_official_membership_join_rows"] += 1
            overlapping = [split for split, sequences in benchmark.items() if sequence in sequences]
            if overlapping:
                counts["benchmark_overlap_rows"] += 1
                for split in overlapping:
                    overlap[split] += 1
                continue
            meta = parse_json_object(row.get("meta", ""))
            subject = clean_text(meta.get("subject.id") or meta.get("subject_id"))
            study = clean_text(meta.get("study.id") or meta.get("study_id") or row["reference.id"])
            grouped[sequence].append({
                "cdr3": sequence,
                "epitope": peptide,
                "mhc_a": TARGET_CONTEXT[0],
                "mhc_b": TARGET_CONTEXT[1],
                "mhc_class": TARGET_CONTEXT[2],
                "v_call": normalized_context(row["v.segm"]),
                "j_call": normalized_context(row["j.segm"]),
                "reference_id": clean_text(row["reference.id"]),
                "vdjdb_score": score,
                "donor_key": f"{study}|{subject}" if subject else "",
                "motif_cids": membership[key],
                "length": len(sequence),
            })

    retained = []
    for sequence, rows in sorted(grouped.items()):
        epitopes = {row["epitope"] for row in rows}
        if len(epitopes) != 1:
            counts["ambiguous_multilabel_cdr3"] += 1
            continue
        representative = min(
            rows,
            key=lambda row: (
                -row["vdjdb_score"], row["epitope"], row["v_call"], row["j_call"],
                row["reference_id"],
            ),
        ).copy()
        representative["source_rows"] = len(rows)
        representative["donor_count_descriptive_only"] = len({
            row["donor_key"] for row in rows if row["donor_key"]
        })
        representative["motif_cids"] = ",".join(sorted({
            cid for row in rows for cid in row["motif_cids"]
        }))
        retained.append(representative)
    counts["pooled_exact_cdr3_after_dedup"] = len(retained)
    return retained, {
        "filter_counts": dict(sorted(counts.items())),
        "benchmark_overlap_rows_by_split": dict(sorted(overlap.items())),
    }


def build_pairs(positives: list[dict], negatives: list[dict], seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    positive_pairs = [(left, right) for left in range(len(positives))
                      for right in range(left + 1, len(positives))]
    rng.shuffle(positive_pairs)
    negative_by_length: dict[int, list[int]] = defaultdict(list)
    negative_composition = np.asarray([composition(record["cdr3"]) for record in negatives])
    for index, record in enumerate(negatives):
        negative_by_length[int(record["length"])].append(index)
    usage: Counter[int] = Counter()
    rows = []
    excluded_no_control_pairs = 0
    forced_orientation_pairs = 0
    for pair_id, (query, positive) in enumerate(positive_pairs):
        query_supported = bool(negative_by_length.get(int(positives[query]["length"]), []))
        positive_supported = bool(negative_by_length.get(int(positives[positive]["length"]), []))
        if not query_supported and not positive_supported:
            excluded_no_control_pairs += 1
            continue
        if query_supported and not positive_supported:
            query, positive = positive, query
            forced_orientation_pairs += 1
        elif not query_supported and positive_supported:
            forced_orientation_pairs += 1
        elif rng.integers(2):
            query, positive = positive, query
        qrow, prow = positives[query], positives[positive]
        candidates = negative_by_length.get(int(prow["length"]), [])
        if not candidates:
            raise ValueError(f"No exact-length official-motif negative for length {prow['length']}.")
        target = composition(prow["cdr3"])

        def rank(index: int) -> tuple:
            candidate = negatives[index]
            v_mismatch = int(not prow["v_call"] or not candidate["v_call"]
                             or prow["v_call"] != candidate["v_call"])
            j_mismatch = int(not prow["j_call"] or not candidate["j_call"]
                             or prow["j_call"] != candidate["j_call"])
            comp_l1 = float(np.abs(negative_composition[index] - target).sum())
            return comp_l1, v_mismatch + j_mismatch, usage[index], candidate["cdr3"]

        negative = min(candidates, key=rank)
        usage[negative] += 1
        nrow = negatives[negative]
        rows.append({
            "pair_id": pair_id,
            "query_index": query,
            "positive_index": positive,
            "negative_index": len(positives) + negative,
            "query_cdr3": qrow["cdr3"],
            "positive_cdr3": prow["cdr3"],
            "negative_cdr3": nrow["cdr3"],
            "query_length": qrow["length"],
            "positive_length": prow["length"],
            "negative_length": nrow["length"],
            "v_exact_match": int(bool(prow["v_call"])
                                 and prow["v_call"] == nrow["v_call"]),
            "j_exact_match": int(bool(prow["j_call"])
                                 and prow["j_call"] == nrow["j_call"]),
            "composition_l1": float(np.abs(negative_composition[negative] - target).sum()),
        })
    if not rows:
        raise ValueError("No distinct pooled positive CDR3 pairs are available.")
    return rows, {
        "positive_pairs": len(rows),
        "all_possible_positive_pairs": len(positive_pairs),
        "excluded_pairs_both_endpoints_lack_exact_length_controls": excluded_no_control_pairs,
        "forced_orientation_to_supported_comparator": forced_orientation_pairs,
        "donor_constraint": "none_pooled_by_user_protocol",
        "distinct_cdr3_triplets": True,
        "exact_length_fraction": 1.0,
        "v_exact_match_fraction": float(np.mean([row["v_exact_match"] for row in rows])),
        "j_exact_match_fraction": float(np.mean([row["j_exact_match"] for row in rows])),
        "mean_composition_l1": float(np.mean([row["composition_l1"] for row in rows])),
        "unique_negatives_used": len(usage),
        "max_negative_reuse": max(usage.values()),
    }


def select_compact_negatives(
    positives: list[dict], negatives: list[dict], per_positive: int, seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Select globally unique same-length controls in seeded per-stratum rounds."""
    if per_positive < 1:
        raise ValueError("Controls per positive must be positive.")
    negative_by_length: dict[int, list[int]] = defaultdict(list)
    negative_composition = np.asarray([composition(record["cdr3"]) for record in negatives])
    for index, record in enumerate(negatives):
        negative_by_length[int(record["length"])].append(index)
    positive_by_length: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(positives):
        positive_by_length[int(record["length"])].append(index)

    selected: list[int] = []
    used: set[int] = set()
    assignments = []
    strata = []
    for length in sorted(positive_by_length):
        positive_indices = sorted(positive_by_length[length], key=lambda index: positives[index]["cdr3"])
        rng = np.random.default_rng(seed + length)
        positive_indices = [positive_indices[index] for index in rng.permutation(len(positive_indices))]
        available_in_stratum = negative_by_length.get(length, [])
        for slot in range(1, per_positive + 1):
            for positive_index in positive_indices:
                candidates = [index for index in available_in_stratum if index not in used]
                if not candidates:
                    break
                positive = positives[positive_index]
                target = composition(positive["cdr3"])

                def rank(index: int) -> tuple:
                    candidate = negatives[index]
                    comp_l1 = float(np.abs(negative_composition[index] - target).sum())
                    v_mismatch = int(not positive["v_call"] or not candidate["v_call"]
                                     or positive["v_call"] != candidate["v_call"])
                    j_mismatch = int(not positive["j_call"] or not candidate["j_call"]
                                     or positive["j_call"] != candidate["j_call"])
                    return comp_l1, v_mismatch + j_mismatch, candidate["cdr3"]

                chosen = min(candidates, key=rank)
                used.add(chosen)
                selected.append(chosen)
                control = negatives[chosen]
                assignments.append({
                    "selection_order": len(assignments),
                    "length": length,
                    "slot_for_positive": slot,
                    "positive_cdr3": positive["cdr3"],
                    "control_cdr3": control["cdr3"],
                    "composition_l1": float(np.abs(negative_composition[chosen] - target).sum()),
                    "v_exact_match": int(bool(positive["v_call"])
                                         and positive["v_call"] == control["v_call"]),
                    "j_exact_match": int(bool(positive["j_call"])
                                         and positive["j_call"] == control["j_call"]),
                })
        selected_count = sum(1 for row in assignments if row["length"] == length)
        target_count = per_positive * len(positive_indices)
        strata.append({
            "length": length,
            "positive_cdr3": len(positive_indices),
            "eligible_negative_cdr3": len(available_in_stratum),
            "nominal_target": target_count,
            "selected_unique_controls": selected_count,
            "shortfall": target_count - selected_count,
            "all_available_retained_if_short": int(len(available_in_stratum) < target_count),
        })
    selected_records = [negatives[index] for index in selected]
    if len({record["cdr3"] for record in selected_records}) != len(selected_records):
        raise ValueError("Compact control selector reused a CDR3.")
    report = {
        "selection_seed": seed,
        "controls_per_positive_cap": per_positive,
        "nominal_global_cap": per_positive * len(positives),
        "eligible_controls_before_compaction": len(negatives),
        "selected_unique_controls": len(selected_records),
        "selection_without_replacement": True,
        "gallery_control_reuse": 0,
        "selection_priority": "minimum AA-composition L1, then V/J mismatch count, then CDR3",
        "length_strata": strata,
    }
    return selected_records, assignments, strata, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vdjdb-tsv", type=Path, required=True)
    parser.add_argument("--cluster-members", type=Path, required=True)
    parser.add_argument("--motif-pwms", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--min-positives", type=int, default=50)
    parser.add_argument("--min-negatives", type=int, default=100)
    parser.add_argument("--controls-per-positive", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    benchmark, benchmark_hashes = read_benchmark_sequences(args.benchmark_dir)
    membership, motif_report = read_official_members(args.cluster_members)
    records, join_report = read_joined_records(
        args.vdjdb_tsv, membership, benchmark, args.min_score
    )
    positives = [record for record in records if record["epitope"] == TARGET_PEPTIDE]
    eligible_negatives = [record for record in records if record["epitope"] != TARGET_PEPTIDE]
    if len(positives) < args.min_positives or len(eligible_negatives) < args.min_negatives:
        blocked = {
            "status": "blocked_insufficient_official_motif_support",
            "positive_cdr3": len(positives), "negative_cdr3": len(eligible_negatives),
            "motif": motif_report, "join": join_report,
        }
        stable_json(args.output_dir / "BLOCKED.json", blocked)
        raise ValueError(
            f"Insufficient support after join: {len(positives)} positives, "
            f"{len(eligible_negatives)} negatives"
        )

    negatives, control_assignments, length_strata, control_report = select_compact_negatives(
        positives, eligible_negatives, args.controls_per_positive, args.seed
    )

    candidates = []
    fields = [
        "candidate_index", "is_ylq", "cdr3", "epitope", "mhc_a", "mhc_b", "mhc_class",
        "v_call", "j_call", "reference_id", "vdjdb_score", "length", "motif_cids",
        "source_rows", "donor_count_descriptive_only",
    ]
    for index, record in enumerate([*positives, *negatives]):
        candidate = dict(record)
        candidate["candidate_index"] = index
        candidate["is_ylq"] = int(index < len(positives))
        candidates.append(candidate)
    if len({row["cdr3"] for row in candidates}) != len(candidates):
        raise ValueError("Candidate set is not exact-CDR3 deduplicated.")
    write_tsv(args.output_dir / "candidates.tsv", candidates, fields)
    write_tsv(
        args.output_dir / "control_selection.tsv", control_assignments,
        list(control_assignments[0]),
    )
    write_tsv(args.output_dir / "control_length_strata.tsv", length_strata, list(length_strata[0]))
    pairs, matching = build_pairs(positives, negatives, args.seed)
    write_tsv(args.output_dir / "pair_manifest.tsv", pairs, list(pairs[0]))
    manifest = {
        "status": "ready",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "target_peptide": TARGET_PEPTIDE,
        "exact_context": {"mhc_a": TARGET_CONTEXT[0], "mhc_b": TARGET_CONTEXT[1],
                          "mhc_class": TARGET_CONTEXT[2]},
        "pooling": "all records pooled; donor metadata is descriptive only and never constrains pairs/retrieval",
        "positive_cdr3": len(positives),
        "negative_cdr3": len(negatives),
        "eligible_negative_cdr3_before_compaction": len(eligible_negatives),
        "candidate_cdr3": len(candidates),
        "matching": matching,
        "compact_control_selection": control_report,
        "motif_inventory": motif_report,
        "vdjdb_join_and_quality": join_report,
        "sources": {
            "vdjdb": {"path": str(args.vdjdb_tsv), "sha256": sha256_file(args.vdjdb_tsv)},
            "cluster_members": {"path": str(args.cluster_members),
                                "sha256": sha256_file(args.cluster_members)},
            "motif_pwms": {"path": str(args.motif_pwms), "sha256": sha256_file(args.motif_pwms)},
            "benchmark_manifests": benchmark_hashes,
        },
        "artifacts": {
            "candidates_sha256": sha256_file(args.output_dir / "candidates.tsv"),
            "pairs_sha256": sha256_file(args.output_dir / "pair_manifest.tsv"),
            "control_selection_sha256": sha256_file(args.output_dir / "control_selection.tsv"),
            "control_length_strata_sha256": sha256_file(args.output_dir / "control_length_strata.tsv"),
        },
        "official_motif_rule": (
            "Exact sequence+peptide membership in official cluster_members.txt after exact "
            "HomoSapiens/TRB/HLA-A*02:01/B2M/MHCI filtering; no PWM scanning or motif mining."
        ),
        "circularity_caveat": (
            "Official motif membership is sequence-cluster-conditioned. This evaluates within-VDJdb "
            "motif/epitope organization, not independent epitope evidence beyond sequence motif/composition."
        ),
        "exploratory_n40_protocol": True,
        "embedding_extraction_performed": False,
    }
    stable_json(args.output_dir / "PREFLIGHT.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
