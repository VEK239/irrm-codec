"""Prepare an exploratory top-10 viral official-motif VDJdb benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

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
from benchmark.prepare_vdjdb_ylq_motif_pooled import (
    TARGET_CONTEXT,
    build_pairs,
    select_compact_negatives,
)


def read_species_list(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(value)
    if not values or len(values) != len(set(values)):
        raise ValueError("Viral antigen-species list must be nonempty and unique.")
    return values


def read_official_members(path: Path) -> tuple[dict[tuple[str, str, str], set[str]], dict]:
    counts: Counter[str] = Counter()
    membership: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "species", "antigen.epitope", "antigen.species", "mhc.a", "mhc.b",
            "mhc.class", "gene", "cdr3aa", "cid", "v.segm", "j.segm",
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
            antigen_species = clean_text(row["antigen.species"])
            cid = clean_text(row["cid"])
            if not sequence or len(sequence) > 40 or set(sequence).difference(VALID_AA):
                counts["invalid_member_cdr3"] += 1
                continue
            if not peptide or not antigen_species or not cid:
                counts["missing_member_label_species_or_cid"] += 1
                continue
            membership[(sequence, peptide, antigen_species)].add(cid)
            counts["valid_exact_context_membership_rows"] += 1
    return membership, {
        "counts": dict(sorted(counts.items())),
        "unique_sequence_peptide_species_memberships": len(membership),
    }


def read_joined_records(
    vdjdb_path: Path,
    membership: dict[tuple[str, str, str], set[str]],
    benchmark: dict[str, set[str]],
    min_score: float,
) -> tuple[list[dict], dict]:
    counts: Counter[str] = Counter()
    overlaps: Counter[str] = Counter()
    grouped: dict[str, list[dict]] = defaultdict(list)
    with vdjdb_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "gene", "cdr3", "v.segm", "j.segm", "species", "mhc.a", "mhc.b",
            "mhc.class", "antigen.epitope", "antigen.species", "reference.id",
            "vdjdb.score", "meta", "cdr3fix",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pinned VDJdb table lacks columns: {sorted(missing)}")
        for row in reader:
            counts["source_rows"] += 1
            if normalized_context(row["gene"]) != "TRB" \
                    or normalized_context(row["species"]) != "HOMOSAPIENS":
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
            antigen_species = clean_text(row["antigen.species"])
            key = (sequence, peptide, antigen_species)
            if key not in membership:
                counts["not_exact_official_motif_member"] += 1
                continue
            counts["exact_official_membership_join_rows"] += 1
            overlapping = [split for split, sequences in benchmark.items() if sequence in sequences]
            if overlapping:
                counts["benchmark_overlap_rows"] += 1
                for split in overlapping:
                    overlaps[split] += 1
                continue
            meta = parse_json_object(row.get("meta", ""))
            subject = clean_text(meta.get("subject.id") or meta.get("subject_id"))
            study = clean_text(meta.get("study.id") or meta.get("study_id") or row["reference.id"])
            grouped[sequence].append({
                "cdr3": sequence,
                "epitope": peptide,
                "antigen_species": antigen_species,
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
        labels = {(row["epitope"], row["antigen_species"]) for row in rows}
        if len(labels) != 1:
            counts["ambiguous_multilabel_or_species_cdr3"] += 1
            continue
        representative = min(
            rows,
            key=lambda row: (
                -row["vdjdb_score"], row["epitope"], row["antigen_species"],
                row["v_call"], row["j_call"], row["reference_id"],
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
    counts["pooled_exact_cdr3_after_global_dedup"] = len(retained)
    return retained, {
        "filter_counts": dict(sorted(counts.items())),
        "benchmark_overlap_rows_by_split": dict(sorted(overlaps.items())),
    }


def rank_epitopes(records: list[dict], viral_species: set[str]) -> tuple[list[dict], list[dict]]:
    species_counts = Counter(record["antigen_species"] for record in records)
    viral = [record for record in records if record["antigen_species"] in viral_species]
    by_epitope: dict[str, list[dict]] = defaultdict(list)
    for record in viral:
        by_epitope[record["epitope"]].append(record)
    ranking = []
    for peptide, rows in by_epitope.items():
        ranking.append({
            "epitope": peptide,
            "unique_cdr3": len(rows),
            "antigen_species": ",".join(sorted({row["antigen_species"] for row in rows})),
            "species_count": len({row["antigen_species"] for row in rows}),
        })
    ranking.sort(key=lambda row: (-row["unique_cdr3"], row["epitope"]))
    inventory = [
        {
            "antigen_species": species,
            "unique_cdr3": count,
            "classified_viral_by_exact_allowlist": int(species in viral_species),
        }
        for species, count in sorted(species_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return ranking, inventory


def choose_top10(ranking: list[dict], requested_threshold: int) -> tuple[list[dict], dict]:
    requested_epitopes = 10
    evaluable = [row for row in ranking if row["unique_cdr3"] >= 2]
    if not evaluable:
        raise ValueError("No viral epitope has two distinct post-filter CDR3s.")
    passing = [row for row in evaluable if row["unique_cdr3"] >= requested_threshold]
    if len(passing) >= requested_epitopes:
        threshold = requested_threshold
        relaxed = False
        selected_count = requested_epitopes
    elif len(evaluable) >= requested_epitopes:
        threshold = int(evaluable[requested_epitopes - 1]["unique_cdr3"])
        relaxed = True
        selected_count = requested_epitopes
    else:
        threshold = int(evaluable[-1]["unique_cdr3"])
        relaxed = True
        selected_count = len(evaluable)
    selected = [row.copy() for row in evaluable if row["unique_cdr3"] >= threshold][
        :selected_count
    ]
    for index, row in enumerate(selected, start=1):
        row["rank"] = index
    return selected, {
        "requested_minimum_unique_cdr3": requested_threshold,
        "epitopes_meeting_requested_threshold": len(passing),
        "requested_epitopes": requested_epitopes,
        "selected_epitopes": len(selected),
        "evaluable_epitope_shortfall": requested_epitopes - len(selected),
        "all_post_filter_viral_epitopes": len(ranking),
        "viral_epitopes_with_two_distinct_cdr3": len(evaluable),
        "viral_epitopes_below_distinct_pair_minimum": len(ranking) - len(evaluable),
        "effective_minimum_unique_cdr3": threshold,
        "threshold_transparently_lowered": relaxed,
        "selection_rule": (
            "post-filter unique CDR3 descending, then exact peptide lexical; at least two "
            "distinct target CDR3s required for within-target pairs"
        ),
        "model_metrics_used_for_selection": False,
    }


def write_epitope_dataset(
    output_dir: Path,
    rank: int,
    epitope: str,
    positives: list[dict],
    eligible_negatives: list[dict],
    controls_per_positive: int,
    max_pairs: int,
    seed: int,
) -> dict:
    directory = output_dir / f"epitope-{rank:02d}-{epitope}"
    directory.mkdir()
    negatives, assignments, strata, control_report = select_compact_negatives(
        positives, eligible_negatives, controls_per_positive, seed + rank * 1000
    )
    candidates = []
    fields = [
        "candidate_index", "is_target", "cdr3", "epitope", "antigen_species",
        "mhc_a", "mhc_b", "mhc_class", "v_call", "j_call", "reference_id",
        "vdjdb_score", "length", "motif_cids", "source_rows",
        "donor_count_descriptive_only",
    ]
    for index, record in enumerate([*positives, *negatives]):
        candidate = dict(record)
        candidate["candidate_index"] = index
        candidate["is_target"] = int(index < len(positives))
        candidates.append(candidate)
    if len({row["cdr3"] for row in candidates}) != len(candidates):
        raise ValueError(f"{epitope} candidate set is not exact-CDR3 deduplicated.")
    write_tsv(directory / "candidates.tsv", candidates, fields)
    write_tsv(directory / "control_selection.tsv", assignments, list(assignments[0]))
    write_tsv(directory / "control_length_strata.tsv", strata, list(strata[0]))
    pairs, matching = build_pairs(positives, negatives, seed + rank * 1000)
    pairs_before_cap = len(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    matching["pairs_before_deterministic_cap"] = pairs_before_cap
    matching["primary_pair_cap"] = max_pairs
    matching["pairs_after_deterministic_cap"] = len(pairs)
    matching["positive_pairs_used_for_evaluation"] = len(pairs)
    matching["pair_cap_rule"] = "first seeded-shuffled distinct positive pairs"
    write_tsv(directory / "pair_manifest.tsv", pairs, list(pairs[0]))
    report = {
        "rank": rank,
        "epitope": epitope,
        "positive_cdr3": len(positives),
        "selected_unique_negative_cdr3": len(negatives),
        "eligible_viral_negative_cdr3_before_compaction": len(eligible_negatives),
        "candidate_cdr3": len(candidates),
        "matching": matching,
        "compact_control_selection": control_report,
        "artifacts": {
            "candidates_sha256": sha256_file(directory / "candidates.tsv"),
            "pairs_sha256": sha256_file(directory / "pair_manifest.tsv"),
            "control_selection_sha256": sha256_file(directory / "control_selection.tsv"),
            "control_length_strata_sha256": sha256_file(directory / "control_length_strata.tsv"),
        },
    }
    stable_json(directory / "PREFLIGHT.json", report)
    return {"report": report, "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vdjdb-tsv", type=Path, required=True)
    parser.add_argument("--cluster-members", type=Path, required=True)
    parser.add_argument("--motif-pwms", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--viral-species-list", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--requested-min-positives", type=int, default=40)
    parser.add_argument("--controls-per-positive", type=int, default=4)
    parser.add_argument("--max-pairs-per-epitope", type=int, default=5000)
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
    all_species = Counter(record["antigen_species"] for record in records)
    write_tsv(
        args.output_dir / "antigen_species_inventory.tsv",
        [{"antigen_species": key, "unique_cdr3": value}
         for key, value in sorted(all_species.items(), key=lambda item: (-item[1], item[0]))],
        ["antigen_species", "unique_cdr3"],
    )
    if args.inventory_only:
        report = {
            "status": "inventory_complete",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "exact_context": list(TARGET_CONTEXT),
            "unique_antigen_species": len(all_species),
            "retained_unique_cdr3": len(records),
            "motif_inventory": motif_report,
            "vdjdb_join_and_quality": join_report,
            "artifact_sha256": sha256_file(args.output_dir / "antigen_species_inventory.tsv"),
        }
        stable_json(args.output_dir / "INVENTORY.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.viral_species_list is None:
        raise ValueError("--viral-species-list is required outside inventory-only mode.")
    viral_species_values = read_species_list(args.viral_species_list)
    ranking, inventory = rank_epitopes(records, set(viral_species_values))
    write_tsv(args.output_dir / "antigen_species_classification.tsv", inventory, list(inventory[0]))
    write_tsv(args.output_dir / "viral_epitope_candidates.tsv", ranking, list(ranking[0]))
    selected, gate = choose_top10(ranking, args.requested_min_positives)
    write_tsv(args.output_dir / "selected_top10.tsv", selected, list(selected[0]))

    viral_records = [record for record in records
                     if record["antigen_species"] in set(viral_species_values)]
    datasets = []
    master: dict[str, dict] = {}
    for selected_row in selected:
        epitope = selected_row["epitope"]
        positives = [record for record in viral_records if record["epitope"] == epitope]
        negatives = [record for record in viral_records if record["epitope"] != epitope]
        dataset = write_epitope_dataset(
            args.output_dir, selected_row["rank"], epitope, positives, negatives,
            args.controls_per_positive, args.max_pairs_per_epitope, args.seed,
        )
        datasets.append(dataset["report"])
        for record in dataset["candidates"]:
            master.setdefault(record["cdr3"], record)
    master_rows = []
    for index, cdr3 in enumerate(sorted(master)):
        row = dict(master[cdr3])
        row["master_index"] = index
        master_rows.append(row)
    write_tsv(
        args.output_dir / "master_candidates.tsv", master_rows,
        ["master_index", "cdr3", "epitope", "antigen_species", "mhc_a", "mhc_b",
         "mhc_class", "v_call", "j_call", "reference_id", "vdjdb_score", "length",
         "motif_cids", "source_rows", "donor_count_descriptive_only"],
    )
    manifest = {
        "status": "ready",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "exact_context": {"mhc_a": TARGET_CONTEXT[0], "mhc_b": TARGET_CONTEXT[1],
                          "mhc_class": TARGET_CONTEXT[2]},
        "pooling": "all records pooled; donor metadata descriptive only",
        "viral_species_rule": (
            "Exact, case-sensitive antigen.species membership in the pinned allowlist; no "
            "substring, epitope, antigen-gene, or model-score inference."
        ),
        "viral_antigen_species_exact": viral_species_values,
        "support_gate": gate,
        "selected": datasets,
        "master_candidate_cdr3": len(master_rows),
        "motif_inventory": motif_report,
        "vdjdb_join_and_quality": join_report,
        "sources": {
            "vdjdb": {"path": str(args.vdjdb_tsv), "sha256": sha256_file(args.vdjdb_tsv)},
            "cluster_members": {"path": str(args.cluster_members),
                                "sha256": sha256_file(args.cluster_members)},
            "motif_pwms": {"path": str(args.motif_pwms),
                            "sha256": sha256_file(args.motif_pwms)},
            "viral_species_list": {"path": str(args.viral_species_list),
                                   "sha256": sha256_file(args.viral_species_list)},
            "benchmark_manifests": benchmark_hashes,
        },
        "artifacts": {
            "selected_top10_sha256": sha256_file(args.output_dir / "selected_top10.tsv"),
            "candidate_ranking_sha256": sha256_file(
                args.output_dir / "viral_epitope_candidates.tsv"
            ),
            "master_candidates_sha256": sha256_file(args.output_dir / "master_candidates.tsv"),
        },
        "official_motif_rule": (
            "Exact sequence+peptide+antigen.species membership in official cluster_members.txt "
            "after exact HomoSapiens/TRB/HLA-A*02:01/B2M/MHCI filtering."
        ),
        "circularity_caveat": (
            "Official motif membership is sequence-cluster-conditioned. This is an exploratory "
            "within-VDJdb motif/epitope-organization analysis, not evidence independent of "
            "sequence motif or composition."
        ),
        "embedding_extraction_performed": False,
    }
    stable_json(args.output_dir / "PREFLIGHT.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
