"""Prepare a leakage-controlled, per-clonotype REDCEA VDJdb benchmark.

The REDCEA member table is used only as a membership/label source.  Its UMAP
coordinates and cluster identifiers are retained for provenance but never used
as model features or evaluation targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from benchmark.prepare_vdjdb_epitope_cohort import read_benchmark_sequences


AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_COLUMNS = {
    "species", "antigen.epitope", "antigen.gene", "antigen.species",
    "mhc.a", "mhc.b", "mhc.class", "gene", "cdr3aa", "cid", "csz",
    "v.segm", "j.segm",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_aa(value: object, max_length: int) -> str | None:
    sequence = str(value).strip().upper()
    if not sequence or len(sequence) > max_length or any(aa not in AA for aa in sequence):
        return None
    return sequence


def prepare_cohort(
    members: pd.DataFrame,
    benchmark_sequences: dict[str, set[str]],
    min_clonotypes: int,
    max_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    missing = REQUIRED_COLUMNS.difference(members.columns)
    if missing:
        raise ValueError(f"REDCEA member table is missing columns: {sorted(missing)}")
    counts: Counter[str] = Counter(source_rows=len(members))
    rows_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for raw in members.to_dict(orient="records"):
        if str(raw["gene"]).strip().upper() != "TRB":
            counts["non_trb_rows"] += 1
            continue
        sequence = canonical_aa(raw["cdr3aa"], max_length=max_length)
        if sequence is None:
            counts["invalid_sequence_rows"] += 1
            continue
        label = str(raw["antigen.epitope"]).strip().upper()
        if not label:
            counts["empty_epitope_rows"] += 1
            continue
        rows_by_sequence[sequence].append({
            "cdr3": sequence,
            "label": label,
            "species": str(raw["species"]).strip(),
            "antigen_gene": str(raw["antigen.gene"]).strip(),
            "antigen_species": str(raw["antigen.species"]).strip(),
            "mhc_a": str(raw["mhc.a"]).strip(),
            "mhc_b": str(raw["mhc.b"]).strip(),
            "mhc_class": str(raw["mhc.class"]).strip(),
            "v_call": str(raw["v.segm"]).strip(),
            "j_call": str(raw["j.segm"]).strip(),
            "cid": str(raw["cid"]).strip(),
            "csz": int(float(raw["csz"])),
            "length": len(sequence),
        })

    retained: list[dict] = []
    overlap_by_split: Counter[str] = Counter()
    for sequence, source_rows in sorted(rows_by_sequence.items()):
        labels = {row["label"] for row in source_rows}
        if len(labels) != 1:
            counts["ambiguous_multilabel_cdr3"] += 1
            continue
        overlapping = [
            split for split, sequences in benchmark_sequences.items() if sequence in sequences
        ]
        if overlapping:
            counts["benchmark_overlap_cdr3"] += 1
            overlap_by_split.update(overlapping)
            continue
        representative = min(
            source_rows,
            key=lambda row: (
                row["label"], row["species"], row["cid"], row["v_call"], row["j_call"]
            ),
        ).copy()
        representative["source_rows"] = len(source_rows)
        representative["source_cids"] = "|".join(sorted({row["cid"] for row in source_rows}))
        retained.append(representative)

    provisional = pd.DataFrame(retained)
    support = provisional.groupby("label")["cdr3"].nunique().sort_index()
    eligible_labels = set(support[support >= min_clonotypes].index)
    cohort = provisional[provisional["label"].isin(eligible_labels)].copy()
    cohort.sort_values(["label", "cdr3"], inplace=True, kind="mergesort")
    cohort.reset_index(drop=True, inplace=True)
    if cohort.empty or cohort["cdr3"].duplicated().any():
        raise ValueError("Prepared cohort is empty or contains duplicate CDR3 sequences.")
    if (cohort.groupby("label").size() < min_clonotypes).any():
        raise AssertionError("A retained label is below the preregistered support threshold.")

    summary = (
        cohort.groupby("label", sort=True)
        .agg(
            unique_cdr3=("cdr3", "size"),
            redcea_clusters=("cid", "nunique"),
            species=("species", lambda x: "|".join(sorted(set(x)))),
            antigen_species=("antigen_species", lambda x: "|".join(sorted(set(x)))),
            length_min=("length", "min"),
            length_median=("length", "median"),
            length_max=("length", "max"),
        )
        .reset_index()
    )
    report = {
        "filter_counts": dict(sorted(counts.items())),
        "benchmark_overlap_by_split": dict(sorted(overlap_by_split.items())),
        "labels_before_support_gate": int(len(support)),
        "labels_after_support_gate": int(cohort["label"].nunique()),
        "clonotypes_after_support_gate": int(len(cohort)),
        "min_clonotypes_per_epitope": int(min_clonotypes),
        "support_before_gate": {str(k): int(v) for k, v in support.items()},
    }
    return cohort, summary, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-members", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-clonotypes", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--source-git-blob", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_clonotypes < 11:
        raise ValueError("At least 11 clonotypes are required for precision@10.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    benchmark_sequences, benchmark_hashes = read_benchmark_sequences(args.benchmark_dir)
    members = pd.read_csv(args.cluster_members, sep="\t", dtype=str, keep_default_na=False)
    cohort, summary, report = prepare_cohort(
        members, benchmark_sequences, args.min_clonotypes, args.max_length
    )
    cohort_path = args.output_dir / "cohort.tsv"
    summary_path = args.output_dir / "epitope_support.tsv"
    cohort.to_csv(cohort_path, sep="\t", index=False, lineterminator="\n")
    summary.to_csv(summary_path, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "status": "accepted_for_frozen_embedding_evaluation",
        "study": "REDCEA motif-member per-clonotype epitope geometry",
        "source": {
            "path": str(args.cluster_members),
            "sha256": sha256(args.cluster_members),
            "git_commit": args.source_git_commit,
            "git_blob": args.source_git_blob,
            "expected_repository_path": "results/redcea/cluster_members_TRB.txt",
        },
        "cohort": {
            "path": str(cohort_path), "sha256": sha256(cohort_path),
            "rows": len(cohort), "labels": int(cohort["label"].nunique()),
            "species": sorted(cohort["species"].unique().tolist()),
            "minimum_label_support": args.min_clonotypes,
        },
        "epitope_support": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "benchmark_manifest_sha256": benchmark_hashes,
        "filters": report,
        "protocol": {
            "unit": "unique exact TRB CDR3 amino-acid clonotype",
            "label": "exact antigen.epitope from REDCEA member table",
            "donors_and_publications": "pooled; not used for filtering or splitting",
            "ambiguous_multilabel_cdr3": "removed entirely",
            "benchmark_train_val_test_overlap": "removed entirely",
            "redcea_cid": "provenance only; not an evaluation unit or feature",
            "redcea_x_y": "excluded; UMAP visualization coordinates are never features",
        },
    }
    stable_json(args.output_dir / "PREFLIGHT.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
