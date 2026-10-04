"""Prepare a deterministic, parent-disjoint single-amino-acid challenge.

Only locked benchmark *test* clonotypes are eligible parents.  The generated
mutants are scoring inputs, never additional training observations for an
IRRM encoder.  Probe folds are assigned at the parent level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parent_folds(parent_ids: np.ndarray, seed: int) -> dict[str, str]:
    ids = np.asarray(sorted(map(str, parent_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = ids[rng.permutation(len(ids))]
    n_train = int(np.floor(0.6 * len(ids)))
    n_val = int(np.floor(0.2 * len(ids)))
    if min(n_train, n_val, len(ids) - n_train - n_val) < 1:
        raise ValueError("At least five parents are required for 60/20/20 probe folds.")
    return {
        str(identifier): ("train" if i < n_train else "val" if i < n_train + n_val else "test")
        for i, identifier in enumerate(shuffled)
    }


def choose_parents(table: pd.DataFrame, indices: np.ndarray, limit: int, seed: int) -> pd.DataFrame:
    eligible = table.iloc[indices].copy()
    eligible = eligible[eligible["junction_aa"].astype(str).str.len() >= 4]
    eligible["length"] = eligible["junction_aa"].astype(str).str.len()
    if eligible.empty:
        raise ValueError("No held-out parent has a mutable internal residue.")
    # Proportional, deterministic length-stratified sampling prevents the most
    # common CDR3 length from monopolizing a bounded challenge.
    if len(eligible) > limit:
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        groups = list(eligible.groupby("length", sort=True))
        raw = {length: limit * len(group) / len(eligible) for length, group in groups}
        quotas = {length: int(np.floor(value)) for length, value in raw.items()}
        remainder_order = sorted(
            groups, key=lambda item: (-(raw[item[0]] - quotas[item[0]]), -len(item[1]), item[0])
        )
        for length, _group in remainder_order[: limit - sum(quotas.values())]:
            quotas[length] += 1
        for length, group in groups:
            take = min(quotas[length], len(group))
            if take:
                selected.extend(rng.choice(group.index.to_numpy(), size=take, replace=False).tolist())
        eligible = eligible.loc[sorted(selected)]
        if len(eligible) != limit:
            raise AssertionError(f"Length-stratified sampler selected {len(eligible)} parents, expected {limit}.")
    return eligible.sort_values("row_index", kind="mergesort").reset_index(drop=True)


def generate_mutants(
    parents: pd.DataFrame,
    benchmark_sequences: set[str],
    variants_per_parent: int,
    seed: int,
) -> pd.DataFrame:
    if variants_per_parent < 2:
        raise ValueError("At least two mutants per parent are required for within-parent ranking.")
    used = set(benchmark_sequences)
    rows: list[dict] = []
    for row in parents.to_dict(orient="records"):
        parent_id = f"parent-{int(row['row_index']):06d}"
        sequence = str(row["junction_aa"])
        candidates = [(pos, aa) for pos in range(1, len(sequence) - 1) for aa in AA if aa != sequence[pos]]
        rng = np.random.default_rng(seed + int(row["row_index"]) * 104729)
        accepted = 0
        for candidate_index in rng.permutation(len(candidates)):
            position, new_aa = candidates[int(candidate_index)]
            mutant = sequence[:position] + new_aa + sequence[position + 1:]
            if mutant in used:
                continue
            used.add(mutant)
            mutant_id = f"{parent_id}-m{accepted + 1:02d}"
            rows.append({
                "clone_id": mutant_id,
                "parent_id": parent_id,
                "parent_row_index": int(row["row_index"]),
                "parent_sequence": sequence,
                "junction_aa": mutant,
                "mutation_position_zero_based": int(position),
                "mutation_position_fraction": float(position / (len(sequence) - 1)),
                "from_aa": sequence[position],
                "to_aa": new_aa,
                "length": len(sequence),
                "v_call": str(row.get("v_call", "")),
                "j_call": str(row.get("j_call", "")),
                "locus": "TRB",
            })
            accepted += 1
            if accepted == variants_per_parent:
                break
        if accepted != variants_per_parent:
            raise ValueError(f"Could not generate {variants_per_parent} unique mutants for {parent_id}.")
    result = pd.DataFrame(rows)
    if result["junction_aa"].duplicated().any() or set(result["junction_aa"]) & benchmark_sequences:
        raise AssertionError("Generated mutants are not unique and benchmark-disjoint.")
    mismatch = [
        sum(a != b for a, b in zip(row.parent_sequence, row.junction_aa))
        for row in result.itertuples(index=False)
    ]
    if set(mismatch) != {1}:
        raise AssertionError("A generated variant is not exactly one substitution from its parent.")
    return result


def prepare(data_dir: Path, output_dir: Path, max_parents: int, variants: int, seed: int) -> dict:
    # Keep deterministic generation utilities importable in lightweight local
    # test environments that do not install the model's PyTorch dependency.
    from rtp_codec.data.multitask import load_prepared_benchmark, select_split_indices
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    table, embeddings = load_prepared_benchmark(data_dir)
    test_indices = select_split_indices(table, data_dir, "test")
    parents = choose_parents(table, test_indices, max_parents, seed)
    folds = parent_folds(np.array([f"parent-{int(i):06d}" for i in parents["row_index"]]), seed)
    parents["parent_id"] = [f"parent-{int(i):06d}" for i in parents["row_index"]]
    parents["probe_split"] = parents["parent_id"].map(folds)
    mutants = generate_mutants(parents, set(table["junction_aa"].astype(str)), variants, seed)
    mutants["probe_split"] = mutants["parent_id"].map(folds)
    if mutants["probe_split"].isna().any():
        raise AssertionError("A mutant lacks a parent-derived probe split.")

    parent_target_path = output_dir / "parent_tcremp.npy"
    np.save(parent_target_path, np.asarray(embeddings[parents["row_index"].to_numpy(int)], dtype=np.float32))
    parents_path = output_dir / "parents.tsv"
    mutants_path = output_dir / "mutants.tsv"
    scoring_path = output_dir / "mutants_airr.tsv"
    parents.to_csv(parents_path, sep="\t", index=False, lineterminator="\n")
    mutants.to_csv(mutants_path, sep="\t", index=False, lineterminator="\n")
    mutants[["clone_id", "junction_aa", "v_call", "j_call", "locus"]].to_csv(
        scoring_path, sep="\t", index=False, lineterminator="\n"
    )
    split_counts = mutants.groupby("probe_split")["parent_id"].nunique().to_dict()
    report = {
        "status": "ready_for_independent_target_scoring",
        "seed": seed,
        "source_split": "locked benchmark test only",
        "parents": int(len(parents)),
        "mutants": int(len(mutants)),
        "variants_per_parent": variants,
        "all_pairs_edit_distance": 1,
        "parent_disjoint_probe_folds": True,
        "probe_parent_counts": {str(key): int(value) for key, value in split_counts.items()},
        "benchmark_mutant_overlap": 0,
        "tcremp_shape": [int(len(parents)), int(embeddings.shape[1])],
        "sources": {
            "dataset": {"path": str(data_dir / "dataset.parquet"), "sha256": sha256(data_dir / "dataset.parquet")},
            "embeddings": {"path": str(data_dir / "embeddings.npy"), "sha256": sha256(data_dir / "embeddings.npy")},
            "test_manifest": {"path": str(data_dir / "manifests/test.tsv"), "sha256": sha256(data_dir / "manifests/test.tsv")},
        },
        "artifacts": {
            path.name: sha256(path) for path in (parents_path, mutants_path, scoring_path, parent_target_path)
        },
    }
    stable_json(output_dir / "PREPARED.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parents", type=int, default=2000)
    parser.add_argument("--variants-per-parent", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(args.data_dir, args.output_dir, args.max_parents, args.variants_per_parent, args.seed)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
