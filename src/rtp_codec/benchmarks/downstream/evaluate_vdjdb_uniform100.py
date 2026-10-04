"""Uniform 100-within / 100-between cosine-distance evaluation for Section 7.

Only epitopes with at least 101 unique clonotypes are retained.  For every
retained query, 100 distinct same-epitope comparators and 100 distinct
different-epitope comparators are drawn without replacement.  The candidate
pairs are deterministic and shared by all representations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_WITHIN = 100
N_BETWEEN = 100
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--native-evaluation", type=Path, required=True)
    parser.add_argument("--external-representations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def build_pairs(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = pd.Series(labels).value_counts()
    eligible_labels = np.array(sorted(counts[counts >= N_WITHIN + 1].index), dtype=object)
    keep = np.isin(labels, eligible_labels)
    original_indices = np.flatnonzero(keep)
    retained_labels = labels[keep]
    groups = {label: np.flatnonzero(retained_labels == label) for label in eligible_labels}
    all_indices = np.arange(len(retained_labels), dtype=np.int32)
    within = np.empty((len(retained_labels), N_WITHIN), dtype=np.int32)
    between = np.empty((len(retained_labels), N_BETWEEN), dtype=np.int32)

    for query in range(len(retained_labels)):
        rng = np.random.default_rng(seed + int(original_indices[query]) * 1_000_003)
        label = retained_labels[query]
        same = groups[label]
        same = same[same != query]
        within[query] = rng.choice(same, size=N_WITHIN, replace=False)
        other = all_indices[retained_labels != label]
        between[query] = rng.choice(other, size=N_BETWEEN, replace=False)
    return original_indices, within, between


def cosine_summary(
    matrix: np.ndarray, within: np.ndarray, between: np.ndarray,
    labels: np.ndarray, batch_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(values).all():
        raise ValueError("Embedding matrix contains a zero-norm or non-finite vector.")
    unit = values / norms
    within_mean = np.empty(len(unit), dtype=np.float32)
    between_mean = np.empty(len(unit), dtype=np.float32)
    for start in range(0, len(unit), batch_size):
        stop = min(start + batch_size, len(unit))
        query = unit[start:stop]
        within_mean[start:stop] = (1.0 - np.sum(query[:, None] * unit[within[start:stop]], axis=2)).mean(axis=1)
        between_mean[start:stop] = (1.0 - np.sum(query[:, None] * unit[between[start:stop]], axis=2)).mean(axis=1)
    per_query = pd.DataFrame({
        "epitope": labels,
        "within_mean": within_mean,
        "between_mean": between_mean,
        "within_between_ratio": within_mean / between_mean,
        "between_minus_within": between_mean - within_mean,
    })
    per_epitope = per_query.groupby("epitope", sort=True).agg(
        queries=("epitope", "size"),
        within_mean=("within_mean", "mean"),
        between_mean=("between_mean", "mean"),
        within_between_ratio=("within_between_ratio", "mean"),
        between_minus_within=("between_minus_within", "mean"),
    ).reset_index()
    return per_query, per_epitope


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    cohort = pd.read_csv(args.cohort, sep="\t")
    labels = cohort["label"].astype(str).to_numpy()
    original_indices, within, between = build_pairs(labels, args.seed)
    retained = cohort.iloc[original_indices].reset_index(drop=True)
    np.savez_compressed(args.output_dir / "uniform_pairs.npz", within=within, between=between, original_indices=original_indices)

    native = {
        "RTP": args.native_evaluation / "embeddings_rtp.npy",
        "RP": args.native_evaluation / "embeddings_rp.npy",
        "T": args.native_evaluation / "embeddings_t.npy",
    }
    catalog = json.loads((args.external_representations / "representations.json").read_text(encoding="utf-8"))
    external = {
        "ESM2-35M": Path(catalog["esm2_35m"]["path"]),
        "TCR-BERT": Path(catalog["tcr_bert"]["path"]),
        "SCEPTR (CDR3-only)": Path(catalog["sceptr_cdr3"]["path"]),
    }
    summary_rows = []
    all_epitopes = []
    for name, path in {**native, **external}.items():
        matrix = np.load(path, mmap_mode="r")[original_indices]
        per_query, per_epitope = cosine_summary(matrix, within, between, retained["label"].astype(str).to_numpy())
        per_query.insert(0, "model", name)
        per_query.to_parquet(args.output_dir / f"per_query_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.parquet", index=False)
        per_epitope.insert(0, "model", name)
        all_epitopes.append(per_epitope)
        summary_rows.append({
            "model": name,
            "macro_within_between_ratio": float(per_epitope["within_between_ratio"].mean()),
            "macro_between_minus_within": float(per_epitope["between_minus_within"].mean()),
            "micro_within_between_ratio": float(per_query["within_mean"].mean() / per_query["between_mean"].mean()),
            "queries": len(per_query),
            "epitopes": len(per_epitope),
        })
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.tsv", sep="\t", index=False, lineterminator="\n")
    pd.concat(all_epitopes, ignore_index=True).to_csv(args.output_dir / "per_epitope.tsv", sep="\t", index=False, lineterminator="\n")
    result = {
        "status": "complete",
        "protocol": {
            "metric": "cosine distance",
            "within_candidates_per_query": N_WITHIN,
            "between_candidates_per_query": N_BETWEEN,
            "within_sampling": "uniform without replacement from same epitope, excluding query",
            "between_sampling": "uniform without replacement from all different-epitope clonotypes",
            "pair_seed": args.seed,
            "query_seed": "seed + original_cohort_index * 1,000,003",
        },
        "cohort": {
            "source": str(args.cohort), "source_sha256": sha256(args.cohort),
            "retained_clonotypes": len(retained), "retained_epitopes": int(retained["label"].nunique()),
            "minimum_epitope_size": N_WITHIN + 1,
        },
        "models": list(native) + list(external),
        "artifacts": {name: sha256(args.output_dir / name) for name in ("uniform_pairs.npz", "summary.tsv", "per_epitope.tsv")},
    }
    (args.output_dir / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
