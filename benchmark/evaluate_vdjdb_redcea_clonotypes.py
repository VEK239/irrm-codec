"""Evaluate frozen DATA-ANCHOR latents on REDCEA motif-member clonotypes.

Every retained clonotype is a query.  Its deterministic candidate gallery has
up to 20 same-epitope clonotypes and four times as many other-epitope controls,
selected without model embeddings.  This keeps the all-clonotype evaluation
tractable while fixing positive prevalence and using identical comparisons for
all seven frozen encoders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    precision_recall_fscore_support,
    silhouette_score,
    v_measure_score,
)

PRIMARY_MODELS = ("r", "t", "p", "rt", "rp", "tp", "rtp")
METRIC_COLUMNS = (
    "precision_at_1", "precision_at_5", "precision_at_10", "average_precision",
    "within_between_ratio", "between_minus_within", "nearest_margin",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_finite_tree(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"Non-finite result at {path}: {value}")


def build_candidate_gallery(
    cohort: pd.DataFrame, max_positives: int, negative_ratio: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels_text = cohort["label"].astype(str).to_numpy()
    label_names = np.array(sorted(set(labels_text)), dtype=object)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    labels = np.array([label_to_id[value] for value in labels_text], dtype=np.int32)
    species = cohort["species"].fillna("").astype(str).to_numpy()
    lengths = cohort["length"].to_numpy(dtype=np.int16)
    v_call = cohort["v_call"].fillna("").astype(str).to_numpy()
    j_call = cohort["j_call"].fillna("").astype(str).to_numpy()
    by_label = {label: np.flatnonzero(labels == label) for label in range(len(label_names))}
    by_label_species_length: dict[tuple[int, str, int], np.ndarray] = {}
    by_label_species_length_vj: dict[tuple[int, str, int, str, str], np.ndarray] = {}
    by_label_species_length_v: dict[tuple[int, str, int, str], np.ndarray] = {}
    by_label_species_length_j: dict[tuple[int, str, int, str], np.ndarray] = {}
    by_label_species: dict[tuple[int, str], dict[int, np.ndarray]] = {}
    by_label_length: dict[tuple[int, int], np.ndarray] = {}
    for label, indices in by_label.items():
        for sp in sorted(set(species[indices])):
            sp_indices = indices[species[indices] == sp]
            lengths_map = {}
            for length in sorted(set(lengths[sp_indices])):
                values = sp_indices[lengths[sp_indices] == length]
                by_label_species_length[(label, sp, int(length))] = values
                for v_value in sorted(set(v_call[values])):
                    by_label_species_length_v[(label, sp, int(length), v_value)] = values[
                        v_call[values] == v_value
                    ]
                for j_value in sorted(set(j_call[values])):
                    by_label_species_length_j[(label, sp, int(length), j_value)] = values[
                        j_call[values] == j_value
                    ]
                for v_value, j_value in sorted(set(zip(v_call[values], j_call[values]))):
                    by_label_species_length_vj[(
                        label, sp, int(length), v_value, j_value
                    )] = values[(v_call[values] == v_value) & (j_call[values] == j_value)]
                lengths_map[int(length)] = values
            by_label_species[(label, sp)] = lengths_map
        for length in sorted(set(lengths[indices])):
            by_label_length[(label, int(length))] = indices[lengths[indices] == length]

    width = max_positives * (negative_ratio + 1)
    candidates = np.full((len(cohort), width), -1, dtype=np.int32)
    relevant = np.zeros((len(cohort), width), dtype=bool)
    tiers = np.full((len(cohort), width), -1, dtype=np.int8)
    tier_counts: dict[str, int] = {}

    def choose_group(query: int, negative_label: int) -> tuple[np.ndarray, int]:
        exact_vj = by_label_species_length_vj.get((
            negative_label, species[query], int(lengths[query]), v_call[query], j_call[query]
        ))
        if exact_vj is not None:
            return exact_vj, 0
        one_gene = [
            group for group in (
                by_label_species_length_v.get((
                    negative_label, species[query], int(lengths[query]), v_call[query]
                )),
                by_label_species_length_j.get((
                    negative_label, species[query], int(lengths[query]), j_call[query]
                )),
            ) if group is not None
        ]
        if one_gene:
            return min(one_gene, key=len), 1
        exact = by_label_species_length.get(
            (negative_label, species[query], int(lengths[query]))
        )
        if exact is not None:
            return exact, 2
        same_species = by_label_species.get((negative_label, species[query]))
        if same_species:
            nearest_length = min(
                same_species, key=lambda value: (abs(value - int(lengths[query])), value)
            )
            return same_species[nearest_length], 3
        available_lengths = sorted(
            length for (label, length) in by_label_length if label == negative_label
        )
        nearest_length = min(
            available_lengths, key=lambda value: (abs(value - int(lengths[query])), value)
        )
        return by_label_length[(negative_label, nearest_length)], 4

    for query in range(len(cohort)):
        rng = np.random.default_rng(seed + query * 1_000_003)
        positive_pool = by_label[int(labels[query])]
        positive_pool = positive_pool[positive_pool != query]
        n_positive = min(max_positives, len(positive_pool))
        if n_positive < 1:
            raise ValueError(f"Query {query} has no same-epitope candidate.")
        positives = (
            positive_pool if len(positive_pool) == n_positive
            else positive_pool[np.sort(rng.choice(len(positive_pool), n_positive, replace=False))]
        )
        candidates[query, :n_positive] = positives
        relevant[query, :n_positive] = True
        tiers[query, :n_positive] = 0

        quota = negative_ratio * n_positive
        other_labels = np.delete(np.arange(len(label_names), dtype=np.int32), labels[query])
        selected: list[int] = []
        selected_tiers: list[int] = []
        if quota > len(other_labels):
            raise ValueError("Negative quota exceeds the number of other epitope labels.")
        for negative_label in rng.permutation(other_labels)[:quota]:
            group, tier = choose_group(query, int(negative_label))
            selected.append(int(group[rng.integers(len(group))]))
            selected_tiers.append(tier)
        start = n_positive
        stop = start + quota
        candidates[query, start:stop] = selected
        tiers[query, start:stop] = selected_tiers
        for tier in selected_tiers:
            name = {
                0: "same_species_length_vj", 1: "same_species_length_one_gene",
                2: "same_species_length", 3: "same_species_nearest_length",
                4: "nearest_length_any_species",
            }[tier]
            tier_counts[name] = tier_counts.get(name, 0) + 1
    return candidates, relevant, tiers, {
        "queries": len(cohort), "width": width, "max_positives": max_positives,
        "negative_ratio": negative_ratio, "selection_seed": seed,
        "negative_match_tiers": dict(sorted(tier_counts.items())),
        "label_names": label_names.tolist(),
    }


def candidate_distances(
    features: np.ndarray, candidates: np.ndarray, metric: str, chunk_size: int = 4096,
) -> tuple[np.ndarray, dict]:
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norms == 0):
        raise ValueError("Features are non-finite or contain a zero-norm row.")
    output = np.full(candidates.shape, np.inf, dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        ids = candidates[start:stop]
        valid = ids >= 0
        safe = np.maximum(ids, 0)
        query = values[start:stop, None, :]
        gallery = values[safe]
        if metric == "cosine":
            query = query / norms[start:stop, None, None]
            gallery = gallery / norms[safe, None]
            block = 1.0 - np.sum(query * gallery, axis=2)
        elif metric == "euclidean":
            block = np.linalg.norm(query - gallery, axis=2)
        else:
            raise ValueError(metric)
        output[start:stop][valid] = np.maximum(block[valid], 0.0)
    return output, {
        "latent_norm_min": float(norms.min()), "latent_norm_median": float(np.median(norms)),
        "latent_norm_max": float(norms.max()),
        "euclidean_scale_note": "raw latent Euclidean distance; compare retrieval/effect, not absolute scale",
    }


def _majority_label(labels: np.ndarray, distances: np.ndarray) -> int:
    unique, counts = np.unique(labels, return_counts=True)
    winners = unique[counts == counts.max()]
    if len(winners) == 1:
        return int(winners[0])
    first_rank = {int(label): int(np.flatnonzero(labels == label)[0]) for label in winners}
    return min((int(label) for label in winners), key=lambda value: (first_rank[value], value))


def summarize_candidates(
    distances: np.ndarray,
    candidates: np.ndarray,
    relevant: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = {1: np.empty(len(distances), dtype=np.int32),
                   5: np.empty(len(distances), dtype=np.int32),
                   10: np.empty(len(distances), dtype=np.int32)}
    for query in range(len(distances)):
        valid = candidates[query] >= 0
        order = np.argsort(distances[query, valid], kind="mergesort")
        hits = relevant[query, valid][order]
        ranked_candidates = candidates[query, valid][order]
        ranked_labels = label_ids[ranked_candidates]
        positive_count = int(hits.sum())
        precision_curve = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        within = distances[query, relevant[query]]
        between = distances[query, valid & ~relevant[query]]
        row = {
            "query_index": query, "label": label_names[int(label_ids[query])],
            "positive_candidates": positive_count,
            "negative_candidates": int(len(hits) - positive_count),
            "precision_at_1": float(hits[:1].mean()),
            "precision_at_5": float(hits[:5].mean()),
            "precision_at_10": float(hits[:10].mean()),
            "average_precision": float((precision_curve * hits).sum() / positive_count),
            "within_mean": float(within.mean()), "between_mean": float(between.mean()),
            "within_between_ratio": float(within.mean() / between.mean()),
            "between_minus_within": float(between.mean() - within.mean()),
            "nearest_same": float(within.min()), "nearest_other": float(between.min()),
            "nearest_margin": float(between.min() - within.min()),
            "nearest_same_is_closer": float(within.min() < between.min()),
        }
        rows.append(row)
        for k in predictions:
            predictions[k][query] = _majority_label(ranked_labels[:k], distances[query, valid][order][:k])
    per_query = pd.DataFrame(rows)
    per_label = per_query.groupby("label", sort=True)[list(METRIC_COLUMNS) + [
        "nearest_same_is_closer", "within_mean", "between_mean"
    ]].mean().reset_index()
    summary = {
        "queries": len(per_query),
        "macro_epitope": {column: float(per_label[column].mean()) for column in METRIC_COLUMNS},
        "micro_query": {column: float(per_query[column].mean()) for column in METRIC_COLUMNS},
        "nearest_same_is_closer_macro": float(per_label["nearest_same_is_closer"].mean()),
        "knn_classification": {},
    }
    for k, prediction in predictions.items():
        precision, recall, f1, support = precision_recall_fscore_support(
            label_ids, prediction, labels=np.arange(len(label_names)), zero_division=0
        )
        summary["knn_classification"][f"k_{k}"] = {
            "macro_f1": float(f1_score(label_ids, prediction, average="macro")),
            "weighted_f1": float(f1_score(label_ids, prediction, average="weighted")),
            "balanced_accuracy": float(balanced_accuracy_score(label_ids, prediction)),
        }
        f1_lookup = {label_names[index]: float(f1[index]) for index in range(len(label_names))}
        per_label[f"knn_f1_at_{k}"] = per_label["label"].map(f1_lookup)
    return summary, per_query, per_label


def clustering_metrics(features: np.ndarray, label_ids: np.ndarray, seed: int, sample: int) -> dict:
    values = np.asarray(features, dtype=np.float64)
    unit = values / np.linalg.norm(values, axis=1, keepdims=True)
    n_labels = len(np.unique(label_ids))
    kmeans = MiniBatchKMeans(
        n_clusters=n_labels, random_state=seed, n_init=10, batch_size=4096,
        max_iter=300, reassignment_ratio=0.01,
    ).fit(unit)
    predicted = kmeans.labels_
    sample_size = min(sample, len(unit))
    return {
        "true_label_geometry": {
            "silhouette_cosine": float(silhouette_score(
                unit, label_ids, metric="cosine", sample_size=sample_size, random_state=seed
            )),
            "silhouette_euclidean_unit": float(silhouette_score(
                unit, label_ids, metric="euclidean", sample_size=sample_size, random_state=seed
            )),
            "davies_bouldin_unit_lower_is_better": float(davies_bouldin_score(unit, label_ids)),
            "calinski_harabasz_unit_higher_is_better": float(calinski_harabasz_score(unit, label_ids)),
        },
        "label_free_minibatch_kmeans_k_equals_epitopes": {
            "adjusted_rand": float(adjusted_rand_score(label_ids, predicted)),
            "adjusted_mutual_information": float(adjusted_mutual_info_score(label_ids, predicted)),
            "v_measure": float(v_measure_score(label_ids, predicted)),
            "homogeneity": float(homogeneity_score(label_ids, predicted)),
            "completeness": float(completeness_score(label_ids, predicted)),
            "inertia": float(kmeans.inertia_), "n_init": 10, "seed": seed,
        },
        "interpretation": (
            "Secondary: epitopes can contain multiple REDCEA motifs, so a one-cluster-per-epitope "
            "K-means geometry is not assumed by the primary retrieval/contrast analysis."
        ),
    }


def baseline_features(cohort: pd.DataFrame) -> dict[str, np.ndarray]:
    amino = "ACDEFGHIKLMNPQRSTVWY"
    composition = np.zeros((len(cohort), len(amino)), dtype=np.float32)
    for row, sequence in enumerate(cohort["cdr3"].astype(str)):
        for column, aa in enumerate(amino):
            composition[row, column] = sequence.count(aa) / len(sequence)
    length = cohort["length"].to_numpy(np.float32)[:, None]
    length = (length - length.mean()) / max(float(length.std()), 1e-8)
    basic = np.concatenate((composition, length), axis=1)
    genes = pd.get_dummies(
        cohort[["v_call", "j_call"]].fillna("").astype(str), dtype=np.float32
    ).to_numpy(np.float32)
    return {"aa_composition_length": basic, "aa_composition_length_vj": np.concatenate((basic, genes), axis=1)}


def bootstrap_by_epitope(per_label: pd.DataFrame, replicates: int, seed: int) -> dict:
    labels = sorted(per_label["label"].unique())
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(labels), size=(replicates, len(labels)))
    report = {"replicates": replicates, "unit": "epitope label", "paired_rtp_minus_other": {}}
    for distance in ("cosine", "euclidean"):
        block = per_label[per_label["distance"] == distance]
        arrays = {
            model: block[block["model"] == model].set_index("label").loc[labels]
            for model in PRIMARY_MODELS
        }
        report["paired_rtp_minus_other"][distance] = {}
        for other in PRIMARY_MODELS[:-1]:
            metrics = {}
            for column in METRIC_COLUMNS:
                delta = arrays["rtp"][column].to_numpy() - arrays[other][column].to_numpy()
                boot = delta[samples].mean(axis=1)
                metrics[column] = {
                    "estimate": float(delta.mean()),
                    "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "direction": "lower is better" if column == "within_between_ratio" else "higher is better",
                }
            report["paired_rtp_minus_other"][distance][other] = metrics
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-positive-candidates", type=int, default=20)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--silhouette-sample", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from benchmark.evaluate_vdjdb_frozen import (
        extract_embeddings, parse_models, validate_run_configs,
    )

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight["status"] != "accepted_for_frozen_embedding_evaluation":
        raise ValueError("Cohort preflight is not accepting.")
    if sha256(args.cohort) != preflight["cohort"]["sha256"]:
        raise ValueError("Cohort differs from accepted preflight.")
    cohort = pd.read_csv(args.cohort, sep="\t")
    if cohort["cdr3"].duplicated().any() or (cohort.groupby("label").size() < 20).any():
        raise ValueError("Cohort violates uniqueness or minimum-support rules.")
    models = parse_models(args.model)
    if tuple(sorted(models)) != tuple(sorted(PRIMARY_MODELS)):
        raise ValueError("The full seven-condition factorial is required.")
    config, model_report = validate_run_configs(models)
    embeddings, extraction = extract_embeddings(
        cohort, models, config, args.tokenizer, args.batch_size, args.output_dir
    )
    candidates, relevant, tiers, gallery_report = build_candidate_gallery(
        cohort, args.max_positive_candidates, args.negative_ratio, args.seed
    )
    gallery_path = args.output_dir / "candidate_gallery.npz"
    np.savez_compressed(gallery_path, candidates=candidates, relevant=relevant, match_tier=tiers)
    label_names = gallery_report["label_names"]
    label_to_id = {label: index for index, label in enumerate(label_names)}
    label_ids = np.array([label_to_id[value] for value in cohort["label"].astype(str)], dtype=np.int32)

    summaries = {}
    clustering = {}
    per_labels = []
    for model in PRIMARY_MODELS:
        summaries[model] = {}
        for metric in ("cosine", "euclidean"):
            distances, norm_report = candidate_distances(embeddings[model], candidates, metric)
            summary, per_query, per_label = summarize_candidates(
                distances, candidates, relevant, label_ids, label_names
            )
            summary["embedding_norms"] = norm_report
            summaries[model][metric] = summary
            per_query.to_parquet(args.output_dir / f"per_query_{model}_{metric}.parquet", index=False)
            per_label.insert(0, "model", model)
            per_label.insert(1, "distance", metric)
            per_labels.append(per_label)
        clustering[model] = clustering_metrics(
            embeddings[model], label_ids, args.seed, args.silhouette_sample
        )

    controls = {}
    for name, features in baseline_features(cohort).items():
        controls[name] = {"retrieval": {}, "clustering": clustering_metrics(
            features, label_ids, args.seed, args.silhouette_sample
        )}
        for metric in ("cosine", "euclidean"):
            distances, _ = candidate_distances(features, candidates, metric)
            summary, _, _ = summarize_candidates(
                distances, candidates, relevant, label_ids, label_names
            )
            controls[name]["retrieval"][metric] = summary

    per_label_table = pd.concat(per_labels, ignore_index=True)
    per_label_path = args.output_dir / "per_epitope_metrics.tsv"
    per_label_table.to_csv(per_label_path, sep="\t", index=False, lineterminator="\n")
    bootstrap = bootstrap_by_epitope(per_label_table, args.bootstrap, args.seed)
    chance = {
        "candidate_positive_fraction": float(relevant.sum() / (candidates >= 0).sum()),
        "expected_precision_at_k_under_random_ranking": float(
            relevant.sum() / (candidates >= 0).sum()
        ),
        "note": "Candidate prevalence is fixed at 1:4 positive:negative; MAP chance varies slightly with gallery size.",
    }
    assert_finite_tree(summaries, "summaries")
    assert_finite_tree(clustering, "clustering")
    assert_finite_tree(controls, "controls")
    assert_finite_tree(bootstrap, "bootstrap")
    stable_json(args.output_dir / "metrics.json", summaries)
    stable_json(args.output_dir / "clustering.json", clustering)
    stable_json(args.output_dir / "controls.json", controls)
    stable_json(args.output_dir / "bootstrap.json", bootstrap)
    artifact_hashes = {
        path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()
    }
    result = {
        "status": "complete", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "study": "REDCEA motif-member per-clonotype geometry across seven matched DATA-ANCHOR encoders",
        "cohort": {"path": str(args.cohort), "sha256": sha256(args.cohort),
                   "clonotypes": len(cohort), "epitopes": int(cohort["label"].nunique())},
        "models": model_report, "extraction": extraction,
        "candidate_gallery": {**gallery_report, "path": str(gallery_path), "sha256": sha256(gallery_path)},
        "chance_control": chance,
        "primary_metrics": ["macro P@1/5/10", "macro MAP over fixed candidate gallery",
                            "within/between distance ratio", "nearest-same margin", "kNN macro-F1"],
        "secondary_metrics": ["silhouette", "Davies-Bouldin", "Calinski-Harabasz",
                              "MiniBatchKMeans ARI/AMI/V-measure"],
        "selection_statement": "No VDJdb outcome selected a checkpoint, model, gallery member, or support threshold.",
        "circularity_caveat": (
            "REDCEA membership is sequence/TCRemP-cluster-conditioned. This measures organization of "
            "REDCEA-supported epitope labels, not independent binding prediction beyond sequence motifs."
        ),
        "single_seed_caveat": (
            "Paired intervals quantify cohort/epitope uncertainty for fixed seed-42 models, not training-seed variability."
        ),
        "artifact_sha256": artifact_hashes, "nonfinite_scan": "passed",
    }
    stable_json(args.output_dir / "RESULTS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
