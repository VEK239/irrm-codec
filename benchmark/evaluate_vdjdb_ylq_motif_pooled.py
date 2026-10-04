"""CPU-only frozen evaluation on pooled official VDJdb motif-member clonotypes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmark.evaluate_vdjdb_frozen import (
    AA,
    assert_finite_tree,
    extract_embeddings,
    parse_models,
    sha256,
    stable_json,
    validate_run_configs,
)


def vector_distance(left: np.ndarray, right: np.ndarray, metric: str) -> np.ndarray:
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    if metric == "cosine":
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        if np.any(denominator == 0):
            raise ValueError("Zero-norm vector prevents cosine distance.")
        return np.maximum(0.0, 1.0 - np.sum(left * right, axis=1) / denominator)
    if metric == "euclidean":
        return np.linalg.norm(left - right, axis=1)
    raise ValueError(metric)


def primary_distances(array: np.ndarray, pairs: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)
    return vector_distance(array[q], array[p], metric), vector_distance(array[q], array[n], metric)


def pair_metrics(within: np.ndarray, between: np.ndarray) -> dict:
    truth = np.concatenate((np.ones(len(within), dtype=int), np.zeros(len(between), dtype=int)))
    score = -np.concatenate((within, between))
    within_mean = float(within.mean())
    between_mean = float(between.mean())
    if between_mean == 0:
        ratio = 1.0 if within_mean == 0 else float("inf")
    else:
        ratio = within_mean / between_mean
    pooled = math.sqrt((float(within.var()) + float(between.var())) / 2.0)
    separation = (between_mean - within_mean) / pooled if pooled > 0 else 0.0
    return {
        "pairs_per_class": len(within),
        "within_mean": within_mean,
        "between_mean": between_mean,
        "between_minus_within": between_mean - within_mean,
        "within_between_ratio": ratio,
        "standardized_separation": separation,
        "same_label_auroc": float(roc_auc_score(truth, score)),
        "same_label_auprc_balanced": float(average_precision_score(truth, score)),
    }


def pooled_retrieval(array: np.ndarray, candidates: pd.DataFrame, metric: str) -> tuple[dict, pd.DataFrame]:
    positive = candidates["is_ylq"].to_numpy(int).astype(bool)
    cdr3 = candidates["cdr3"].astype(str).to_numpy()
    if metric == "cosine":
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero-norm latent prevents cosine retrieval.")
        prepared = array / norms
    else:
        prepared = array.astype(np.float64, copy=False)
    rows = []
    all_indices = np.arange(len(candidates))
    for query in np.flatnonzero(positive):
        if metric == "cosine":
            distance = np.maximum(0.0, 1.0 - prepared @ prepared[query])
        else:
            distance = np.linalg.norm(prepared - prepared[query], axis=1)
        allowed = all_indices[all_indices != query]
        relevant = positive[allowed]
        total_relevant = int(relevant.sum())
        order = np.argsort(distance[allowed], kind="mergesort")
        ranked = relevant[order]
        hits = ranked[:total_relevant].astype(float)
        precision = np.cumsum(hits) / np.arange(1, total_relevant + 1)
        rows.append({
            "query_index": int(query),
            "query_cdr3": cdr3[query],
            "relevant_pooled": total_relevant,
            "precision_at_1": float(ranked[0]),
            "precision_at_5": float(ranked[:5].mean()),
            "precision_at_10": float(ranked[:10].mean()),
            "ap_at_r": float((precision * hits).sum() / total_relevant),
        })
    table = pd.DataFrame(rows)
    return {
        "eligible_positive_queries": len(table),
        "gallery_excludes_query_only": True,
        "donor_filter": "none_pooled_by_user_protocol",
        "precision_at_1": float(table["precision_at_1"].mean()),
        "precision_at_5": float(table["precision_at_5"].mean()),
        "precision_at_10": float(table["precision_at_10"].mean()),
        "map_at_r": float(table["ap_at_r"].mean()),
    }, table


def bootstrap_clonotypes(
    pair_values: dict[str, tuple[np.ndarray, np.ndarray]],
    pairs: pd.DataFrame,
    retrieval: dict[str, pd.DataFrame],
    replicates: int,
    seed: int,
) -> dict:
    clusters = sorted(set(pairs["query_cdr3"].astype(str)).union(
        *(set(table["query_cdr3"].astype(str)) for table in retrieval.values())))
    cluster_index = {value: index for index, value in enumerate(clusters)}
    pair_cluster = pairs["query_cdr3"].map(cluster_index).to_numpy(int)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(clusters), np.full(len(clusters), 1 / len(clusters)), size=replicates
    )
    values_by_model: dict[str, dict[str, np.ndarray]] = {}
    for name, (within, between) in pair_values.items():
        result = {metric: np.empty(replicates) for metric in (
            "within_between_ratio", "between_minus_within", "auroc", "auprc",
            "precision_at_1", "map_at_r",
        )}
        query_table = retrieval[name]
        query_cluster = query_table["query_cdr3"].map(cluster_index).to_numpy(int)
        query_p1 = query_table["precision_at_1"].to_numpy(float)
        query_ap = query_table["ap_at_r"].to_numpy(float)
        truth = np.concatenate((np.ones(len(within), dtype=int), np.zeros(len(between), dtype=int)))
        score = -np.concatenate((within, between))
        for replicate, cluster_weight in enumerate(weights):
            pair_weight = cluster_weight[pair_cluster].astype(float)
            within_mean = float(np.average(within, weights=pair_weight))
            between_mean = float(np.average(between, weights=pair_weight))
            result["within_between_ratio"][replicate] = within_mean / between_mean
            result["between_minus_within"][replicate] = between_mean - within_mean
            sample_weight = np.concatenate((pair_weight, pair_weight))
            result["auroc"][replicate] = roc_auc_score(truth, score, sample_weight=sample_weight)
            result["auprc"][replicate] = average_precision_score(
                truth, score, sample_weight=sample_weight
            )
            query_weight = cluster_weight[query_cluster].astype(float)
            result["precision_at_1"][replicate] = np.average(query_p1, weights=query_weight)
            result["map_at_r"][replicate] = np.average(query_ap, weights=query_weight)
        values_by_model[name] = result

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    report = {
        "replicates": replicates,
        "cluster": "query exact CDR3 clonotype",
        "paired_across_frozen_models": True,
        "models": {
            name: {metric: {"ci95": interval(values)} for metric, values in model_values.items()}
            for name, model_values in values_by_model.items()
        },
        "paired_differences_rtp_minus_other": {},
    }
    for other in ("r", "p", "rp"):
        report["paired_differences_rtp_minus_other"][other] = {
            metric: {
                "ci95": interval(values_by_model["rtp"][metric] - values_by_model[other][metric]),
                "direction": "negative_favors_rtp" if metric == "within_between_ratio"
                    else "positive_favors_rtp",
            }
            for metric in values_by_model["rtp"]
        }
    return report


def composition_features(candidates: pd.DataFrame) -> np.ndarray:
    result = np.zeros((len(candidates), len(AA)), dtype=float)
    for row, sequence in enumerate(candidates["cdr3"].astype(str)):
        for column, aa in enumerate(AA):
            result[row, column] = sequence.count(aa) / len(sequence)
    return result


def vj_pair_values(candidates: pd.DataFrame, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    v = candidates["v_call"].fillna("").astype(str).to_numpy()
    j = candidates["j_call"].fillna("").astype(str).to_numpy()
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)

    def value(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        v_mismatch = (v[left] == "") | (v[right] == "") | (v[left] != v[right])
        j_mismatch = (j[left] == "") | (j[right] == "") | (j[left] != j[right])
        return (v_mismatch.astype(float) + j_mismatch.astype(float)) / 2.0

    return value(q, p), value(q, n)


def shuffled_label_control(
    pair_values: dict[str, tuple[np.ndarray, np.ndarray]],
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    permutations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed + 30000)
    labels = candidates["is_ylq"].to_numpy(int).astype(bool)
    lengths = candidates["length"].to_numpy(int)
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)
    groups = [np.flatnonzero(lengths == length) for length in sorted(set(lengths))]
    observed = {
        name: float(between.mean() - within.mean())
        for name, (within, between) in pair_values.items()
    }
    null = {name: np.full(permutations, np.nan) for name in pair_values}
    valid_pair_counts = []
    for replicate in range(permutations):
        shuffled = labels.copy()
        for group in groups:
            shuffled[group] = rng.permutation(shuffled[group])
        same_mask = shuffled[q] & shuffled[p]
        different_mask = shuffled[q] ^ shuffled[n]
        valid_pair_counts.append((int(same_mask.sum()), int(different_mask.sum())))
        if not same_mask.any() or not different_mask.any():
            continue
        for name, (within, between) in pair_values.items():
            null[name][replicate] = float(
                between[different_mask].mean() - within[same_mask].mean()
            )
    models = {}
    for name, values in null.items():
        finite = values[np.isfinite(values)]
        if len(finite) < permutations * 0.95:
            raise ValueError(f"Too few valid shuffled-label replicates for {name}: {len(finite)}")
        models[name] = {
            "observed_between_minus_within": observed[name],
            "null_ci95": [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))],
            "one_sided_p": float((1 + np.sum(finite >= observed[name])) / (1 + len(finite))),
        }
    return {
        "permutations": permutations,
        "strata": "exact CDR3 length",
        "fixed_edge_design": "primary query-positive and query-negative edges; labels permuted",
        "valid_same_pair_range": [min(value[0] for value in valid_pair_counts),
                                  max(value[0] for value in valid_pair_counts)],
        "valid_different_pair_range": [min(value[1] for value in valid_pair_counts),
                                       max(value[1] for value in valid_pair_counts)],
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    preflight = json.loads((args.preflight_dir / "PREFLIGHT.json").read_text(encoding="utf-8"))
    if preflight["status"] != "ready":
        raise ValueError("Pooled official-motif preflight is not accepting.")
    candidate_path = args.preflight_dir / "candidates.tsv"
    pair_path = args.preflight_dir / "pair_manifest.tsv"
    if sha256(candidate_path) != preflight["artifacts"]["candidates_sha256"]:
        raise ValueError("Candidate hash changed after CPU gate.")
    if sha256(pair_path) != preflight["artifacts"]["pairs_sha256"]:
        raise ValueError("Pair hash changed after CPU gate.")
    candidates = pd.read_csv(candidate_path, sep="\t")
    pairs = pd.read_csv(pair_path, sep="\t")
    if candidates["cdr3"].duplicated().any():
        raise ValueError("Candidate table is not exact-CDR3 deduplicated.")
    if set(candidates["mhc_a"]) != {"HLA-A*02:01"} or set(candidates["mhc_b"]) != {"B2M"} \
            or set(candidates["mhc_class"]) != {"MHCI"}:
        raise ValueError("Candidate table is not restricted to exact A*02:01/B2M/MHCI.")
    if candidates["motif_cids"].fillna("").eq("").any():
        raise ValueError("A candidate lacks explicit official motif membership.")
    labels = candidates["is_ylq"].to_numpy(int)
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)
    if not (np.all(labels[q] == 1) and np.all(labels[p] == 1) and np.all(labels[n] == 0)):
        raise ValueError("Pair manifest does not contain YLQ/YLQ/non-YLQ triplets.")
    if not (np.all(q != p) and np.all(q != n) and np.all(p != n)):
        raise ValueError("Pair manifest contains repeated clonotypes within a triplet.")
    if not np.all(candidates["length"].to_numpy(int)[p] == candidates["length"].to_numpy(int)[n]):
        raise ValueError("Negative comparator is not exact-length matched.")

    models = parse_models(args.model)
    config, model_report = validate_run_configs(models)
    embeddings, extraction = extract_embeddings(
        candidates, models, config, args.tokenizer, args.batch_size, args.output_dir
    )
    metrics: dict[str, dict] = {}
    bootstraps: dict[str, dict] = {}
    cosine_pair_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in ("cosine", "euclidean"):
        metrics[metric] = {}
        pair_values = {}
        retrieval_tables = {}
        for name, array in embeddings.items():
            within, between = primary_distances(array, pairs, metric)
            pair_values[name] = (within, between)
            np.savez_compressed(
                args.output_dir / f"primary_distances_{metric}_{name}.npz",
                within=within, between=between,
            )
            retrieval_value, retrieval_table = pooled_retrieval(array, candidates, metric)
            retrieval_table.to_csv(
                args.output_dir / f"retrieval_{metric}_{name}.tsv", sep="\t", index=False
            )
            retrieval_tables[name] = retrieval_table
            metrics[metric][name] = {
                "pairwise_and_contrast": pair_metrics(within, between),
                "retrieval": retrieval_value,
            }
        bootstraps[metric] = bootstrap_clonotypes(
            pair_values, pairs, retrieval_tables, args.bootstrap, args.seed
        )
        if metric == "cosine":
            cosine_pair_values = pair_values

    lengths = candidates["length"].to_numpy(float)
    length_within = np.abs(lengths[q] - lengths[p])
    length_between = np.abs(lengths[q] - lengths[n])
    composition = composition_features(candidates)
    composition_control = {}
    for metric in ("cosine", "euclidean"):
        within = vector_distance(composition[q], composition[p], metric)
        between = vector_distance(composition[q], composition[n], metric)
        composition_control[metric] = pair_metrics(within, between)
    vj_within, vj_between = vj_pair_values(candidates, pairs)
    controls = {
        "exact_length": pair_metrics(length_within, length_between),
        "amino_acid_composition": composition_control,
        "v_j_identity": pair_metrics(vj_within, vj_between),
        "shuffled_labels_cosine": shuffled_label_control(
            cosine_pair_values, candidates, pairs, args.permutations, args.seed
        ),
    }
    assert_finite_tree(metrics, "metrics")
    assert_finite_tree(bootstraps, "bootstraps")
    assert_finite_tree(controls, "controls")
    stable_json(args.output_dir / "metrics.json", metrics)
    stable_json(args.output_dir / "bootstrap.json", bootstraps)
    stable_json(args.output_dir / "controls.json", controls)
    candidates.to_csv(args.output_dir / "candidates.tsv", sep="\t", index=False)
    pairs.to_csv(args.output_dir / "pair_manifest.tsv", sep="\t", index=False)
    hashes = {path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}
    result = {
        "status": "complete",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "target": "exact YLQPRTFLL, official motif-member, pooled HLA-A*02:01/B2M/MHCI",
        "positive_cdr3": preflight["positive_cdr3"],
        "negative_cdr3": preflight["negative_cdr3"],
        "candidate_cdr3": preflight["candidate_cdr3"],
        "pairs": len(pairs),
        "pooling": preflight["pooling"],
        "models": model_report,
        "extraction": extraction,
        "artifact_sha256": hashes,
        "selection_statement": (
            "All seven nonempty matched R/T/P factorial checkpoints reported; "
            "no YLQ-based selection."
        ),
        "circularity_caveat": preflight["circularity_caveat"],
        "claim_scope": "within-VDJdb motif/epitope organization; not cross-donor or independent motif evidence",
        "exploratory_n40_protocol": preflight["exploratory_n40_protocol"],
        "compact_control_selection": preflight["compact_control_selection"],
        "models_trained": False,
    }
    stable_json(args.output_dir / "RESULTS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
