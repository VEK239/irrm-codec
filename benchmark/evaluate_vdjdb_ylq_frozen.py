"""Frozen-encoder evaluation for exact YLQPRTFLL in one dominant MHC context."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmark.evaluate_vdjdb_frozen import (
    AA,
    assert_finite_tree,
    distance_matrix,
    extract_embeddings,
    pair_metrics,
    parse_models,
    sha256,
    stable_json,
    validate_run_configs,
)


def binary_retrieval(matrix: np.ndarray, candidates: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    positive = candidates["is_ylq"].to_numpy(int).astype(bool)
    donors = candidates["donor_key"].astype(str).to_numpy()
    rows = []
    for query in np.flatnonzero(positive):
        allowed = np.flatnonzero((donors != donors[query]) & (np.arange(len(candidates)) != query))
        relevant = positive[allowed]
        r = int(relevant.sum())
        if r == 0:
            continue
        ranked = relevant[np.argsort(matrix[query, allowed], kind="mergesort")]
        hits = ranked[:r].astype(float)
        precision = np.cumsum(hits) / np.arange(1, r + 1)
        rows.append({
            "query_index": int(query), "label": "YLQPRTFLL", "query_donor": donors[query],
            "relevant_donor_excluded": r,
            "precision_at_1": float(ranked[:1].sum()),
            "precision_at_5": float(ranked[:5].sum() / 5),
            "precision_at_10": float(ranked[:10].sum() / 10),
            "ap_at_r": float((precision * hits).sum() / r),
        })
    table = pd.DataFrame(rows)
    return {
        "eligible_positive_queries": len(table),
        "precision_at_1": float(table["precision_at_1"].mean()),
        "precision_at_5": float(table["precision_at_5"].mean()),
        "precision_at_10": float(table["precision_at_10"].mean()),
        "map_at_r": float(table["ap_at_r"].mean()),
    }, table


def composition_matrix(candidates: pd.DataFrame) -> np.ndarray:
    result = np.zeros((len(candidates), len(AA)), dtype=float)
    for row, sequence in enumerate(candidates["cdr3"].astype(str)):
        for column, aa in enumerate(AA):
            result[row, column] = sequence.count(aa) / len(sequence)
    return result


def vj_distance(candidates: pd.DataFrame) -> np.ndarray:
    v = candidates["v_call"].fillna("").astype(str).to_numpy()
    j = candidates["j_call"].fillna("").astype(str).to_numpy()
    v_comparable = (v[:, None] != "") & (v[None, :] != "")
    j_comparable = (j[:, None] != "") & (j[None, :] != "")
    v_mismatch = (~v_comparable | (v[:, None] != v[None, :])).astype(float)
    j_mismatch = (~j_comparable | (j[:, None] != j[None, :])).astype(float)
    return (v_mismatch + j_mismatch) / 2.0


def bootstrap_ylq(
    matrices: dict[str, np.ndarray], pairs: pd.DataFrame,
    retrieval: dict[str, pd.DataFrame], replicates: int, seed: int,
) -> dict:
    """Paired query-donor bootstrap for all prespecified primary metrics."""
    donors = sorted(set(pairs["query_donor"].astype(str)).union(
        *(set(table["query_donor"].astype(str)) for table in retrieval.values())))
    donor_index = {donor: index for index, donor in enumerate(donors)}
    pair_donor = pairs["query_donor"].map(donor_index).to_numpy(int)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(donors), np.full(len(donors), 1 / len(donors)), size=replicates
    )
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)
    metric_values: dict[str, dict[str, np.ndarray]] = {}
    for name, matrix in matrices.items():
        within = matrix[q, p]
        between = matrix[q, n]
        result = {key: np.empty(replicates) for key in (
            "within_between_ratio", "between_minus_within", "auroc", "auprc",
            "precision_at_1", "map_at_r",
        )}
        query_table = retrieval[name]
        query_donor = query_table["query_donor"].map(donor_index).to_numpy(int)
        query_p1 = query_table["precision_at_1"].to_numpy(float)
        query_ap = query_table["ap_at_r"].to_numpy(float)
        for replicate, donor_weight in enumerate(weights):
            pair_weight = donor_weight[pair_donor].astype(float)
            within_mean = float(np.average(within, weights=pair_weight))
            between_mean = float(np.average(between, weights=pair_weight))
            result["within_between_ratio"][replicate] = within_mean / between_mean
            result["between_minus_within"][replicate] = between_mean - within_mean
            truth = np.concatenate((np.ones(len(within), dtype=int), np.zeros(len(between), dtype=int)))
            score = -np.concatenate((within, between))
            sample_weight = np.concatenate((pair_weight, pair_weight))
            result["auroc"][replicate] = roc_auc_score(
                truth, score, sample_weight=sample_weight
            )
            result["auprc"][replicate] = average_precision_score(
                truth, score, sample_weight=sample_weight
            )
            query_weight = donor_weight[query_donor].astype(float)
            result["precision_at_1"][replicate] = np.average(query_p1, weights=query_weight)
            result["map_at_r"][replicate] = np.average(query_ap, weights=query_weight)
        metric_values[name] = result

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    report = {
        "replicates": replicates,
        "cluster": "query donor",
        "paired_across_frozen_models": True,
        "models": {
            name: {metric: {"ci95": interval(values)} for metric, values in result.items()}
            for name, result in metric_values.items()
        },
        "paired_differences_rtp_minus_other": {},
    }
    for other in ("r", "p", "rp"):
        report["paired_differences_rtp_minus_other"][other] = {
            metric: {
                "ci95": interval(metric_values["rtp"][metric] - metric_values[other][metric]),
                "direction": "negative_favors_rtp" if metric == "within_between_ratio"
                    else "positive_favors_rtp",
            }
            for metric in metric_values["rtp"]
        }
    return report


def shuffle_null(
    matrices: dict[str, np.ndarray], candidates: pd.DataFrame,
    permutations: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed + 20000)
    labels = candidates["is_ylq"].to_numpy(int).astype(bool)
    lengths = candidates["length"].to_numpy(int)
    donors = candidates["donor_key"].astype(str).to_numpy()
    left, right = np.triu_indices(len(candidates), k=1)
    eligible = (lengths[left] == lengths[right]) & (donors[left] != donors[right])
    left, right = left[eligible], right[eligible]
    if len(left) > 100000:
        selected = rng.choice(len(left), size=100000, replace=False)
        left, right = left[selected], right[selected]

    def contrast(distance: np.ndarray, value: np.ndarray) -> float:
        positive_pair = value[left] & value[right]
        negative_pair = value[left] ^ value[right]
        if not positive_pair.any() or not negative_pair.any():
            return float("nan")
        return float(distance[left[negative_pair], right[negative_pair]].mean() -
                     distance[left[positive_pair], right[positive_pair]].mean())

    observed = {name: contrast(matrix, labels) for name, matrix in matrices.items()}
    null = {name: np.empty(permutations) for name in matrices}
    groups = [np.flatnonzero(lengths == length) for length in sorted(set(lengths))]
    for replicate in range(permutations):
        shuffled = labels.copy()
        for group in groups:
            shuffled[group] = rng.permutation(shuffled[group])
        for name, matrix in matrices.items():
            null[name][replicate] = contrast(matrix, shuffled)
    return {
        "pairs": len(left), "permutations": permutations,
        "strata": "exact CDR3 length", "models": {
            name: {
                "observed_between_minus_within": observed[name],
                "null_ci95": [float(np.nanpercentile(null[name], 2.5)),
                              float(np.nanpercentile(null[name], 97.5))],
                "one_sided_p": float((1 + np.nansum(null[name] >= observed[name])) /
                                     (1 + np.isfinite(null[name]).sum())),
            } for name in matrices
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    preflight = json.loads((args.preflight_dir / "PREFLIGHT.json").read_text())
    if preflight["status"] != "ready":
        raise ValueError("YLQ preflight is not accepting.")
    candidate_path = args.preflight_dir / "candidates.tsv"
    pair_path = args.preflight_dir / "pair_manifest.tsv"
    if sha256(candidate_path) != preflight["artifacts"]["candidates_sha256"]:
        raise ValueError("Candidate table hash changed.")
    if sha256(pair_path) != preflight["artifacts"]["pairs_sha256"]:
        raise ValueError("Pair manifest hash changed.")
    candidates = pd.read_csv(candidate_path, sep="\t")
    pairs = pd.read_csv(pair_path, sep="\t")
    if candidates["cdr3"].duplicated().any():
        raise ValueError("YLQ candidate table is not exact-CDR3 deduplicated.")
    ylq = candidates["is_ylq"].to_numpy(int)
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)
    donors = candidates["donor_key"].astype(str).to_numpy()
    lengths = candidates["length"].to_numpy(int)
    if not (np.all(ylq[q] == 1) and np.all(ylq[p] == 1) and np.all(ylq[n] == 0)):
        raise ValueError("Pair manifest does not contain YLQ/YLQ/non-YLQ triplets.")
    if not (np.all(donors[q] != donors[p]) and np.all(donors[q] != donors[n])
            and np.all(donors[p] != donors[n])):
        raise ValueError("Pair manifest is not fully donor-disjoint within triplets.")
    if not np.all(lengths[p] == lengths[n]):
        raise ValueError("Negative comparator is not exact-length matched to the positive comparator.")
    pairs["label"] = "YLQPRTFLL"
    models = parse_models(args.model)
    config, model_report = validate_run_configs(models)
    embeddings, extraction = extract_embeddings(
        candidates.rename(columns={"candidate_index": "row_index"}), models, config,
        args.tokenizer, 128, args.output_dir,
    )

    metrics = {}
    bootstraps = {}
    matrices_by_metric = {}
    for metric in ("cosine", "euclidean"):
        matrices = {name: distance_matrix(value, metric) for name, value in embeddings.items()}
        matrices_by_metric[metric] = matrices
        metrics[metric] = {}
        retrieval_tables = {}
        for name, matrix in matrices.items():
            pair_value, per_pair_label = pair_metrics(matrix, pairs)
            pair_value.update({
                "within_mean": float(per_pair_label.loc[0, "within_mean"]),
                "between_mean": float(per_pair_label.loc[0, "between_mean"]),
                "between_minus_within": float(
                    per_pair_label.loc[0, "between_mean"] - per_pair_label.loc[0, "within_mean"]
                ),
            })
            retrieval_value, per_query = binary_retrieval(matrix, candidates)
            metrics[metric][name] = {"pairwise_and_contrast": pair_value, "retrieval": retrieval_value}
            retrieval_tables[name] = per_query
            per_query.to_csv(args.output_dir / f"retrieval_{metric}_{name}.tsv", sep="\t", index=False)
        bootstraps[metric] = bootstrap_ylq(
            matrices, pairs, retrieval_tables, args.bootstrap, args.seed
        )

    lengths = candidates["length"].to_numpy(float)
    length_control, _ = pair_metrics(np.abs(lengths[:, None] - lengths[None, :]), pairs)
    comp = composition_matrix(candidates)
    composition_control = {}
    for metric in ("cosine", "euclidean"):
        composition_control[metric], _ = pair_metrics(distance_matrix(comp, metric), pairs)
    vj_control, _ = pair_metrics(vj_distance(candidates), pairs)
    controls = {
        "exact_length": length_control,
        "amino_acid_composition": composition_control,
        "v_j_identity": vj_control,
        "shuffled_labels": shuffle_null(
            matrices_by_metric["cosine"], candidates, args.permutations, args.seed
        ),
    }
    assert_finite_tree(metrics, "metrics")
    assert_finite_tree(bootstraps, "bootstraps")
    assert_finite_tree(controls, "controls")
    stable_json(args.output_dir / "metrics.json", metrics)
    stable_json(args.output_dir / "bootstrap.json", bootstraps)
    stable_json(args.output_dir / "controls.json", controls)
    pairs.to_csv(args.output_dir / "pair_manifest.tsv", sep="\t", index=False)
    candidates.to_csv(args.output_dir / "candidates.tsv", sep="\t", index=False)
    artifact_hashes = {
        path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()
    }
    result = {
        "status": "complete", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "target": "exact YLQPRTFLL in dominant exact MHC context",
        "primary_context": preflight["primary_context"],
        "positive_cdr3": preflight["positive_cdr3"],
        "positive_donors": preflight["positive_donors"],
        "negative_cdr3": preflight["same_context_non_ylq_negative_cdr3"],
        "pairs": len(pairs), "models": model_report, "extraction": extraction,
        "motif_expansion_performed": False,
        "selection_statement": "All four prespecified frozen checkpoints are reported; YLQ outcomes selected none.",
        "artifact_sha256": artifact_hashes,
    }
    stable_json(args.output_dir / "RESULTS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
