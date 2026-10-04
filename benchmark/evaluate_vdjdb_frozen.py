"""Evaluate matched frozen IRRM-CODEC encoders on a locked VDJdb cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from irrm_codec.multitask_data import resolve_encoder_tokenizer
from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer
from irrm_codec.tokenization import PAD_ID


AA = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-preflight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True,
                        help="NAME:CHECKPOINT:RUN_CONFIG; repeat for r,p,rp,rtp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-pairs-per-label", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_models(values: list[str]) -> dict[str, tuple[Path, Path]]:
    result = {}
    for value in values:
        fields = value.split(":", 2)
        if len(fields) != 3:
            raise ValueError(f"Invalid --model value: {value!r}")
        name, checkpoint, config = fields
        if name in result:
            raise ValueError(f"Duplicate model name: {name}")
        result[name] = (Path(checkpoint), Path(config))
    if set(result) != {"r", "p", "rp", "rtp"}:
        raise ValueError("Exact required model names are r, p, rp, and rtp.")
    return result


def validate_run_configs(models: dict[str, tuple[Path, Path]]) -> tuple[IRRMCodecConfig, dict]:
    expected_weights = {
        "r": (1.0, 0.0, 0.0),
        "p": (0.0, 0.0, 1.0),
        "rp": (1.0, 0.0, 1.0),
        "rtp": (1.0, 1.0, 1.0),
    }
    configs = {name: json.loads(path.read_text(encoding="utf-8")) for name, (_, path) in models.items()}
    reference = configs["rtp"]
    ignored = {"output_dir", "reconstruction_loss_weight", "tcremp_loss_weight", "pgen_loss_weight"}
    ref_training = {key: value for key, value in reference["training"].items() if key not in ignored}
    report = {}
    for name, payload in configs.items():
        if payload["model"] != reference["model"]:
            raise ValueError(f"{name} architecture differs from full DATA-ANCHOR reference.")
        comparable = {key: value for key, value in payload["training"].items() if key not in ignored}
        if comparable != ref_training:
            differing = sorted(key for key in set(comparable) | set(ref_training)
                               if comparable.get(key) != ref_training.get(key))
            raise ValueError(f"{name} has non-weight training differences: {differing}")
        weights = (
            float(payload["training"]["reconstruction_loss_weight"]),
            float(payload["training"]["tcremp_loss_weight"]),
            float(payload["training"]["pgen_loss_weight"]),
        )
        if weights != expected_weights[name]:
            raise ValueError(f"{name} has weights {weights}, expected {expected_weights[name]}")
        report[name] = {
            "checkpoint": str(models[name][0]),
            "checkpoint_sha256": sha256(models[name][0]),
            "run_config": str(models[name][1]),
            "run_config_sha256": sha256(models[name][1]),
            "loss_weights_reconstruction_tcremp_pgen": list(weights),
            "parameter_count": payload["parameter_count"],
        }
    return IRRMCodecConfig(**reference["model"]), report


def extract_embeddings(
    cohort: pd.DataFrame,
    models: dict[str, tuple[Path, Path]],
    model_config: IRRMCodecConfig,
    tokenizer_path: Path,
    batch_size: int,
    output: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(tokenizer_path))
    if tokenizer.vocab_size != model_config.input_vocab_size:
        raise ValueError("Tokenizer vocabulary differs from matched checkpoint architecture.")
    encoded = [tokenizer.encode(sequence, model_config.max_sequence_len)
               for sequence in cohort["cdr3"].astype(str)]
    results = {}
    metadata = {}
    for name, (checkpoint_path, _) in models.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["model_config"] != asdict(model_config):
            raise ValueError(f"{name} checkpoint model_config differs from run_config.")
        model = IRRMCodecTransformer(model_config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(encoded), batch_size):
                rows = encoded[start:start + batch_size]
                width = max(map(len, rows))
                tokens = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
                for index, ids in enumerate(rows):
                    tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                batches.append(model.encode(tokens, tokens.ne(PAD_ID)).numpy().astype(np.float32))
        array = np.concatenate(batches)
        if array.shape != (len(cohort), model_config.latent_dim) or not np.isfinite(array).all():
            raise ValueError(f"{name} emitted invalid latent array {array.shape}.")
        path = output / f"embeddings_{name}.npy"
        np.save(path, array, allow_pickle=False)
        results[name] = array
        metadata[name] = {
            "best_epoch_one_based": int(checkpoint["epoch"]),
            "best_validation_loss": float(checkpoint["best_val_loss"]),
            "shape": list(array.shape),
            "finite": True,
            "sha256": sha256(path),
        }
        del checkpoint, model
    return results, metadata


def distance_matrix(array: np.ndarray, metric: str) -> np.ndarray:
    x = array.astype(np.float64)
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero-norm latent prevents cosine distance.")
        unit = x / norms
        return np.maximum(0.0, 1.0 - unit @ unit.T)
    squared = np.sum(x * x, axis=1, keepdims=True)
    return np.sqrt(np.maximum(0.0, squared + squared.T - 2.0 * (x @ x.T)))


def build_pair_manifest(cohort: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    labels = cohort["label"].astype(str).to_numpy()
    donors = cohort["donor_key"].astype(str).to_numpy()
    lengths = cohort["length"].to_numpy(dtype=int)
    by_label = {label: np.flatnonzero(labels == label) for label in sorted(set(labels))}
    by_length = {length: np.flatnonzero(lengths == length) for length in sorted(set(lengths))}
    rows = []
    for label_index, (label, indices) in enumerate(by_label.items()):
        positives = [(int(indices[i]), int(indices[j])) for i in range(len(indices))
                     for j in range(i + 1, len(indices)) if donors[indices[i]] != donors[indices[j]]]
        rng = np.random.default_rng(seed + label_index)
        if len(positives) > limit:
            chosen = rng.choice(len(positives), size=limit, replace=False)
            positives = [positives[index] for index in sorted(chosen)]
        for pair_index, (query, positive) in enumerate(positives):
            if rng.integers(2):
                query, positive = positive, query
            candidates = by_length.get(int(lengths[positive]), np.empty(0, dtype=int))
            candidates = candidates[(labels[candidates] != label) & (donors[candidates] != donors[query])]
            match_type = "exact_length"
            if len(candidates) == 0:
                allowed = np.flatnonzero((labels != label) & (donors != donors[query]))
                difference = np.abs(lengths[allowed] - lengths[positive])
                candidates = allowed[difference == difference.min()]
                match_type = "nearest_length"
            negative = int(candidates[rng.integers(len(candidates))])
            rows.append({
                "label": label, "query_index": query, "positive_index": positive,
                "negative_index": negative, "query_donor": donors[query],
                "positive_donor": donors[positive], "negative_donor": donors[negative],
                "query_length": int(lengths[query]), "positive_length": int(lengths[positive]),
                "negative_length": int(lengths[negative]), "match_type": match_type,
                "pair_index_within_label": pair_index,
            })
    result = pd.DataFrame(rows)
    if result.empty or (result["query_donor"] == result["positive_donor"]).any():
        raise ValueError("Could not construct donor-disjoint positive pairs.")
    return result


def pair_metrics(matrix: np.ndarray, pairs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)
    within = matrix[q, p]
    between = matrix[q, n]
    truth = np.concatenate((np.ones(len(within), dtype=int), np.zeros(len(between), dtype=int)))
    score = -np.concatenate((within, between))
    per_label = []
    label_values = pairs["label"].astype(str).to_numpy()
    for label in sorted(set(label_values)):
        mask = label_values == label
        pooled = math.sqrt((float(within[mask].var()) + float(between[mask].var())) / 2.0)
        per_label.append({
            "label": label,
            "pairs": int(mask.sum()),
            "within_mean": float(within[mask].mean()),
            "between_mean": float(between[mask].mean()),
            "within_between_ratio": float(within[mask].mean() / between[mask].mean()),
            "standardized_separation": float((between[mask].mean() - within[mask].mean()) / pooled)
                if pooled > 0 else float("nan"),
        })
    table = pd.DataFrame(per_label)
    return {
        "pairs_per_class": len(within),
        "same_label_auroc": float(roc_auc_score(truth, score)),
        "same_label_auprc_balanced": float(average_precision_score(truth, score)),
        "macro_within_between_ratio": float(table["within_between_ratio"].mean()),
        "macro_standardized_separation": float(table["standardized_separation"].mean()),
        "within_expected_lower_fraction": float((table["within_mean"] < table["between_mean"]).mean()),
    }, table


def retrieval_metrics(matrix: np.ndarray, cohort: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    labels = cohort["label"].astype(str).to_numpy()
    donors = cohort["donor_key"].astype(str).to_numpy()
    rows = []
    for query in range(len(cohort)):
        allowed = np.flatnonzero((donors != donors[query]) & (np.arange(len(cohort)) != query))
        relevant = labels[allowed] == labels[query]
        total_relevant = int(relevant.sum())
        if total_relevant == 0:
            continue
        order = np.argsort(matrix[query, allowed], kind="mergesort")
        ranked = relevant[order]
        values = {"query_index": query, "label": labels[query], "query_donor": donors[query],
                  "relevant_donor_excluded": total_relevant}
        for k in (1, 5, 10):
            values[f"precision_at_{k}"] = float(ranked[:k].sum() / k)
        r = total_relevant
        hits = ranked[:r].astype(float)
        precision = np.cumsum(hits) / np.arange(1, r + 1)
        values["ap_at_r"] = float((precision * hits).sum() / r)
        rows.append(values)
    table = pd.DataFrame(rows)
    return {
        "eligible_queries": len(table),
        "precision_at_1": float(table["precision_at_1"].mean()),
        "precision_at_5": float(table["precision_at_5"].mean()),
        "precision_at_10": float(table["precision_at_10"].mean()),
        "map_at_r": float(table["ap_at_r"].mean()),
    }, table


def bootstrap_primary(
    matrices: dict[str, np.ndarray], pairs: pd.DataFrame,
    retrieval: dict[str, pd.DataFrame], replicates: int, seed: int,
) -> dict:
    donors = sorted(
        set(pairs["query_donor"].astype(str)).union(
            *(set(table["query_donor"].astype(str)) for table in retrieval.values())
        )
    )
    donor_index = {donor: index for index, donor in enumerate(donors)}
    pair_donor = pairs["query_donor"].map(donor_index).to_numpy(int)
    labels = sorted(set(pairs["label"].astype(str)))
    label_index = {label: index for index, label in enumerate(labels)}
    pair_label = pairs["label"].map(label_index).to_numpy(int)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(donors), np.full(len(donors), 1 / len(donors)), size=replicates)
    model_boot = {}
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)
    for name, matrix in matrices.items():
        within = matrix[q, p]
        between = matrix[q, n]
        donor_label_within = np.zeros((len(donors), len(labels)))
        donor_label_between = np.zeros_like(donor_label_within)
        donor_label_count = np.zeros_like(donor_label_within)
        np.add.at(donor_label_within, (pair_donor, pair_label), within)
        np.add.at(donor_label_between, (pair_donor, pair_label), between)
        np.add.at(donor_label_count, (pair_donor, pair_label), 1)
        summed_within = weights @ donor_label_within
        summed_between = weights @ donor_label_between
        counts = weights @ donor_label_count
        mean_within = np.full_like(summed_within, np.nan)
        mean_between = np.full_like(summed_between, np.nan)
        np.divide(summed_within, counts, out=mean_within, where=counts > 0)
        np.divide(summed_between, counts, out=mean_between, where=counts > 0)
        ratios = np.full_like(mean_within, np.nan)
        np.divide(mean_within, mean_between, out=ratios, where=mean_between > 0)
        macro_ratio = np.nanmean(ratios, axis=1)

        query_table = retrieval[name]
        query_donor_index = query_table["query_donor"].map(donor_index).to_numpy(int)
        donor_ap_sum = np.zeros(len(donors))
        donor_query_count = np.zeros(len(donors))
        np.add.at(donor_ap_sum, query_donor_index, query_table["ap_at_r"].to_numpy(float))
        np.add.at(donor_query_count, query_donor_index, 1)
        boot_map = (weights @ donor_ap_sum) / np.maximum(weights @ donor_query_count, 1)
        model_boot[name] = {"macro_ratio": macro_ratio, "map_at_r": boot_map}

    def summarize(values: np.ndarray) -> dict:
        return {"ci95": [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]}

    report = {"replicates": replicates, "cluster": "query donor", "models": {}}
    for name, values in model_boot.items():
        report["models"][name] = {
            "macro_within_between_ratio": summarize(values["macro_ratio"]),
            "map_at_r": summarize(values["map_at_r"]),
        }
    report["paired_differences_rtp_minus_other"] = {}
    for other in ("r", "p", "rp"):
        report["paired_differences_rtp_minus_other"][other] = {
            "macro_ratio_lower_is_better": summarize(
                model_boot["rtp"]["macro_ratio"] - model_boot[other]["macro_ratio"]
            ),
            "map_at_r_higher_is_better": summarize(
                model_boot["rtp"]["map_at_r"] - model_boot[other]["map_at_r"]
            ),
        }
    return report


def assert_finite_tree(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"Non-finite result at {path}: {value}")


def controls(
    cohort: pd.DataFrame, pairs: pd.DataFrame, matrices: dict[str, np.ndarray],
    permutations: int, seed: int,
) -> dict:
    lengths = cohort["length"].to_numpy(float)
    length_matrix = np.abs(lengths[:, None] - lengths[None, :])
    length_metrics, _ = pair_metrics(length_matrix, pairs)
    composition = np.zeros((len(cohort), len(AA)), dtype=float)
    for index, sequence in enumerate(cohort["cdr3"].astype(str)):
        for aa_index, aa in enumerate(AA):
            composition[index, aa_index] = sequence.count(aa) / len(sequence)
    composition_metrics = {}
    for metric in ("cosine", "euclidean"):
        value, _ = pair_metrics(distance_matrix(composition, metric), pairs)
        composition_metrics[metric] = value

    rng = np.random.default_rng(seed + 10000)
    all_i, all_j = np.triu_indices(len(cohort), k=1)
    donor = cohort["donor_key"].astype(str).to_numpy()
    same_length = lengths[all_i] == lengths[all_j]
    donor_disjoint = donor[all_i] != donor[all_j]
    eligible = np.flatnonzero(same_length & donor_disjoint)
    if len(eligible) > 100000:
        eligible = rng.choice(eligible, size=100000, replace=False)
    pair_i, pair_j = all_i[eligible], all_j[eligible]
    labels = cohort["label"].astype(str).to_numpy()
    actual_same = labels[pair_i] == labels[pair_j]
    model_distance = {name: matrix[pair_i, pair_j] for name, matrix in matrices.items()}
    actual = {
        name: float(distance[~actual_same].mean() - distance[actual_same].mean())
        for name, distance in model_distance.items()
    }
    null = {name: np.empty(permutations, dtype=float) for name in matrices}
    groups = [np.flatnonzero(lengths == value) for value in sorted(set(lengths))]
    for replicate in range(permutations):
        shuffled = labels.copy()
        for group in groups:
            shuffled[group] = rng.permutation(shuffled[group])
        same = shuffled[pair_i] == shuffled[pair_j]
        for name, distance in model_distance.items():
            null[name][replicate] = float(distance[~same].mean() - distance[same].mean())
    return {
        "length_only_on_primary_pairs": length_metrics,
        "amino_acid_composition": composition_metrics,
        "shuffled_label_null": {
            "pairs": len(pair_i), "permutations": permutations,
            "permutation_strata": "exact CDR3 length",
            "models": {
                name: {
                    "observed_between_minus_within": actual[name],
                    "null_ci95": [float(np.percentile(null[name], 2.5)), float(np.percentile(null[name], 97.5))],
                    "one_sided_p": float((1 + np.sum(null[name] >= actual[name])) / (permutations + 1)),
                } for name in matrices
            },
        },
    }


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    preflight = json.loads(args.cohort_preflight.read_text(encoding="utf-8"))
    if preflight["status"] != "ready_for_cohort_review_not_embedding_extraction":
        raise ValueError("Cohort preflight status is not accepting.")
    if sha256(args.cohort) != preflight["cohort"]["cohort_sha256"]:
        raise ValueError("Cohort hash differs from accepted CPU preflight.")
    cohort = pd.read_csv(args.cohort, sep="\t")
    if cohort["cdr3"].duplicated().any() or len(cohort) != preflight["cohort"]["unique_cdr3"]:
        raise ValueError("Cohort is not the expected unique-clonotype table.")

    config, model_report = validate_run_configs(models)
    embeddings, extraction = extract_embeddings(
        cohort, models, config, args.tokenizer, args.batch_size, args.output_dir
    )
    pairs = build_pair_manifest(cohort, args.max_pairs_per_label, args.seed)
    pairs.to_csv(args.output_dir / "pair_manifest.tsv", sep="\t", index=False)
    matrices_by_metric = {}
    metrics = {}
    retrieval_tables = {}
    per_label_tables = []
    for metric in ("cosine", "euclidean"):
        matrices = {name: distance_matrix(array, metric) for name, array in embeddings.items()}
        matrices_by_metric[metric] = matrices
        metrics[metric] = {}
        retrieval_tables[metric] = {}
        for name, matrix in matrices.items():
            pair_value, per_label = pair_metrics(matrix, pairs)
            retrieval_value, per_query = retrieval_metrics(matrix, cohort)
            metrics[metric][name] = {"pairwise_and_contrast": pair_value, "retrieval": retrieval_value}
            retrieval_tables[metric][name] = per_query
            per_label.insert(0, "model", name)
            per_label.insert(0, "distance", metric)
            per_label_tables.append(per_label)
            per_query.to_csv(args.output_dir / f"retrieval_queries_{metric}_{name}.tsv", sep="\t", index=False)
        stable_json(
            args.output_dir / f"bootstrap_{metric}.json",
            bootstrap_primary(matrices, pairs, retrieval_tables[metric], args.bootstrap, args.seed),
        )
    pd.concat(per_label_tables, ignore_index=True).to_csv(
        args.output_dir / "per_label_metrics.tsv", sep="\t", index=False
    )
    control_report = controls(
        cohort, pairs, matrices_by_metric["cosine"], args.permutations, args.seed
    )
    assert_finite_tree(metrics, "metrics")
    assert_finite_tree(control_report, "controls")
    stable_json(args.output_dir / "metrics.json", metrics)
    stable_json(args.output_dir / "controls.json", control_report)

    artifact_hashes = {
        path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()
    }
    manifest = {
        "status": "complete",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "study": "matched DATA-ANCHOR frozen-encoder VDJdb epitope separation",
        "cohort": {"path": str(args.cohort), "sha256": sha256(args.cohort), "rows": len(cohort),
                   "labels": int(cohort["label"].nunique()), "donors": int(cohort["donor_key"].nunique())},
        "models": model_report,
        "extraction": extraction,
        "pairs": {"rows": len(pairs), "sha256": sha256(args.output_dir / "pair_manifest.tsv"),
                  "nearest_length_fallbacks": int((pairs["match_type"] != "exact_length").sum())},
        "selection_statement": "No checkpoint or model was selected using VDJdb outcomes; all are reported.",
        "inactive_head_statement": "Only frozen encoder latents are evaluated; inactive-head predictions are ignored.",
        "artifact_sha256": artifact_hashes,
        "nonfinite_scan": "passed",
    }
    stable_json(args.output_dir / "RESULTS.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
