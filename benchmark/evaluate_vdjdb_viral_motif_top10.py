"""Evaluate seven frozen DATA-ANCHOR head combinations on ten viral epitopes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.evaluate_vdjdb_frozen import (
    assert_finite_tree,
    extract_embeddings,
    sha256,
    stable_json,
)
from benchmark.evaluate_vdjdb_ylq_motif_pooled import (
    bootstrap_clonotypes,
    composition_features,
    pair_metrics,
    pooled_retrieval,
    primary_distances,
    shuffled_label_control,
    vector_distance,
    vj_pair_values,
)
from irrm_codec.multitask_transformer import IRRMCodecConfig


MODEL_NAMES = ("r", "t", "p", "rt", "rp", "tp", "rtp")
EXPECTED_WEIGHTS = {
    "r": (1.0, 0.0, 0.0),
    "t": (0.0, 1.0, 0.0),
    "p": (0.0, 0.0, 1.0),
    "rt": (1.0, 1.0, 0.0),
    "rp": (1.0, 0.0, 1.0),
    "tp": (0.0, 1.0, 1.0),
    "rtp": (1.0, 1.0, 1.0),
}
PARAMETER_COUNT = 21_535_913


def parse_seven_models(values: list[str]) -> dict[str, tuple[Path, Path]]:
    result = {}
    for value in values:
        fields = value.split(":", 2)
        if len(fields) != 3:
            raise ValueError(f"Invalid --model value: {value!r}")
        name, checkpoint, config = fields
        if name in result:
            raise ValueError(f"Duplicate model name: {name}")
        result[name] = (Path(checkpoint), Path(config))
    if set(result) != set(MODEL_NAMES):
        raise ValueError(f"Exact required model names are {', '.join(MODEL_NAMES)}.")
    return {name: result[name] for name in MODEL_NAMES}


def validate_seven_configs(
    models: dict[str, tuple[Path, Path]],
) -> tuple[IRRMCodecConfig, dict]:
    configs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, (_, path) in models.items()
    }
    reference = configs["rtp"]
    ignored = {"output_dir", "reconstruction_loss_weight", "tcremp_loss_weight", "pgen_loss_weight"}
    reference_training = {
        key: value for key, value in reference["training"].items() if key not in ignored
    }
    report = {}
    for name in MODEL_NAMES:
        payload = configs[name]
        if payload["model"] != reference["model"]:
            raise ValueError(f"{name} architecture differs from RTP.")
        comparable = {
            key: value for key, value in payload["training"].items() if key not in ignored
        }
        if comparable != reference_training:
            differing = sorted(
                key for key in set(comparable) | set(reference_training)
                if comparable.get(key) != reference_training.get(key)
            )
            raise ValueError(f"{name} has non-weight training differences: {differing}")
        weights = (
            float(payload["training"]["reconstruction_loss_weight"]),
            float(payload["training"]["tcremp_loss_weight"]),
            float(payload["training"]["pgen_loss_weight"]),
        )
        if weights != EXPECTED_WEIGHTS[name]:
            raise ValueError(f"{name} has weights {weights}; expected {EXPECTED_WEIGHTS[name]}.")
        if int(payload["parameter_count"]) != PARAMETER_COUNT:
            raise ValueError(f"{name} parameter count is not {PARAMETER_COUNT}.")
        report[name] = {
            "checkpoint": str(models[name][0]),
            "checkpoint_sha256": sha256(models[name][0]),
            "run_config": str(models[name][1]),
            "run_config_sha256": sha256(models[name][1]),
            "loss_weights_reconstruction_tcremp_pgen": list(weights),
            "parameter_count": int(payload["parameter_count"]),
        }
    return IRRMCodecConfig(**reference["model"]), report


def validate_epitope_tables(directory: Path, report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates_path = directory / "candidates.tsv"
    pairs_path = directory / "pair_manifest.tsv"
    if sha256(candidates_path) != report["artifacts"]["candidates_sha256"] \
            or sha256(pairs_path) != report["artifacts"]["pairs_sha256"]:
        raise ValueError(f"Preflight artifacts changed in {directory}.")
    candidates = pd.read_csv(candidates_path, sep="\t")
    pairs = pd.read_csv(pairs_path, sep="\t")
    if candidates["cdr3"].duplicated().any():
        raise ValueError(f"Duplicate CDR3 in {directory}.")
    if set(candidates["mhc_a"]) != {"HLA-A*02:01"} \
            or set(candidates["mhc_b"]) != {"B2M"} \
            or set(candidates["mhc_class"]) != {"MHCI"}:
        raise ValueError(f"Context mismatch in {directory}.")
    if candidates["motif_cids"].fillna("").eq("").any():
        raise ValueError(f"Missing official motif membership in {directory}.")
    labels = candidates["is_target"].to_numpy(int)
    q = pairs["query_index"].to_numpy(int)
    p = pairs["positive_index"].to_numpy(int)
    n = pairs["negative_index"].to_numpy(int)
    if not (np.all(labels[q] == 1) and np.all(labels[p] == 1) and np.all(labels[n] == 0)):
        raise ValueError(f"Invalid target/negative triplets in {directory}.")
    if not np.all(candidates["length"].to_numpy(int)[p]
                  == candidates["length"].to_numpy(int)[n]):
        raise ValueError(f"Non-exact-length negative comparator in {directory}.")
    return candidates, pairs


def summarize_distribution(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    win_rows = []
    for (distance, metric, model), group in rows.groupby(["distance", "metric", "model"]):
        values = group["value"].to_numpy(float)
        summary_rows.append({
            "distance": distance,
            "metric": metric,
            "model": model,
            "epitopes": len(values),
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
        })
    lower_better = {"within_between_ratio"}
    for (distance, metric, epitope), group in rows.groupby(["distance", "metric", "epitope"]):
        target = group["value"].min() if metric in lower_better else group["value"].max()
        winners = sorted(group.loc[np.isclose(group["value"], target), "model"].tolist())
        for model in winners:
            win_rows.append({
                "distance": distance,
                "metric": metric,
                "epitope": epitope,
                "model": model,
                "tied_winners": len(winners),
                "fractional_win": 1.0 / len(winners),
            })
    wins = pd.DataFrame(win_rows)
    win_summary = wins.groupby(["distance", "metric", "model"], as_index=False).agg(
        win_count=("fractional_win", "sum"),
        epitopes_with_best_or_tied=("epitope", "count"),
    )
    complete = pd.MultiIndex.from_product(
        [sorted(rows["distance"].unique()), sorted(rows["metric"].unique()), MODEL_NAMES],
        names=["distance", "metric", "model"],
    ).to_frame(index=False)
    win_summary = complete.merge(
        win_summary, on=["distance", "metric", "model"], how="left"
    ).fillna({"win_count": 0.0, "epitopes_with_best_or_tied": 0})
    win_summary["epitopes_with_best_or_tied"] = win_summary[
        "epitopes_with_best_or_tied"
    ].astype(int)
    return pd.DataFrame(summary_rows), wins, win_summary


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
    expected_epitopes = int(preflight["support_gate"]["selected_epitopes"])
    if preflight["status"] != "ready" or len(preflight["selected"]) != expected_epitopes \
            or expected_epitopes < 2 or expected_epitopes > 10:
        raise ValueError("Up-to-top-10 viral motif preflight is not accepting.")
    master_path = args.preflight_dir / "master_candidates.tsv"
    if sha256(master_path) != preflight["artifacts"]["master_candidates_sha256"]:
        raise ValueError("Master candidate hash changed after CPU gate.")
    master = pd.read_csv(master_path, sep="\t")
    if master["cdr3"].duplicated().any():
        raise ValueError("Master candidates are not unique by exact CDR3.")

    models = parse_seven_models(args.model)
    config, model_report = validate_seven_configs(models)
    embeddings, extraction = extract_embeddings(
        master, models, config, args.tokenizer, args.batch_size, args.output_dir
    )
    master_index = {cdr3: index for index, cdr3 in enumerate(master["cdr3"].astype(str))}
    metric_rows = []
    control_rows = []
    epitope_results = []
    paired_effect_rows = []
    for selected in preflight["selected"]:
        rank = int(selected["rank"])
        epitope = selected["epitope"]
        source_dir = args.preflight_dir / f"epitope-{rank:02d}-{epitope}"
        target_dir = args.output_dir / f"epitope-{rank:02d}-{epitope}"
        target_dir.mkdir()
        candidates, pairs = validate_epitope_tables(source_dir, selected)
        candidates = candidates.copy()
        candidates["is_ylq"] = candidates["is_target"]
        indices = np.asarray([master_index[value] for value in candidates["cdr3"].astype(str)])
        subset_embeddings = {name: array[indices] for name, array in embeddings.items()}
        metrics = {"cosine": {}, "euclidean": {}}
        bootstraps = {}
        cosine_values = {}
        for distance in ("cosine", "euclidean"):
            pair_values = {}
            retrieval_tables = {}
            for name, array in subset_embeddings.items():
                within, between = primary_distances(array, pairs, distance)
                pair_values[name] = (within, between)
                retrieval, retrieval_table = pooled_retrieval(array, candidates, distance)
                retrieval_tables[name] = retrieval_table
                pair_result = pair_metrics(within, between)
                metrics[distance][name] = {
                    "pairwise_and_contrast": pair_result,
                    "retrieval": retrieval,
                }
                values = {
                    "within_between_ratio": pair_result["within_between_ratio"],
                    "between_minus_within": pair_result["between_minus_within"],
                    "same_label_auroc": pair_result["same_label_auroc"],
                    "same_label_auprc_balanced": pair_result["same_label_auprc_balanced"],
                    "precision_at_1": retrieval["precision_at_1"],
                    "map_at_r": retrieval["map_at_r"],
                }
                for metric, value in values.items():
                    metric_rows.append({
                        "rank": rank, "epitope": epitope, "distance": distance,
                        "model": name, "metric": metric, "value": value,
                    })
            if int(selected["positive_cdr3"]) < 3:
                bootstraps[distance] = {
                    "status": "not_estimable_with_fewer_than_three_target_clonotypes",
                    "requested_replicates": args.bootstrap,
                }
            else:
                try:
                    bootstraps[distance] = bootstrap_clonotypes(
                        pair_values, pairs, retrieval_tables, args.bootstrap,
                        args.seed + rank * 1000,
                    )
                    bootstraps[distance]["small_n_caveat"] = (
                        "descriptive/unstable: fewer than 5 target clonotypes"
                        if int(selected["positive_cdr3"]) < 5 else "none"
                    )
                    difference = bootstraps[distance][
                        "paired_differences_rtp_minus_other"
                    ]["rp"]
                    for metric, payload in difference.items():
                        paired_effect_rows.append({
                            "rank": rank, "epitope": epitope, "distance": distance,
                            "metric": metric, "rtp_minus_rp_ci95_low": payload["ci95"][0],
                            "rtp_minus_rp_ci95_high": payload["ci95"][1],
                            "direction": payload["direction"],
                        })
                except (ValueError, ZeroDivisionError) as error:
                    bootstraps[distance] = {
                        "status": "not_estimable_at_available_target_support",
                        "reason": str(error),
                        "requested_replicates": args.bootstrap,
                    }
            if distance == "cosine":
                cosine_values = pair_values

        q = pairs["query_index"].to_numpy(int)
        p = pairs["positive_index"].to_numpy(int)
        n = pairs["negative_index"].to_numpy(int)
        lengths = candidates["length"].to_numpy(float)
        exact_length_result = pair_metrics(
            np.abs(lengths[q] - lengths[p]), np.abs(lengths[q] - lengths[n])
        )
        controls = {
            "exact_length": exact_length_result,
            "amino_acid_composition": {},
        }
        for metric in ("within_between_ratio", "between_minus_within",
                       "same_label_auroc", "same_label_auprc_balanced"):
            control_rows.append({
                "rank": rank, "epitope": epitope, "distance": "absolute",
                "control": "exact_cdr3_length", "metric": metric,
                "value": exact_length_result[metric],
            })
        composition = composition_features(candidates)
        for distance in ("cosine", "euclidean"):
            result = pair_metrics(
                vector_distance(composition[q], composition[p], distance),
                vector_distance(composition[q], composition[n], distance),
            )
            controls["amino_acid_composition"][distance] = result
            for metric in ("within_between_ratio", "between_minus_within",
                           "same_label_auroc", "same_label_auprc_balanced"):
                control_rows.append({
                    "rank": rank, "epitope": epitope, "distance": distance,
                    "control": "amino_acid_composition", "metric": metric,
                    "value": result[metric],
                })
        vj_within, vj_between = vj_pair_values(candidates, pairs)
        vj_result = pair_metrics(vj_within, vj_between)
        controls["v_j_identity"] = vj_result
        for metric in ("within_between_ratio", "between_minus_within",
                       "same_label_auroc", "same_label_auprc_balanced"):
            control_rows.append({
                "rank": rank, "epitope": epitope, "distance": "mismatch_fraction",
                "control": "v_j_identity", "metric": metric, "value": vj_result[metric],
            })
        try:
            controls["shuffled_labels_cosine"] = shuffled_label_control(
                cosine_values, candidates, pairs, args.permutations, args.seed + rank * 1000
            )
        except ValueError as error:
            controls["shuffled_labels_cosine"] = {
                "status": "not_estimable_at_available_target_support",
                "reason": str(error),
                "requested_permutations": args.permutations,
            }
        assert_finite_tree(metrics, f"metrics_{epitope}")
        assert_finite_tree(bootstraps, f"bootstrap_{epitope}")
        assert_finite_tree(controls, f"controls_{epitope}")
        stable_json(target_dir / "metrics.json", metrics)
        stable_json(target_dir / "bootstrap.json", bootstraps)
        stable_json(target_dir / "controls.json", controls)
        candidates.drop(columns=["is_ylq"]).to_csv(
            target_dir / "candidates.tsv", sep="\t", index=False
        )
        pairs.to_csv(target_dir / "pair_manifest.tsv", sep="\t", index=False)
        epitope_results.append({
            "rank": rank,
            "epitope": epitope,
            "positive_cdr3": int(selected["positive_cdr3"]),
            "negative_cdr3": int(selected["selected_unique_negative_cdr3"]),
            "pairs": len(pairs),
            "artifact_sha256": {
                path.name: sha256(path) for path in sorted(target_dir.iterdir()) if path.is_file()
            },
        })

    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(args.output_dir / "per_epitope_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(control_rows).to_csv(
        args.output_dir / "per_epitope_controls.tsv", sep="\t", index=False
    )
    pd.DataFrame(paired_effect_rows).to_csv(
        args.output_dir / "per_epitope_rtp_minus_rp_bootstrap.tsv", sep="\t", index=False
    )
    distribution, wins, win_summary = summarize_distribution(metric_table)
    distribution.to_csv(args.output_dir / "distribution_summary.tsv", sep="\t", index=False)
    wins.to_csv(args.output_dir / "win_assignments.tsv", sep="\t", index=False)
    win_summary.to_csv(args.output_dir / "win_summary.tsv", sep="\t", index=False)
    point_effects = metric_table.pivot_table(
        index=["rank", "epitope", "distance", "metric"], columns="model", values="value"
    ).reset_index()
    point_effects["rtp_minus_rp"] = point_effects["rtp"] - point_effects["rp"]
    point_effects.to_csv(args.output_dir / "rtp_minus_rp_effects.tsv", sep="\t", index=False)
    effect_summary_rows = []
    for (distance, metric), group in point_effects.groupby(["distance", "metric"]):
        values = group["rtp_minus_rp"].to_numpy(float)
        negative_favors_rtp = metric == "within_between_ratio"
        effect_summary_rows.append({
            "distance": distance,
            "metric": metric,
            "epitopes": len(values),
            "median_rtp_minus_rp": float(np.median(values)),
            "q1_rtp_minus_rp": float(np.percentile(values, 25)),
            "q3_rtp_minus_rp": float(np.percentile(values, 75)),
            "rtp_win_count": int(np.sum(values < 0 if negative_favors_rtp else values > 0)),
            "direction": "negative_favors_rtp" if negative_favors_rtp
                else "positive_favors_rtp",
        })
    pd.DataFrame(effect_summary_rows).to_csv(
        args.output_dir / "rtp_minus_rp_distribution.tsv", sep="\t", index=False
    )

    result = {
        "status": "complete",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "models": model_report,
        "extraction": extraction,
        "epitopes": epitope_results,
        "epitope_weighting": "each epitope contributes one point to distribution summaries",
        "pooled_pair_estimate_reported": False,
        "selection_statement": (
            "Up to ten selected only by post-filter motif-member support; no model metric used. "
            f"Actual evaluable count={expected_epitopes}, shortfall="
            f"{preflight['support_gate']['evaluable_epitope_shortfall']}."
        ),
        "claim_scope": "exploratory within-VDJdb motif/epitope organization; pooled donors",
        "circularity_caveat": preflight["circularity_caveat"],
        "single_seed": True,
        "models_trained_during_evaluation": False,
    }
    hashes = {
        path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()
    }
    result["artifact_sha256"] = hashes
    stable_json(args.output_dir / "RESULTS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
