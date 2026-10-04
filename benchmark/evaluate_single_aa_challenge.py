"""Frozen-encoder delta benchmark on controlled one-AA substitutions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
AA = "ACDEFGHIKLMNPQRSTVWY"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_x, std_x = x.mean(0), x.std(0)
    std_x[std_x < 1e-8] = 1.0
    z = (x - mean_x) / std_x
    mean_y = y.mean(0)
    centered = y - mean_y
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ centered)
    return coef, mean_x, std_x, mean_y


def ridge_predict(model: tuple[np.ndarray, ...], x: np.ndarray) -> np.ndarray:
    coef, mean_x, std_x, mean_y = model
    return (x - mean_x) / std_x @ coef + mean_y


def finite_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float((spearmanr if kind == "spearman" else pearsonr)(x, y).statistic)


def choose_alpha_regression(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> float:
    scores = []
    scale = y[train].std(0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    for alpha in ALPHAS:
        pred = ridge_predict(ridge_fit(x[train], y[train], alpha), x[val])
        scores.append((float(np.mean(((pred - y[val]) / scale) ** 2)), alpha))
    return min(scores)[1]


def choose_alpha_classifier(x: np.ndarray, labels: np.ndarray, train: np.ndarray, val: np.ndarray) -> tuple[float, np.ndarray]:
    # Keep the complete declared label space. A test-only rare position remains
    # unlearnable (correctly scoring as an error) rather than crashing encoding.
    classes = np.array(sorted(set(labels)))
    mapping = {value: i for i, value in enumerate(classes)}
    onehot = np.eye(len(classes))[np.array([mapping[value] for value in labels])]
    scores = []
    for alpha in ALPHAS:
        pred = classes[np.argmax(ridge_predict(ridge_fit(x[train], onehot[train], alpha), x[val]), axis=1)]
        scores.append((-f1_score(labels[val], pred, average="macro", zero_division=0), alpha))
    return min(scores)[1], classes


def regression_metrics(truth: np.ndarray, pred: np.ndarray, train_truth: np.ndarray) -> dict:
    scale = float(np.std(train_truth)) or 1.0
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "nrmse_train_sd": float(np.sqrt(mean_squared_error(truth, pred)) / scale),
        "mae": float(mean_absolute_error(truth, pred)),
        "r2": float(r2_score(truth, pred)),
        "pearson": finite_correlation(truth, pred, "pearson"),
        "spearman": finite_correlation(truth, pred, "spearman"),
    }


def tcremp_metrics(truth: np.ndarray, pred: np.ndarray, train_truth: np.ndarray) -> dict:
    denom = np.linalg.norm(truth, axis=1) * np.linalg.norm(pred, axis=1)
    cosine = np.divide(np.sum(truth * pred, axis=1), denom, out=np.zeros(len(truth)), where=denom > 1e-12)
    scale = train_truth.std(0)
    active = scale >= 1e-8
    return {
        "mean_delta_cosine": float(cosine.mean()),
        "median_delta_cosine": float(np.median(cosine)),
        "standardized_rmse": float(np.sqrt(np.mean(((truth[:, active] - pred[:, active]) / scale[active]) ** 2))),
        "raw_rmse": float(np.sqrt(np.mean((truth - pred) ** 2))),
        "active_dimensions": int(active.sum()),
    }


def rank_summary(frame: pd.DataFrame, true_pgen: np.ndarray, pred_pgen: np.ndarray,
                 true_t: np.ndarray, pred_t: np.ndarray, mask: np.ndarray) -> dict:
    rows = np.flatnonzero(mask)
    pgen_r, tcremp_r, pgen_pairs, tcremp_pairs = [], [], [], []
    for parent in sorted(frame.iloc[rows]["parent_id"].unique()):
        idx = rows[frame.iloc[rows]["parent_id"].to_numpy() == parent]
        if len(idx) < 2:
            continue
        true_te = np.linalg.norm(true_t[idx], axis=1)
        pred_te = np.linalg.norm(pred_t[idx], axis=1)
        pgen_r.append(finite_correlation(true_pgen[idx], pred_pgen[idx], "spearman"))
        tcremp_r.append(finite_correlation(true_te, pred_te, "spearman"))
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pgen_pairs.append(np.sign(true_pgen[idx[a]] - true_pgen[idx[b]]) == np.sign(pred_pgen[idx[a]] - pred_pgen[idx[b]]))
                tcremp_pairs.append(np.sign(true_te[a] - true_te[b]) == np.sign(pred_te[a] - pred_te[b]))
    return {
        "parent_mean_pgen_spearman": float(np.mean(pgen_r)),
        "parent_mean_tcremp_effect_spearman": float(np.mean(tcremp_r)),
        "pgen_pair_order_accuracy": float(np.mean(pgen_pairs)),
        "tcremp_effect_pair_order_accuracy": float(np.mean(tcremp_pairs)),
        "parents": len(pgen_r),
    }


def identity_features(frame: pd.DataFrame, include_position: bool) -> np.ndarray:
    result = np.zeros((len(frame), 20 + (40 if include_position else 0) + (40 if include_position else 0)), np.float64)
    lookup = {aa: i for i, aa in enumerate(AA)}
    for i, row in enumerate(frame.itertuples(index=False)):
        result[i, lookup[row.to_aa]] += 1
        result[i, lookup[row.from_aa]] -= 1
        if include_position:
            result[i, 20 + min(int(row.mutation_position_zero_based), 39)] = 1
            result[i, 60 + min(int(row.length), 39)] = 1
    return result


def evaluate_representation(name: str, x: np.ndarray, frame: pd.DataFrame, delta_pgen: np.ndarray,
                            delta_t: np.ndarray, splits: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    train, val, test = (splits[key] for key in ("train", "val", "test"))
    p_alpha = choose_alpha_regression(x, delta_pgen[:, None], train, val)
    p_model = ridge_fit(x[train], delta_pgen[train, None], p_alpha)
    p_pred = ridge_predict(p_model, x).ravel()
    t_alpha = choose_alpha_regression(x, delta_t, train, val)
    t_model = ridge_fit(x[train], delta_t[train], t_alpha)
    t_pred = ridge_predict(t_model, x)

    classifiers = {}
    class_predictions = {}
    for label_name, labels in {
        "mutation_position": frame["mutation_position_zero_based"].to_numpy(int),
        "target_amino_acid": frame["to_aa"].astype(str).to_numpy(),
    }.items():
        alpha, classes = choose_alpha_classifier(x, labels, train, val)
        class_to_index = {value: i for i, value in enumerate(classes)}
        onehot = np.eye(len(classes))[np.array([class_to_index[value] for value in labels])]
        pred = classes[np.argmax(ridge_predict(ridge_fit(x[train], onehot[train], alpha), x), axis=1)]
        classifiers[label_name] = {
            "alpha": alpha,
            "macro_f1": float(f1_score(labels[test], pred[test], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[test], pred[test])),
        }
        class_predictions[label_name] = pred
    direction_truth = (delta_pgen > 0).astype(int)
    direction_alpha, direction_classes = choose_alpha_classifier(x, direction_truth, train, val)
    direction_onehot = np.eye(len(direction_classes))[direction_truth]
    direction_pred = direction_classes[np.argmax(ridge_predict(
        ridge_fit(x[train], direction_onehot[train], direction_alpha), x), axis=1)]
    metrics = {
        "representation": name,
        "dimension": int(x.shape[1]),
        "pgen": {"alpha": p_alpha, **regression_metrics(delta_pgen[test], p_pred[test], delta_pgen[train])},
        "tcremp": {"alpha": t_alpha, **tcremp_metrics(delta_t[test], t_pred[test], delta_t[train])},
        "mutation_identity": classifiers,
        "pgen_direction": {
            "alpha": direction_alpha,
            "macro_f1": float(f1_score(direction_truth[test], direction_pred[test], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(direction_truth[test], direction_pred[test])),
        },
        "within_parent_ranking": rank_summary(frame, delta_pgen, p_pred, delta_t, t_pred, test),
    }
    predictions = {"pgen": p_pred, "tcremp": t_pred,
                   "position": class_predictions["mutation_position"],
                   "to_aa": class_predictions["target_amino_acid"]}
    return metrics, predictions


def parent_bootstrap(frame: pd.DataFrame, truth_p: np.ndarray, truth_t: np.ndarray,
                     predictions: dict[str, dict[str, np.ndarray]], test: np.ndarray,
                     repeats: int, seed: int) -> dict:
    parent_rows = {parent: np.flatnonzero((frame["parent_id"].to_numpy() == parent) & test)
                   for parent in sorted(frame.loc[test, "parent_id"].unique())}
    parents = np.array(list(parent_rows), dtype=object)
    rng = np.random.default_rng(seed)
    result = {}
    for comparator in sorted(set(predictions) - {"rtp"}):
        samples = {"pgen_nrmse": [], "tcremp_one_minus_cosine": []}
        for _ in range(repeats):
            chosen = rng.choice(parents, size=len(parents), replace=True)
            idx = np.concatenate([parent_rows[parent] for parent in chosen])
            scale = np.std(truth_p[np.flatnonzero(frame["probe_split"].to_numpy() == "train")]) or 1.0
            def score(name: str) -> tuple[float, float]:
                p = float(np.sqrt(np.mean((predictions[name]["pgen"][idx] - truth_p[idx]) ** 2)) / scale)
                pred, true = predictions[name]["tcremp"][idx], truth_t[idx]
                denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
                cos = np.divide(np.sum(pred * true, axis=1), denom, out=np.zeros(len(idx)), where=denom > 1e-12)
                return p, float(1 - cos.mean())
            rtp, other = score("rtp"), score(comparator)
            samples["pgen_nrmse"].append(rtp[0] - other[0])
            samples["tcremp_one_minus_cosine"].append(rtp[1] - other[1])
        result[comparator] = {
            metric: {"rtp_minus_comparator_mean": float(np.mean(values)),
                     "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])]}
            for metric, values in samples.items()
        }
    return result


def bootstrap_or_disabled(frame: pd.DataFrame, truth_p: np.ndarray, truth_t: np.ndarray,
                          predictions: dict[str, dict[str, np.ndarray]], test: np.ndarray,
                          repeats: int, seed: int) -> dict:
    if repeats < 0:
        raise ValueError("bootstrap repeats must be non-negative")
    if repeats == 0:
        return {
            "status": "disabled_by_user",
            "repeats": 0,
            "uncertainty_available": False,
            "significance_claim_permitted": False,
        }
    return {
        "status": "enabled",
        "repeats": repeats,
        "uncertainty_available": True,
        "significance_claim_permitted": True,
        "comparisons": parent_bootstrap(frame, truth_p, truth_t, predictions, test, repeats, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    parents = pd.read_csv(args.prepared / "parents.tsv", sep="\t")
    frame = pd.read_csv(args.targets / "targets.tsv", sep="\t")
    target_manifest = json.loads((args.targets / "TARGETS.json").read_text())
    if target_manifest["status"] != "accepted_for_frozen_delta_evaluation":
        raise ValueError("Target gate did not accept this challenge.")
    parent_t = np.load(args.prepared / "parent_tcremp.npy", mmap_mode="r")
    mutant_t = np.load(args.targets / "mutant_tcremp.npy", mmap_mode="r")
    parent_row = {identifier: i for i, identifier in enumerate(parents["parent_id"].astype(str))}
    pidx = np.array([parent_row[value] for value in frame["parent_id"].astype(str)])
    delta_t = np.asarray(mutant_t, np.float64) - np.asarray(parent_t[pidx], np.float64)
    parent_pgen = parents.set_index("parent_id")["log10_pgen_1mm"]
    delta_pgen = frame["log10_pgen_1mm"].to_numpy(float) - parent_pgen.loc[frame["parent_id"]].to_numpy(float)
    if not np.isfinite(delta_t).all() or not np.isfinite(delta_pgen).all():
        raise ValueError("Nonfinite delta targets.")

    from benchmark.evaluate_vdjdb_frozen import extract_embeddings, parse_models, validate_run_configs
    models = parse_models(args.model)
    config, parity = validate_run_configs(models)
    sequences = pd.DataFrame({"cdr3": pd.concat((parents["junction_aa"], frame["junction_aa"]), ignore_index=True)})
    embedded, extraction = extract_embeddings(sequences, models, config, args.tokenizer,
                                               args.batch_size, args.output_dir)
    n_parent = len(parents)
    representations = {name: array[n_parent:] - array[:n_parent][pidx] for name, array in embedded.items()}
    representations["concat_rtp"] = np.concatenate([representations[name] for name in ("r", "t", "p")], axis=1)
    split_masks = {name: frame["probe_split"].to_numpy() == name for name in ("train", "val", "test")}
    pca = PCA(n_components=128, random_state=args.seed).fit(representations["concat_rtp"][split_masks["train"]])
    representations["pca_concat_rtp_128"] = pca.transform(representations["concat_rtp"])
    representations["delta_composition_control"] = identity_features(frame, False)
    representations["mutation_identity_control"] = identity_features(frame, True)

    metrics, predictions = {}, {}
    for name, array in representations.items():
        metrics[name], predictions[name] = evaluate_representation(
            name, np.asarray(array, np.float64), frame, delta_pgen, delta_t, split_masks
        )
    seven = ["r", "t", "p", "rt", "rp", "tp", "rtp"]
    errors = {
        name: {
            "sequence": float(1 - np.mean([metrics[name]["mutation_identity"][key]["macro_f1"]
                                            for key in ("mutation_position", "target_amino_acid")])),
            "tcremp": float(1 - metrics[name]["tcremp"]["mean_delta_cosine"]),
            "pgen": float(metrics[name]["pgen"]["nrmse_train_sd"]),
        } for name in seven
    }
    best = {task: min(errors[name][task] for name in seven) for task in ("sequence", "tcremp", "pgen")}
    aggregate = {}
    for name in seven:
        regrets = {task: (errors[name][task] - best[task]) / max(best[task], 1e-8) for task in best}
        aggregate[name] = {"task_errors": errors[name], "relative_regrets": regrets,
                           "mean_relative_regret": float(np.mean(list(regrets.values()))),
                           "worst_task_relative_regret": float(max(regrets.values()))}
    bootstrap = bootstrap_or_disabled(frame, delta_pgen, delta_t, predictions, split_masks["test"],
                                      args.bootstrap, args.seed)
    payload = {
        "status": "complete",
        "study": "parent-disjoint controlled one-amino-acid frozen-delta benchmark",
        "all_pairs_edit_distance": 1,
        "probe_split_unit": "parent clonotype",
        "rows": len(frame),
        "parents": int(frame["parent_id"].nunique()),
        "split_parent_counts": {
            str(key): int(value)
            for key, value in frame.groupby("probe_split")["parent_id"].nunique().items()
        },
        "model_parity": parity,
        "extraction": extraction,
        "metrics": metrics,
        "preregistered_aggregate": {"definition": "relative error regret against best of seven per task; lower is better",
                                    "best_task_errors": best, "models": aggregate},
        "parent_clustered_bootstrap": bootstrap,
        "controls": ["delta_composition_control", "mutation_identity_control"],
        "ensemble_baselines": ["concat_rtp", "pca_concat_rtp_128"],
        "limitations": [
            "fixed seed42 encoders",
            "linear probes only",
            "mutants are in-silico and are not observed repertoire clonotypes",
            "bootstrap uncertainty was disabled by user direction; point estimates only and no significance claim",
        ],
    }
    result_path = args.output_dir / "RESULTS.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    hashes = {path.name: sha256(path) for path in args.output_dir.iterdir() if path.is_file()}
    (args.output_dir / "HASHES.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "results": str(result_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
