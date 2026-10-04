"""Frozen linear-probe benchmark for balanced DATA-ANCHOR latent sufficiency.

The seven encoders are never updated.  Fresh ridge readouts are fitted on the
locked training manifest, their regularization is selected on validation, and
the locked test split is evaluated once.  A fixed-position sequence probe is
used instead of any checkpoint decoder so reconstruction-supervised encoders
receive no privileged task head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_CLASS = {aa: index + 1 for index, aa in enumerate(AA)}
PAD_CLASS = 0
EOS_CLASS = len(AA) + 1
N_CLASSES = len(AA) + 2
TASKS = ("sequence", "tcremp", "pgen")
SPECIALISTS = {"sequence": "r", "tcremp": "t", "pgen": "p"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alphas", default="0.01,1,100")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sequence_targets(sequences: list[str], max_length: int) -> np.ndarray:
    """Return fixed targets: PAD=0, amino acids=1..20, EOS=21."""
    targets = np.full((len(sequences), max_length + 1), PAD_CLASS, dtype=np.uint8)
    for row, sequence in enumerate(sequences):
        if not sequence or len(sequence) > max_length or any(aa not in AA_TO_CLASS for aa in sequence):
            raise ValueError(f"Invalid benchmark sequence at row {row}: {sequence!r}")
        targets[row, : len(sequence)] = [AA_TO_CLASS[aa] for aa in sequence]
        targets[row, len(sequence)] = EOS_CLASS
    return targets


def one_hot_flat(targets: np.ndarray) -> np.ndarray:
    eye = np.eye(N_CLASSES, dtype=np.float32)
    return eye[targets].reshape(len(targets), -1)


def sequence_metrics(scores: np.ndarray, targets: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    predicted = scores.reshape(len(targets), targets.shape[1], N_CLASSES).argmax(axis=2)
    active = np.arange(targets.shape[1])[None, :] <= (targets == EOS_CLASS).argmax(axis=1)[:, None]
    correct = predicted == targets
    row_error = 1.0 - (correct & active).sum(axis=1) / active.sum(axis=1)
    exact = np.all(correct | ~active, axis=1)
    predicted_eos = np.where((predicted == EOS_CLASS).any(axis=1),
                             (predicted == EOS_CLASS).argmax(axis=1), targets.shape[1])
    true_length = (targets == EOS_CLASS).argmax(axis=1)
    return {
        "token_accuracy": float(1.0 - row_error.mean()),
        "exact_accuracy": float(exact.mean()),
        "length_accuracy": float((predicted_eos == true_length).mean()),
        "task_loss": float(row_error.mean()),
    }, row_error.astype(np.float64)


def tcremp_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    residual = predicted - target
    row_mse = np.mean(residual * residual, axis=1)
    pred_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    safe = (pred_norm > 0) & (target_norm > 0)
    cosine = np.zeros(len(target), dtype=np.float64)
    cosine[safe] = np.sum(predicted[safe] * target[safe], axis=1) / (
        pred_norm[safe] * target_norm[safe]
    )
    return {
        "standardized_mse": float(row_mse.mean()),
        "standardized_rmse": float(np.sqrt(row_mse.mean())),
        "mean_cosine": float(cosine.mean()),
        "task_loss": float(row_mse.mean()),
    }, row_mse.astype(np.float64)


def pgen_metrics(predicted: np.ndarray, target: np.ndarray, std: float) -> tuple[dict[str, float], np.ndarray]:
    residual = predicted.reshape(-1) - target.reshape(-1)
    squared = residual * residual
    raw = residual * std
    return {
        "standardized_mse": float(squared.mean()),
        "standardized_rmse": float(np.sqrt(squared.mean())),
        "raw_rmse_log10_pgen_1mm": float(np.sqrt(np.mean(raw * raw))),
        "raw_mae_log10_pgen_1mm": float(np.mean(np.abs(raw))),
        "task_loss": float(squared.mean()),
    }, squared.astype(np.float64)


def standardize_representation(
    array: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    mean = np.asarray(array[train_indices], dtype=np.float64).mean(axis=0)
    std = np.asarray(array[train_indices], dtype=np.float64).std(axis=0)
    zero = std < 1e-8
    std[zero] = 1.0
    result = (np.asarray(array, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite standardized representation.")
    return result.astype(np.float32), {
        "train_only": True,
        "zero_variance_features": int(zero.sum()),
        "dimension": int(result.shape[1]),
    }


def composition_features(
    table: pd.DataFrame,
    include_vj: bool,
    fit_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    sequences = table["junction_aa"].astype(str).tolist()
    values = np.zeros((len(table), len(AA) + 1), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        values[row, : len(AA)] = [sequence.count(aa) / len(sequence) for aa in AA]
        values[row, -1] = len(sequence) / 40.0
    columns = [f"fraction_{aa}" for aa in AA] + ["length_over_40"]
    if include_vj:
        if not {"v_call", "j_call"}.issubset(table.columns):
            raise ValueError("V/J control requested but benchmark lacks v_call/j_call.")
        if fit_indices is None:
            raise ValueError("V/J vocabulary must be fitted from explicit train indices.")
        categoricals = table[["v_call", "j_call"]].fillna("").astype(str)
        encoded_parts = []
        for field, prefix in (("v_call", "v"), ("j_call", "j")):
            vocabulary = sorted(set(categoricals.iloc[fit_indices][field]))
            mapping = {value: index for index, value in enumerate(vocabulary)}
            encoded = np.zeros((len(table), len(vocabulary)), dtype=np.float32)
            for row, value in enumerate(categoricals[field]):
                index = mapping.get(value)
                if index is not None:
                    encoded[row, index] = 1.0
            encoded_parts.append(encoded)
            columns.extend([f"{prefix}_{value}" for value in vocabulary])
        values = np.concatenate([values, *encoded_parts], axis=1)
    return values, {"columns": columns, "dimension": len(columns), "sequence_only": not include_vj}


def append_intercept(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)


def ridge_weights(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xa = append_intercept(np.asarray(x, dtype=np.float64))
    gram = xa.T @ xa
    penalty = np.eye(gram.shape[0], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + penalty, xa.T @ np.asarray(y, dtype=np.float64))


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return append_intercept(np.asarray(x, dtype=np.float64)) @ weights


def fit_probe(
    x: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    target: np.ndarray,
    alphas: list[float],
    score,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    chosen: tuple[float, float, np.ndarray] | None = None
    trace = {}
    xa = append_intercept(np.asarray(x[train], dtype=np.float64))
    gram = xa.T @ xa
    cross = xa.T @ np.asarray(target[train], dtype=np.float64)
    for alpha in alphas:
        penalty = np.eye(gram.shape[0], dtype=np.float64) * alpha
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(gram + penalty, cross)
        val_pred = ridge_predict(x[val], weights)
        loss = float(score(val_pred, val))
        trace[str(alpha)] = loss
        if chosen is None or (loss, alpha) < (chosen[0], chosen[1]):
            chosen = (loss, alpha, weights)
    assert chosen is not None
    return chosen[1], ridge_predict(x[val], chosen[2]), ridge_predict(x[test], chosen[2]), trace


def aggregate_losses(
    losses: dict[str, float], specialist: dict[str, float], null: dict[str, float]
) -> dict[str, object]:
    regret = {}
    skill = {}
    within = {}
    for task in TASKS:
        gap = null[task] - specialist[task]
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError(f"Invalid validation specialist/null gap for {task}: {gap}")
        regret[task] = float((losses[task] - specialist[task]) / gap)
        skill[task] = float(np.clip((null[task] - losses[task]) / gap, 0.0, 1.0))
        within[task] = bool(losses[task] <= 1.05 * specialist[task])
    return {
        "normalized_regret_by_task": regret,
        "mean_normalized_regret": float(np.mean(list(regret.values()))),
        "worst_task_normalized_regret": float(max(regret.values())),
        "fixed_reference_hypervolume": float(np.prod(list(skill.values()))),
        "tasks_within_5pct_of_specialist_loss": int(sum(within.values())),
        "within_5pct_by_task": within,
    }


def bootstrap_aggregate_deltas(
    per_row: dict[str, dict[str, np.ndarray]],
    specialist: dict[str, float],
    null: dict[str, float],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = len(next(iter(next(iter(per_row.values())).values())))
    output = {}
    for comparator in sorted(set(per_row) - {"rtp"}):
        mean_delta = np.empty(replicates, dtype=np.float64)
        worst_delta = np.empty(replicates, dtype=np.float64)
        task_delta = {task: np.empty(replicates, dtype=np.float64) for task in TASKS}
        for replicate in range(replicates):
            sample = rng.integers(0, n, n)
            aggregates = {}
            for name in ("rtp", comparator):
                losses = {task: float(per_row[name][task][sample].mean()) for task in TASKS}
                aggregates[name] = aggregate_losses(losses, specialist, null)
            mean_delta[replicate] = (aggregates["rtp"]["mean_normalized_regret"]
                                     - aggregates[comparator]["mean_normalized_regret"])
            worst_delta[replicate] = (aggregates["rtp"]["worst_task_normalized_regret"]
                                      - aggregates[comparator]["worst_task_normalized_regret"])
            for task in TASKS:
                task_delta[task][replicate] = (
                    per_row["rtp"][task][sample].mean() - per_row[comparator][task][sample].mean()
                )
        output[comparator] = {
            "orientation": "RTP minus comparator; negative favors RTP for all regret/loss endpoints",
            "mean_normalized_regret": ci(mean_delta),
            "worst_task_normalized_regret": ci(worst_delta),
            "raw_task_loss": {task: ci(values) for task, values in task_delta.items()},
        }
    return output


def ci(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def file_manifest(paths: list[Path]) -> dict[str, dict[str, object]]:
    return {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in paths}


def main() -> None:
    # Keep statistical helpers importable in lightweight local test environments
    # that do not have the cluster's PyTorch runtime.
    from benchmark.evaluate_vdjdb_frozen import (
        extract_embeddings,
        parse_models,
        validate_run_configs,
    )
    from irrm_codec.multitask_data import (
        TargetStandardizer,
        load_prepared_benchmark,
        select_split_indices,
    )

    args = parse_args()
    started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    alphas = sorted({float(item) for item in args.alphas.split(",")})
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("All ridge alphas must be positive.")

    table, tcremp_raw = load_prepared_benchmark(args.data_dir)
    train = select_split_indices(table, args.data_dir, "train")
    val = select_split_indices(table, args.data_dir, "val")
    test = select_split_indices(table, args.data_dir, "test")
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Locked manifests overlap.")
    normalizer = TargetStandardizer.load(args.normalizer)
    if normalizer.pgen_target != "log10_pgen_1mm" or normalizer.train_rows != len(train):
        raise ValueError("Locked train-only normalizer does not match requested target/split.")
    tcremp_target = ((np.asarray(tcremp_raw, dtype=np.float32) - normalizer.tcremp_mean)
                     / normalizer.tcremp_std).astype(np.float32)
    pgen_target = ((table["log10_pgen_1mm"].to_numpy(dtype=np.float32) - normalizer.pgen_mean)
                   / normalizer.pgen_std).astype(np.float32)[:, None]
    seq_target = sequence_targets(table["junction_aa"].astype(str).tolist(), 40)
    seq_one_hot = one_hot_flat(seq_target)

    models = parse_models(args.model)
    model_config, model_report = validate_run_configs(models)
    if model_config.latent_dim != 128 or model_config.max_sequence_len != 40:
        raise ValueError("This preregistration requires the matched 128-D, length-40 models.")
    cohort = table[["junction_aa"]].rename(columns={"junction_aa": "cdr3"})
    embeddings, extraction = extract_embeddings(
        cohort, models, model_config, args.tokenizer, args.batch_size, args.output_dir,
    )
    representations: dict[str, np.ndarray] = dict(embeddings)
    concat = np.concatenate([embeddings[name] for name in ("r", "t", "p")], axis=1)
    representations["concat_r_t_p"] = concat
    pca = PCA(n_components=128, svd_solver="randomized", random_state=args.seed)
    pca.fit(concat[train])
    representations["pca128_concat_r_t_p"] = pca.transform(concat).astype(np.float32)
    comp, comp_meta = composition_features(table, include_vj=False)
    comp_vj, comp_vj_meta = composition_features(table, include_vj=True, fit_indices=train)
    representations["composition_length"] = comp
    representations["composition_length_vj"] = comp_vj

    representation_meta = {}
    results = {}
    per_row = {}
    validation_losses = {}
    test_losses = {}
    for name, raw in representations.items():
        x, scaling = standardize_representation(raw, train)
        sequence_alpha, sequence_val_pred, sequence_test_pred, sequence_trace = fit_probe(
            x, train, val, test, seq_one_hot, alphas,
            lambda pred, indices: sequence_metrics(pred, seq_target[indices])[0]["task_loss"],
        )
        t_alpha, t_val_pred, t_test_pred, t_trace = fit_probe(
            x, train, val, test, tcremp_target, alphas,
            lambda pred, indices: tcremp_metrics(pred, tcremp_target[indices])[0]["task_loss"],
        )
        p_alpha, p_val_pred, p_test_pred, p_trace = fit_probe(
            x, train, val, test, pgen_target, alphas,
            lambda pred, indices: pgen_metrics(pred, pgen_target[indices], normalizer.pgen_std)[0]["task_loss"],
        )
        seq_val, _ = sequence_metrics(sequence_val_pred, seq_target[val])
        seq_test, seq_rows = sequence_metrics(sequence_test_pred, seq_target[test])
        t_val, _ = tcremp_metrics(t_val_pred, tcremp_target[val])
        t_test, t_rows = tcremp_metrics(t_test_pred, tcremp_target[test])
        p_val, _ = pgen_metrics(p_val_pred, pgen_target[val], normalizer.pgen_std)
        p_test, p_rows = pgen_metrics(p_test_pred, pgen_target[test], normalizer.pgen_std)
        validation_losses[name] = {
            "sequence": seq_val["task_loss"], "tcremp": t_val["task_loss"], "pgen": p_val["task_loss"],
        }
        test_losses[name] = {
            "sequence": seq_test["task_loss"], "tcremp": t_test["task_loss"], "pgen": p_test["task_loss"],
        }
        per_row[name] = {"sequence": seq_rows, "tcremp": t_rows, "pgen": p_rows}
        dim = x.shape[1]
        results[name] = {
            "validation": {"sequence": seq_val, "tcremp": t_val, "pgen": p_val},
            "test": {"sequence": seq_test, "tcremp": t_test, "pgen": p_test},
            "selected_alpha_validation_only": {
                "sequence": sequence_alpha, "tcremp": t_alpha, "pgen": p_alpha,
            },
            "validation_alpha_trace": {
                "sequence": sequence_trace, "tcremp": t_trace, "pgen": p_trace,
            },
            "probe_parameter_counts": {
                "sequence": int((dim + 1) * seq_one_hot.shape[1]),
                "tcremp": int((dim + 1) * tcremp_target.shape[1]),
                "pgen": int(dim + 1),
            },
        }
        representation_meta[name] = {**scaling}
        del x

    # Train-derived constant probes are the fixed null representation.
    majority = np.zeros((len(val), seq_target.shape[1] * N_CLASSES), dtype=np.float32)
    majority_test = np.zeros((len(test), seq_target.shape[1] * N_CLASSES), dtype=np.float32)
    train_majority = np.apply_along_axis(lambda x: np.bincount(x, minlength=N_CLASSES).argmax(), 0,
                                         seq_target[train])
    for position, cls in enumerate(train_majority):
        majority[:, position * N_CLASSES + cls] = 1
        majority_test[:, position * N_CLASSES + cls] = 1
    null_val_seq, _ = sequence_metrics(majority, seq_target[val])
    null_test_seq, null_seq_rows = sequence_metrics(majority_test, seq_target[test])
    zero_t_val = np.zeros_like(tcremp_target[val])
    zero_t_test = np.zeros_like(tcremp_target[test])
    null_val_t, _ = tcremp_metrics(zero_t_val, tcremp_target[val])
    null_test_t, null_t_rows = tcremp_metrics(zero_t_test, tcremp_target[test])
    zero_p_val = np.zeros_like(pgen_target[val])
    zero_p_test = np.zeros_like(pgen_target[test])
    null_val_p, _ = pgen_metrics(zero_p_val, pgen_target[val], normalizer.pgen_std)
    null_test_p, null_p_rows = pgen_metrics(zero_p_test, pgen_target[test], normalizer.pgen_std)
    validation_null = {"sequence": null_val_seq["task_loss"], "tcremp": null_val_t["task_loss"],
                       "pgen": null_val_p["task_loss"]}
    test_null = {"sequence": null_test_seq["task_loss"], "tcremp": null_test_t["task_loss"],
                 "pgen": null_test_p["task_loss"]}
    per_row["constant_null"] = {"sequence": null_seq_rows, "tcremp": null_t_rows, "pgen": null_p_rows}
    validation_specialist = {task: validation_losses[name][task] for task, name in SPECIALISTS.items()}
    for name in results:
        results[name]["validation_aggregate"] = aggregate_losses(
            validation_losses[name], validation_specialist, validation_null,
        )
        # Validation-defined reference and scale are intentionally reused on test.
        results[name]["test_aggregate_validation_fixed_scale"] = aggregate_losses(
            test_losses[name], validation_specialist, validation_null,
        )

    parameter_counts = {name: int(report["parameter_count"]) for name, report in model_report.items()}
    for name in model_report:
        representation_meta[name].update({
            "encoder_count": 1, "encoder_parameter_count": parameter_counts[name],
            "representation_dimension": 128,
        })
    representation_meta["concat_r_t_p"].update({
        "encoder_count": 3,
        "encoder_parameter_count": sum(parameter_counts[name] for name in ("r", "t", "p")),
        "representation_dimension": 384,
    })
    representation_meta["pca128_concat_r_t_p"].update({
        "encoder_count": 3,
        "encoder_parameter_count": sum(parameter_counts[name] for name in ("r", "t", "p")),
        "representation_dimension": 128,
        "train_only_pca_parameters": int(pca.components_.size + pca.mean_.size),
        "train_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    })
    representation_meta["composition_length"].update({
        "encoder_count": 0, "encoder_parameter_count": 0, **comp_meta,
    })
    representation_meta["composition_length_vj"].update({
        "encoder_count": 0, "encoder_parameter_count": 0, **comp_vj_meta,
        "annotation_aware_modality": True,
    })

    bootstrap = bootstrap_aggregate_deltas(
        {name: per_row[name] for name in results}, validation_specialist, validation_null,
        args.bootstrap, args.seed + 1000,
    )
    input_paths = [
        args.data_dir / "dataset.parquet", args.data_dir / "embeddings.npy",
        args.data_dir / "manifests" / "train.tsv", args.data_dir / "manifests" / "val.tsv",
        args.data_dir / "manifests" / "test.tsv", args.normalizer, args.tokenizer,
        *[path for pair in models.values() for path in pair],
    ]
    payload = {
        "status": "complete",
        "study": "frozen DATA-ANCHOR latent sufficiency with fresh linear probes",
        "selection": "ridge alpha selected on validation; test evaluated once; all models reported",
        "fixed_specialists": SPECIALISTS,
        "task_loss_orientation": "lower is better: sequence token error, TCRemP standardized MSE, pgen standardized MSE",
        "aggregate_definition": {
            "regret": "(loss - fixed validation specialist loss)/(validation constant-null loss - validation specialist loss)",
            "hypervolume": "product over tasks of clipped (validation-null - loss)/(validation-null - validation-specialist)",
            "test_scale": "validation specialist and null values reused unchanged",
        },
        "split_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "split_overlap": 0,
        "alphas": alphas,
        "models": model_report,
        "embedding_extraction": extraction,
        "representations": representation_meta,
        "validation_null_losses": validation_null,
        "test_null_losses_descriptive": test_null,
        "validation_specialist_losses": validation_specialist,
        "results": results,
        "paired_test_clonotype_bootstrap": bootstrap,
        "bootstrap_replicates": args.bootstrap,
        "limitations": [
            "All encoders are single seed 42; bootstrap intervals do not quantify training-seed variability.",
            "Linear-probe sufficiency does not establish causal biological usefulness.",
            "The V/J control is annotation-aware and is not an equivalent sequence-only inference condition.",
        ],
        "runtime_seconds": float(time.time() - started),
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME")},
        "inputs": file_manifest(input_paths),
        "model_config": asdict(model_config),
    }
    results_path = args.output_dir / "RESULTS.json"
    write_json(results_path, payload)
    summary_rows = []
    for name, result in results.items():
        row = {"representation": name}
        for task in TASKS:
            row[f"val_{task}_loss"] = result["validation"][task]["task_loss"]
            row[f"test_{task}_loss"] = result["test"][task]["task_loss"]
        row.update({f"test_{key}": value for key, value in
                    result["test_aggregate_validation_fixed_scale"].items()
                    if isinstance(value, (int, float))})
        summary_rows.append(row)
    summary_path = args.output_dir / "summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    outputs = file_manifest([path for path in args.output_dir.iterdir() if path.is_file()])
    write_json(args.output_dir / "OUTPUT_HASHES.json", outputs)


if __name__ == "__main__":
    main()
