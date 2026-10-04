"""Preregistered secondary tests of matched frozen DATA-ANCHOR representations.

This benchmark intentionally asks complementary *representation sufficiency*
questions on the locked TRB train/validation/test split.  It does not search for
a task on which RTP wins.  Probe hyperparameters are selected on validation;
test labels are used once for reporting and never for representation/model
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

MODELS = ("r", "t", "p", "rt", "rp", "tp", "rtp")
AA = "ACDEFGHIKLMNPQRSTVWY"
CONSERVATIVE_GROUPS = ("AILMV", "FWY", "STNQ", "KRH", "DE", "CGP")
FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
ALPHAS = (0.1, 10.0, 1000.0)
JOINT_QUANTILES = (0.20, 0.25, 0.30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_finite_tree(value: object, where: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{where}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"Non-finite value at {where}: {value}")


def preregistration() -> dict:
    return {
        "status": "preregistered_before_test_evaluation",
        "models": list(MODELS),
        "primary_comparison": "RTP versus every other matched encoder; all conditions reported",
        "no_selection_rule": "No track, subgroup, representation, or test metric may be omitted based on RTP rank.",
        "tracks": {
            "low_label": {
                "nested_train_fractions": list(FRACTIONS),
                "subset_rule": "prefixes of one seed-42 permutation of the locked train manifest",
                "probe": "ridge with intercept; representation standardized on each train subset",
                "alpha_grid_validation_only": list(ALPHAS),
                "targets": {
                    "sequence_content": "20 AA fractions plus standardized CDR3 length",
                    "tcremp": "fixed seed-42 random-sign 128D sketch of locked train-standardized 9000D TCRemP vector",
                    "pgen": "locked train-standardized log10_pgen_1mm",
                },
                "metrics": ["sequence RMSE/cosine and length MAE", "TCRemP MSE/cosine", "pgen RMSE/MAE/R2"],
            },
            "strata": {
                "fit": "full locked train; alpha selected on validation globally, never within test subgroup",
                "length_bins": ["<=12", "13-15", "16-18", ">=19"],
                "gene_families": "TRBV/TRBJ prefix before '-' and '*'; report test groups n>=100",
                "pgen_bins": "quartiles defined from locked train targets",
            },
            "joint_retrieval": {
                "queries": "all locked test rows",
                "gallery": "256 deterministic seed-42 test candidates per query, excluding self",
                "relevance": (
                    "simultaneously below the query-specific q-quantile for sequence 2-mer/length distance, "
                    "TCRemP cosine distance, and absolute standardized pgen difference"
                ),
                "primary_q": 0.25,
                "sensitivity_q": list(JOINT_QUANTILES),
                "metrics": ["P@1", "P@5", "P@10", "MAP", "eligible query count"],
                "caveat": "Target-defined relevance measures joint target sufficiency, not independent biology.",
            },
            "robustness": {
                "rows": "deterministic 5000-row test subset",
                "perturbations": [
                    "one within-biochemical-group AA substitution",
                    "one outside-group AA substitution",
                    "one central encoder token replaced by UNK id 3",
                ],
                "no_indels": True,
                "metrics": ["cosine drift", "Euclidean drift", "nonconservative greater than conservative fraction"],
            },
            "synthetic_discrimination": {
                "status": "disabled unless an explicit provenance manifest is supplied",
                "required_manifest": ["source_path", "source_sha256", "generator", "generator_sha256", "benchmark_overlap=0"],
                "reason": "No nearby file is accepted as synthetic merely from its filename.",
            },
        },
        "representations": [
            *MODELS,
            "concat_r_t_p_384d",
            "pca128_of_concat_fit_train_only",
            "aa_composition_length_vj_train_schema",
        ],
        "caveats": [
            "All neural models are one training seed; sampling intervals do not represent training-seed variation.",
            "The low-cost sequence probe tests composition/length, not exact autoregressive reconstruction.",
            "TCRemP probing/relevance uses one preregistered random projection for CPU feasibility; it is not the full 9000D target.",
            "Concat/PCA use three encoders and are efficiency upper bounds, not single-encoder peers.",
        ],
    }


def nested_subsets(indices: np.ndarray, fractions: tuple[float, ...], seed: int) -> dict[float, np.ndarray]:
    order = np.random.default_rng(seed).permutation(np.asarray(indices, dtype=np.int64))
    result = {}
    previous: set[int] = set()
    for fraction in fractions:
        size = max(100, int(round(len(order) * fraction)))
        subset = order[: min(size, len(order))]
        if not previous.issubset(set(map(int, subset))):
            raise AssertionError("Nested subset construction failed.")
        previous = set(map(int, subset))
        result[fraction] = subset
    return result


def sequence_features(sequences: pd.Series) -> np.ndarray:
    output = np.zeros((len(sequences), len(AA) + 1), dtype=np.float32)
    for row, sequence in enumerate(sequences.astype(str)):
        for column, aa in enumerate(AA):
            output[row, column] = sequence.count(aa) / len(sequence)
        output[row, -1] = len(sequence)
    return output


def bigram_features(sequences: pd.Series) -> np.ndarray:
    lookup = {a + b: i for i, (a, b) in enumerate((a, b) for a in AA for b in AA)}
    output = np.zeros((len(sequences), len(lookup)), dtype=np.float32)
    for row, sequence in enumerate(sequences.astype(str)):
        for left, right in zip(sequence, sequence[1:]):
            output[row, lookup[left + right]] += 1
        norm = np.linalg.norm(output[row])
        if norm:
            output[row] /= norm
    return output


def gene_family(value: object) -> str:
    text = str(value).upper().split("*", 1)[0]
    return text.split("-", 1)[0]


def control_features(table: pd.DataFrame, train: np.ndarray) -> np.ndarray:
    basic = sequence_features(table["junction_aa"])
    mean = basic[train].mean(axis=0)
    std = basic[train].std(axis=0)
    basic = (basic - mean) / np.where(std < 1e-8, 1.0, std)
    categories = {}
    for column in ("v_call", "j_call"):
        values = table[column].map(gene_family).to_numpy()
        categories[column] = {value: i for i, value in enumerate(sorted(set(values[train])))}
    width = sum(len(value) for value in categories.values())
    genes = np.zeros((len(table), width), dtype=np.float32)
    offset = 0
    for column in ("v_call", "j_call"):
        values = table[column].map(gene_family).to_numpy()
        for row, value in enumerate(values):
            index = categories[column].get(value)
            if index is not None:
                genes[row, offset + index] = 1
        offset += len(categories[column])
    return np.concatenate((basic, genes), axis=1)


def standardize_targets(
    table: pd.DataFrame, tcremp: np.ndarray, standardizer: object,
) -> tuple[np.ndarray, dict[str, slice]]:
    sequence = sequence_features(table["junction_aa"]).astype(np.float64)
    sequence[:, -1] = (sequence[:, -1] - table.loc[table["split"] == "train", "junction_aa"].str.len().mean()) / max(
        float(table.loc[table["split"] == "train", "junction_aa"].str.len().std(ddof=0)), 1e-8
    )
    t = np.asarray(tcremp, dtype=np.float64)
    p = ((table[standardizer.pgen_target].to_numpy(np.float64) - standardizer.pgen_mean) / standardizer.pgen_std)[:, None]
    y = np.concatenate((sequence, t, p), axis=1)
    slices = {
        "sequence": slice(0, sequence.shape[1]),
        "tcremp": slice(sequence.shape[1], sequence.shape[1] + t.shape[1]),
        "pgen": slice(sequence.shape[1] + t.shape[1], y.shape[1]),
    }
    return y, slices


def project_tcremp(
    tcremp: np.ndarray, standardizer: object, dimension: int = 128, seed: int = 42,
    chunk_size: int = 512,
) -> tuple[np.ndarray, dict]:
    """Apply one fixed Johnson-Lindenstrauss sign sketch after locked standardization."""
    source_dim = int(tcremp.shape[1])
    rng = np.random.default_rng(seed)
    projection = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(source_dim, dimension))
    projection /= np.sqrt(dimension)
    output = np.empty((len(tcremp), dimension), dtype=np.float32)
    for start in range(0, len(tcremp), chunk_size):
        stop = min(start + chunk_size, len(tcremp))
        standardized = (
            np.asarray(tcremp[start:stop], dtype=np.float32) - standardizer.tcremp_mean
        ) / standardizer.tcremp_std
        output[start:stop] = standardized @ projection
    if not np.isfinite(output).all():
        raise ValueError("TCRemP random projection emitted non-finite values.")
    return output, {
        "method": "dense Rademacher random projection after locked train standardization",
        "seed": seed, "source_dimension": source_dim, "output_dimension": dimension,
        "scale": "1/sqrt(output_dimension)",
        "projection_sha256": hashlib.sha256(projection.tobytes()).hexdigest(),
    }


def ridge_coefficients(x: np.ndarray, y: np.ndarray, indices: np.ndarray, alpha: float) -> dict:
    xx = np.asarray(x[indices], dtype=np.float64)
    yy = np.asarray(y[indices], dtype=np.float64)
    mean_x, scale_x = xx.mean(axis=0), xx.std(axis=0)
    scale_x = np.where(scale_x < 1e-8, 1.0, scale_x)
    mean_y = yy.mean(axis=0)
    z = (xx - mean_x) / scale_x
    gram = z.T @ z
    beta = np.linalg.solve(gram + np.eye(gram.shape[0]) * alpha, z.T @ (yy - mean_y))
    return {"mean_x": mean_x, "scale_x": scale_x, "mean_y": mean_y, "beta": beta, "alpha": alpha}


def ridge_predict(fit: dict, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    z = (np.asarray(x[indices], dtype=np.float64) - fit["mean_x"]) / fit["scale_x"]
    return z @ fit["beta"] + fit["mean_y"]


def cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > 0
    return float(np.mean(np.sum(a[valid] * b[valid], axis=1) / denominator[valid]))


def target_metrics(truth: np.ndarray, predicted: np.ndarray, slices: dict[str, slice]) -> dict:
    sequence_true, sequence_pred = truth[:, slices["sequence"]], predicted[:, slices["sequence"]]
    t_true, t_pred = truth[:, slices["tcremp"]], predicted[:, slices["tcremp"]]
    p_true, p_pred = truth[:, slices["pgen"]].ravel(), predicted[:, slices["pgen"]].ravel()
    residual = p_true - p_pred
    return {
        "sequence": {
            "aa_composition_rmse": float(np.sqrt(np.mean((sequence_true[:, :-1] - sequence_pred[:, :-1]) ** 2))),
            "aa_composition_cosine": cosine_mean(sequence_true[:, :-1], sequence_pred[:, :-1]),
            "standardized_length_mae": float(np.mean(np.abs(sequence_true[:, -1] - sequence_pred[:, -1]))),
        },
        "tcremp": {
            "standardized_mse": float(np.mean((t_true - t_pred) ** 2)),
            "cosine": cosine_mean(t_true, t_pred),
        },
        "pgen": {
            "standardized_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "standardized_mae": float(np.mean(np.abs(residual))),
            "r2": float(1 - np.sum(residual ** 2) / np.sum((p_true - p_true.mean()) ** 2)),
        },
    }


def selection_losses(metrics: dict) -> dict[str, float]:
    return {
        "sequence": metrics["sequence"]["aa_composition_rmse"] + metrics["sequence"]["standardized_length_mae"],
        "tcremp": metrics["tcremp"]["standardized_mse"],
        "pgen": metrics["pgen"]["standardized_rmse"],
    }


def choose_task_fits(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, slices: dict[str, slice],
) -> tuple[dict[str, dict], dict]:
    fits = {alpha: ridge_coefficients(x, y, train, alpha) for alpha in ALPHAS}
    validations = {alpha: target_metrics(y[val], ridge_predict(fit, x, val), slices) for alpha, fit in fits.items()}
    selected = {}
    report = {}
    for task in ("sequence", "tcremp", "pgen"):
        alpha = min(ALPHAS, key=lambda value: (selection_losses(validations[value])[task], value))
        selected[task] = fits[alpha]
        report[task] = {"alpha": alpha, "validation": validations[alpha][task]}
    return selected, report


def merged_predictions(fits: dict[str, dict], x: np.ndarray, indices: np.ndarray, slices: dict[str, slice]) -> np.ndarray:
    width = max(value.stop for value in slices.values())
    result = np.empty((len(indices), width), dtype=np.float64)
    for task, section in slices.items():
        result[:, section] = ridge_predict(fits[task], x, indices)[:, section]
    return result


def length_strata(table: pd.DataFrame, test: np.ndarray) -> dict[str, np.ndarray]:
    lengths = table.iloc[test]["junction_aa"].str.len().to_numpy()
    rules = {
        "le12": lengths <= 12,
        "13_15": (lengths >= 13) & (lengths <= 15),
        "16_18": (lengths >= 16) & (lengths <= 18),
        "ge19": lengths >= 19,
    }
    return {name: np.flatnonzero(mask) for name, mask in rules.items() if mask.sum() >= 100}


def categorical_strata(table: pd.DataFrame, test: np.ndarray, column: str) -> dict[str, np.ndarray]:
    values = table.iloc[test][column].map(gene_family).to_numpy()
    return {str(value): np.flatnonzero(values == value) for value in sorted(set(values)) if np.sum(values == value) >= 100}


def pgen_strata(table: pd.DataFrame, train: np.ndarray, test: np.ndarray, target: str) -> dict[str, np.ndarray]:
    edges = np.quantile(table.iloc[train][target].to_numpy(float), [0.25, 0.5, 0.75])
    groups = np.digitize(table.iloc[test][target].to_numpy(float), edges, right=True)
    return {f"q{value + 1}": np.flatnonzero(groups == value) for value in range(4)}


def make_candidate_gallery(n: int, width: int, seed: int) -> np.ndarray:
    if n <= width:
        raise ValueError("Test set must be larger than the fixed candidate gallery.")
    gallery = np.empty((n, width), dtype=np.int32)
    for query in range(n):
        rng = np.random.default_rng(seed + query * 1_000_003)
        selected = rng.choice(n - 1, width, replace=False)
        gallery[query] = selected + (selected >= query)
    return gallery


def cosine_candidate_distance(features: np.ndarray, gallery: np.ndarray, chunk: int = 1024) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = values / np.where(norms == 0, 1.0, norms)
    result = np.empty(gallery.shape, dtype=np.float32)
    for start in range(0, len(values), chunk):
        stop = min(start + chunk, len(values))
        result[start:stop] = 1 - np.sum(unit[start:stop, None, :] * unit[gallery[start:stop]], axis=2)
    return np.maximum(result, 0)


def joint_relevance(
    sequences: pd.Series, tcremp: np.ndarray, pgen: np.ndarray, gallery: np.ndarray, quantile: float,
) -> np.ndarray:
    bigrams = bigram_features(sequences)
    seq = cosine_candidate_distance(bigrams, gallery)
    lengths = sequences.str.len().to_numpy(float)
    seq += 0.2 * np.abs(lengths[:, None] - lengths[gallery]) / 40.0
    t_distance = cosine_candidate_distance(tcremp, gallery)
    p_distance = np.abs(pgen[:, None] - pgen[gallery])
    thresholds = [np.quantile(value, quantile, axis=1, keepdims=True) for value in (seq, t_distance, p_distance)]
    return (seq <= thresholds[0]) & (t_distance <= thresholds[1]) & (p_distance <= thresholds[2])


def retrieval_metrics(distance: np.ndarray, relevant: np.ndarray) -> dict:
    rows = []
    for query in range(len(distance)):
        positives = int(relevant[query].sum())
        if positives == 0:
            continue
        order = np.argsort(distance[query], kind="mergesort")
        hits = relevant[query, order].astype(float)
        precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        rows.append([
            hits[0], hits[:5].mean(), hits[:10].mean(), float((precision * hits).sum() / positives), positives
        ])
    values = np.asarray(rows, dtype=float)
    return {
        "eligible_queries": int(len(values)),
        "precision_at_1": float(values[:, 0].mean()),
        "precision_at_5": float(values[:, 1].mean()),
        "precision_at_10": float(values[:, 2].mean()),
        "map": float(values[:, 3].mean()),
        "mean_relevant_candidates": float(values[:, 4].mean()),
    }


def deterministic_substitutions(sequence: str, seed: int) -> tuple[str, str]:
    if len(sequence) < 3:
        raise ValueError("Robustness substitution requires a sequence of length at least three.")
    rng = np.random.default_rng(seed)
    position = int(rng.integers(1, len(sequence) - 1))
    source = sequence[position]
    group = next(group for group in CONSERVATIVE_GROUPS if source in group)
    conservative_pool = [aa for aa in group if aa != source]
    if not conservative_pool:
        conservative_pool = [aa for aa in AA if aa != source]
    nonconservative_pool = [aa for aa in AA if aa not in group]
    conservative = sequence[:position] + conservative_pool[int(rng.integers(len(conservative_pool)))] + sequence[position + 1:]
    nonconservative = sequence[:position] + nonconservative_pool[int(rng.integers(len(nonconservative_pool)))] + sequence[position + 1:]
    return conservative, nonconservative


def embedding_drift(original: np.ndarray, altered: np.ndarray) -> dict:
    euclidean = np.linalg.norm(original - altered, axis=1)
    cosine = 1 - np.sum(original * altered, axis=1) / (
        np.linalg.norm(original, axis=1) * np.linalg.norm(altered, axis=1)
    )
    return {
        "euclidean_mean": float(euclidean.mean()), "euclidean_median": float(np.median(euclidean)),
        "cosine_mean": float(cosine.mean()), "cosine_median": float(np.median(cosine)),
    }


def extract_altered_embeddings(
    token_sets: dict[str, list[list[int]]], models: dict, config, batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    import torch
    from rtp_codec.models.codec import RTPCodecTransformer
    from rtp_codec.tokenization.character import PAD_ID

    result = {kind: {} for kind in token_sets}
    for name, (checkpoint_path, _) in models.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["model_config"] != asdict(config):
            raise ValueError(f"{name} altered-input extraction found config mismatch.")
        model = RTPCodecTransformer(config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        with torch.no_grad():
            for kind, rows in token_sets.items():
                blocks = []
                for start in range(0, len(rows), batch_size):
                    block = rows[start:start + batch_size]
                    width = max(map(len, block))
                    tokens = torch.full((len(block), width), PAD_ID, dtype=torch.long)
                    for index, ids in enumerate(block):
                        tokens[index, :len(ids)] = torch.tensor(ids)
                    blocks.append(model.encode(tokens, tokens.ne(PAD_ID)).numpy().astype(np.float32))
                result[kind][name] = np.concatenate(blocks)
        del checkpoint, model
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--accepted-preflight", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict, object, dict]:
    from rtp_codec.benchmarks.downstream.evaluate_vdjdb_frozen import parse_models, validate_run_configs
    from rtp_codec.data.multitask import (
        TargetStandardizer,
        load_prepared_benchmark,
        resolve_encoder_tokenizer,
        select_split_indices,
    )

    table, tcremp = load_prepared_benchmark(args.data_dir)
    indices = {split: select_split_indices(table, args.data_dir, split) for split in ("train", "val", "test")}
    if any(np.intersect1d(indices[a], indices[b]).size for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Locked split manifests overlap.")
    models = parse_models(args.model)
    if set(models) != set(MODELS):
        raise ValueError("Full seven-condition factorial required.")
    config, model_report = validate_run_configs(models)
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(args.tokenizer))
    if tokenizer.vocab_size != config.input_vocab_size:
        raise ValueError("Tokenizer/checkpoint vocabulary mismatch.")
    standardizer = TargetStandardizer.load(args.data_dir / "target_standardizer.npz")
    if standardizer.train_rows != len(indices["train"]) or standardizer.pgen_target != "log10_pgen_1mm":
        raise ValueError("Locked train-only normalizer mismatch.")
    return table, tcremp, indices, config, {"models": models, "report": model_report, "tokenizer": tokenizer, "standardizer": standardizer}


def main() -> None:
    from rtp_codec.benchmarks.downstream.evaluate_vdjdb_frozen import extract_embeddings

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    table, tcremp, indices, config, runtime = validate_inputs(args)
    registration = preregistration()
    stable_json(args.output_dir / "PREREGISTRATION.json", registration)
    input_hashes = {
        "dataset": sha256(args.data_dir / "dataset.parquet"),
        "tcremp": sha256(args.data_dir / "embeddings.npy"),
        "normalizer": sha256(args.data_dir / "target_standardizer.npz"),
        "tokenizer": sha256(args.tokenizer),
        "splits": {split: sha256(args.data_dir / "manifests" / f"{split}.tsv") for split in indices},
    }
    preflight = {
        "status": "accepted_for_secondary_frozen_evaluation",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "rows": len(table), "splits": {key: len(value) for key, value in indices.items()},
        "input_hashes": input_hashes, "model_report": runtime["report"],
        "preregistration_sha256": sha256(args.output_dir / "PREREGISTRATION.json"),
        "synthetic_track": "blocked_no_validated_source_manifest",
    }
    stable_json(args.output_dir / "PREFLIGHT.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    if args.accepted_preflight is None:
        raise ValueError("--accepted-preflight is required for evaluation.")
    accepted = json.loads(args.accepted_preflight.read_text(encoding="utf-8"))
    if accepted["status"] != "accepted_for_secondary_frozen_evaluation" or accepted["input_hashes"] != input_hashes:
        raise ValueError("Accepted preflight is absent, stale, or refers to different inputs.")

    cohort = pd.DataFrame({"cdr3": table["junction_aa"].astype(str)})
    embeddings, extraction = extract_embeddings(
        cohort, runtime["models"], config, args.tokenizer, args.batch_size, args.output_dir
    )
    representations = dict(embeddings)
    concat = np.concatenate([embeddings[name] for name in ("r", "t", "p")], axis=1)
    representations["concat_r_t_p"] = concat
    pca = PCA(n_components=128, svd_solver="randomized", random_state=args.seed)
    pca.fit(concat[indices["train"]])
    representations["pca128_concat_r_t_p"] = pca.transform(concat).astype(np.float32)
    representations["sequence_length_vj_control"] = control_features(table, indices["train"])
    np.save(args.output_dir / "pca_components.npy", pca.components_, allow_pickle=False)

    tcremp_sketch, tcremp_projection_report = project_tcremp(
        tcremp, runtime["standardizer"], dimension=128, seed=args.seed
    )
    np.save(args.output_dir / "tcremp_random_projection.npy", tcremp_sketch, allow_pickle=False)
    y, slices = standardize_targets(table, tcremp_sketch, runtime["standardizer"])
    subsets = nested_subsets(indices["train"], FRACTIONS, args.seed)
    efficiency, final_fits = {}, {}
    rows = []
    for representation, features in representations.items():
        efficiency[representation] = {}
        for fraction, subset in subsets.items():
            fits, validation = choose_task_fits(features, y, subset, indices["val"], slices)
            prediction = merged_predictions(fits, features, indices["test"], slices)
            metrics = target_metrics(y[indices["test"]], prediction, slices)
            efficiency[representation][str(fraction)] = {
                "train_rows": len(subset), "validation_selection": validation, "test": metrics,
            }
            for task in metrics:
                rows.append({"representation": representation, "fraction": fraction, "train_rows": len(subset),
                             "task": task, **metrics[task]})
            if fraction == 1.0:
                final_fits[representation] = fits
    pd.DataFrame(rows).to_csv(args.output_dir / "low_label_curves.tsv", sep="\t", index=False)
    stable_json(args.output_dir / "low_label.json", efficiency)

    strata_definitions = {
        "length": length_strata(table, indices["test"]),
        "v_family": categorical_strata(table, indices["test"], "v_call"),
        "j_family": categorical_strata(table, indices["test"], "j_call"),
        "pgen_train_quartile": pgen_strata(table, indices["train"], indices["test"], runtime["standardizer"].pgen_target),
    }
    strata_rows = []
    for representation, features in representations.items():
        prediction = merged_predictions(final_fits[representation], features, indices["test"], slices)
        test_truth = y[indices["test"]]
        for stratum_type, groups in strata_definitions.items():
            for stratum, positions in groups.items():
                metrics = target_metrics(test_truth[positions], prediction[positions], slices)
                for task, values in metrics.items():
                    strata_rows.append({"representation": representation, "stratum_type": stratum_type,
                                        "stratum": stratum, "rows": len(positions), "task": task, **values})
    pd.DataFrame(strata_rows).to_csv(args.output_dir / "stratified_test_metrics.tsv", sep="\t", index=False)

    test = indices["test"]
    gallery = make_candidate_gallery(len(test), 256, args.seed)
    np.save(args.output_dir / "joint_gallery.npy", gallery, allow_pickle=False)
    test_sequences = table.iloc[test]["junction_aa"].reset_index(drop=True)
    test_tcremp = tcremp_sketch[test]
    test_pgen = y[test, slices["pgen"]].ravel()
    retrieval = {}
    for quantile in JOINT_QUANTILES:
        relevant = joint_relevance(test_sequences, test_tcremp, test_pgen, gallery, quantile)
        retrieval[str(quantile)] = {
            name: retrieval_metrics(cosine_candidate_distance(features[test], gallery), relevant)
            for name, features in representations.items()
        }
        retrieval[str(quantile)]["relevance"] = {
            "eligible_queries": int(np.sum(relevant.sum(axis=1) > 0)),
            "mean_relevant": float(relevant.sum(axis=1).mean()),
        }
    stable_json(args.output_dir / "joint_retrieval.json", retrieval)

    robust_positions = np.random.default_rng(args.seed).choice(test, size=min(5000, len(test)), replace=False)
    robust_sequences = table.iloc[robust_positions]["junction_aa"].astype(str).tolist()
    conservative, nonconservative = zip(*[
        deterministic_substitutions(sequence, args.seed + row * 17)
        for row, sequence in enumerate(robust_sequences)
    ])
    tokenizer = runtime["tokenizer"]
    token_sets = {
        "original": [tokenizer.encode(value, config.max_sequence_len) for value in robust_sequences],
        "conservative": [tokenizer.encode(value, config.max_sequence_len) for value in conservative],
        "nonconservative": [tokenizer.encode(value, config.max_sequence_len) for value in nonconservative],
    }
    masked = []
    for row, tokens in enumerate(token_sets["original"]):
        copy = list(tokens)
        position = int(np.random.default_rng(args.seed + row * 31).integers(len(copy)))
        copy[position] = 3
        masked.append(copy)
    token_sets["token_mask"] = masked
    altered = extract_altered_embeddings(token_sets, runtime["models"], config, args.batch_size)
    for kind in altered:
        altered[kind]["concat_r_t_p"] = np.concatenate(
            [altered[kind][name] for name in ("r", "t", "p")], axis=1
        )
        altered[kind]["pca128_concat_r_t_p"] = pca.transform(
            altered[kind]["concat_r_t_p"]
        ).astype(np.float32)
    robustness = {}
    for name in (*MODELS, "concat_r_t_p", "pca128_concat_r_t_p"):
        original = altered["original"][name]
        conservative_drift = np.linalg.norm(original - altered["conservative"][name], axis=1)
        nonconservative_drift = np.linalg.norm(original - altered["nonconservative"][name], axis=1)
        robustness[name] = {
            kind: embedding_drift(original, altered[kind][name])
            for kind in ("conservative", "nonconservative", "token_mask")
        }
        robustness[name]["nonconservative_euclidean_gt_conservative_fraction"] = float(
            np.mean(nonconservative_drift > conservative_drift)
        )
    stable_json(args.output_dir / "robustness.json", robustness)

    report = {
        "status": "complete", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "study": "secondary frozen-representation benchmark on locked TRB splits",
        "input_hashes": input_hashes, "models": runtime["report"], "extraction": extraction,
        "representations": {name: list(value.shape) for name, value in representations.items()},
        "pca": {"fit_split": "train", "explained_variance_sum": float(pca.explained_variance_ratio_.sum())},
        "tcremp_projection": tcremp_projection_report,
        "tracks": {
            "low_label": "low_label.json", "stratified": "stratified_test_metrics.tsv",
            "joint_retrieval": "joint_retrieval.json", "robustness": "robustness.json",
            "synthetic_discrimination": "blocked_no_validated_source_manifest",
        },
        "preregistration": {"path": str(args.output_dir / "PREREGISTRATION.json"),
                            "sha256": sha256(args.output_dir / "PREREGISTRATION.json")},
        "caveats": registration["caveats"], "nonfinite_scan": "passed",
    }
    assert_finite_tree(efficiency, "efficiency")
    assert_finite_tree(retrieval, "retrieval")
    assert_finite_tree(robustness, "robustness")
    report["artifact_sha256"] = {
        path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()
    }
    stable_json(args.output_dir / "RESULTS.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
