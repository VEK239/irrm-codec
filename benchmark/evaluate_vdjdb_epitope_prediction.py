"""Frozen-embedding multiclass epitope prediction on the manuscript VDJdb cohort.

The selected rows are exactly ``original_indices`` from the published uniform-100
separation evaluation.  No encoder is loaded or fine-tuned here.  Every dense
representation uses the same cached folds, training-only scaling, and the same
small, training-only C selection procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


NATIVE_NAMES = ("r", "t", "p", "rt", "rp", "tp", "rtp")
EXTERNAL_FILES = {
    "esm2_35m": "esm2_35m.npy",
    "tcr_bert": "tcr_bert.npy",
    "sceptr_cdr3": "sceptr_cdr3.npy",
}
DISPLAY = {
    "r": "R", "t": "T", "p": "P", "rt": "RT", "rp": "RP", "tp": "TP",
    "rtp": "RTP", "esm2_35m": "ESM2-35M", "tcr_bert": "TCR-BERT",
    "sceptr_cdr3": "SCEPTR (CDR3-only)", "kmer_tfidf": "3-mer TF-IDF",
}
PRIMARY_ORDER = (
    "rtp", "rp", "r", "t", "p", "rt", "tp", "tcr_bert", "sceptr_cdr3",
    "esm2_35m", "kmer_tfidf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--uniform-evaluation", type=Path, required=True)
    parser.add_argument("--native-evaluation", type=Path, required=True)
    parser.add_argument("--external-representations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--representations", nargs="+", default=list(PRIMARY_ORDER))
    parser.add_argument("--skip-similarity-grouped", action="store_true")
    return parser.parse_args()


def selected_cohort(cohort_path: Path, uniform_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    cohort = pd.read_csv(cohort_path, sep="\t")
    pairs = np.load(uniform_dir / "uniform_pairs.npz")
    original = np.asarray(pairs["original_indices"], dtype=np.int64)
    if len(original) != 65_756 or len(np.unique(original)) != len(original):
        raise ValueError("Uniform evaluation does not contain the locked 65,756 unique rows.")
    selected = cohort.iloc[original].copy().reset_index(drop=True)
    selected.insert(0, "prediction_row_index", np.arange(len(selected), dtype=np.int64))
    selected.insert(1, "source_cohort_row_index", original)
    if selected["label"].nunique() != 40 or selected.groupby("label").size().min() < 101:
        raise ValueError("Selected cohort differs from the locked 40-epitope support gate.")
    return selected, original


def audit_dataset(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    class_sizes = frame.groupby("label", sort=True).size().rename("sequences").reset_index()
    duplicate_rows = int(frame["cdr3"].duplicated(keep=False).sum())
    labels_per_sequence = frame.groupby("cdr3")["label"].nunique()
    conflicting = int((labels_per_sequence > 1).sum())
    sizes = class_sizes["sequences"].to_numpy()
    report = {
        "sequences": int(len(frame)),
        "epitopes": int(len(class_sizes)),
        "class_size_min": int(sizes.min()),
        "class_size_median": float(np.median(sizes)),
        "class_size_max": int(sizes.max()),
        "duplicate_cdr3_rows": duplicate_rows,
        "duplicate_cdr3_sequences_exist": bool(duplicate_rows),
        "identical_cdr3_with_different_labels": conflicting,
    }
    return report, class_sizes


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def redcea_component_groups(frame: pd.DataFrame) -> np.ndarray:
    """Connect rows sharing any retained REDCEA cluster identifier."""
    union = UnionFind(len(frame))
    owner: dict[str, int] = {}
    for index, raw in enumerate(frame["source_cids"].astype(str)):
        identifiers = [value for value in raw.split("|") if value]
        if not identifiers:
            identifiers = [str(frame.iloc[index]["cid"])]
        for identifier in identifiers:
            if identifier in owner:
                union.union(index, owner[identifier])
            else:
                owner[identifier] = index
    roots = np.array([union.find(i) for i in range(len(frame))], dtype=np.int64)
    _, groups = np.unique(roots, return_inverse=True)
    return groups.astype(np.int64)


def _valid_folds(folds: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> bool:
    expected = set(np.unique(y).tolist())
    return all(set(np.unique(y[train])) == expected and set(np.unique(y[test])) == expected
               for train, test in folds)


def make_splits(
    y: np.ndarray, exact_groups: np.ndarray, similarity_groups: np.ndarray,
    seeds: list[int], requested_folds: int, include_similarity: bool,
) -> tuple[dict[str, list[tuple[int, int, np.ndarray, np.ndarray]]], dict]:
    protocols: dict[str, list[tuple[int, int, np.ndarray, np.ndarray]]] = {
        "sequence_stratified": [], "exact_cdr3_grouped": []
    }
    report: dict[str, object] = {}
    for repeat, seed in enumerate(seeds):
        splitter = StratifiedKFold(requested_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(y)), y))
        for fold, (train, test) in enumerate(folds):
            item = (repeat, fold, train.astype(np.int64), test.astype(np.int64))
            protocols["sequence_stratified"].append(item)
            protocols["exact_cdr3_grouped"].append(item)
    if len(np.unique(exact_groups)) == len(exact_groups):
        report["exact_cdr3_grouped"] = {
            "alias_of": "sequence_stratified",
            "reason": "the locked cohort has one row per unique exact CDR3",
        }
    else:
        raise ValueError("Unexpected duplicate exact CDR3s in the locked deduplicated cohort.")

    if include_similarity:
        per_label_groups = pd.DataFrame({"y": y, "group": similarity_groups}).groupby("y")[
            "group"
        ].nunique()
        maximum = min(requested_folds, int(per_label_groups.min()))
        chosen: tuple[int, list[list[tuple[np.ndarray, np.ndarray]]]] | None = None
        for n_splits in range(maximum, 1, -1):
            repeated = []
            for seed in seeds:
                splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
                folds = list(splitter.split(np.zeros(len(y)), y, similarity_groups))
                repeated.append(folds)
            if all(_valid_folds(folds, y) for folds in repeated):
                chosen = n_splits, repeated
                break
        if chosen is None:
            raise ValueError("Could not construct class-complete REDCEA-grouped folds.")
        n_splits, repeated = chosen
        protocols["redcea_similarity_grouped"] = []
        for repeat, folds in enumerate(repeated):
            for fold, (train, test) in enumerate(folds):
                if np.intersect1d(similarity_groups[train], similarity_groups[test]).size:
                    raise AssertionError("REDCEA group leakage detected.")
                protocols["redcea_similarity_grouped"].append(
                    (repeat, fold, train.astype(np.int64), test.astype(np.int64))
                )
        report["redcea_similarity_grouped"] = {
            "folds": n_splits,
            "groups": int(len(np.unique(similarity_groups))),
            "minimum_groups_per_epitope": int(per_label_groups.min()),
            "definition": "connected components of rows sharing any retained REDCEA source_cid",
        }
    return protocols, report


def write_split_assignments(
    path: Path, protocols: dict[str, list[tuple[int, int, np.ndarray, np.ndarray]]],
    frame: pd.DataFrame,
) -> None:
    blocks = []
    for protocol, folds in protocols.items():
        by_repeat: dict[int, np.ndarray] = {}
        for repeat, fold, _, test in folds:
            by_repeat.setdefault(repeat, np.full(len(frame), -1, dtype=np.int16))[test] = fold
        for repeat, assignments in by_repeat.items():
            if np.any(assignments < 0):
                raise AssertionError(f"Incomplete assignments for {protocol} repeat {repeat}")
            blocks.append(pd.DataFrame({
                "protocol": protocol,
                "repeat": repeat,
                "prediction_row_index": frame["prediction_row_index"],
                "source_cohort_row_index": frame["source_cohort_row_index"],
                "cdr3": frame["cdr3"],
                "label": frame["label"],
                "test_fold": assignments,
            }))
    pd.concat(blocks, ignore_index=True).to_csv(path, index=False, lineterminator="\n")


def load_dense_matrix(
    name: str, original: np.ndarray, native_dir: Path, external_dir: Path, source_rows: int,
) -> np.ndarray:
    path = (native_dir / f"embeddings_{name}.npy") if name in NATIVE_NAMES else (
        external_dir / EXTERNAL_FILES[name]
    )
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[0] != source_rows:
        raise ValueError(f"{name} shape {matrix.shape} does not align to {source_rows} source rows")
    selected = np.asarray(matrix[original], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError(f"{name} contains non-finite values")
    return selected


def inner_split(
    train: np.ndarray, y: np.ndarray, groups: np.ndarray | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if groups is None:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        inner_train, inner_val = next(split.split(np.zeros(len(train)), y[train]))
    else:
        sub_y, sub_groups = y[train], groups[train]
        per_label = pd.DataFrame({"y": sub_y, "g": sub_groups}).groupby("y")["g"].nunique()
        n_splits = min(3, int(per_label.min()))
        if n_splits < 2:
            raise ValueError("Outer training data cannot support group-aware C selection.")
        split = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        candidates = list(split.split(np.zeros(len(train)), sub_y, sub_groups))
        valid = [item for item in candidates if _valid_folds([item], sub_y)]
        if not valid:
            raise ValueError("No class-complete group-aware inner split.")
        inner_train, inner_val = valid[0]
    return train[inner_train], train[inner_val]


def classifier(c_value: float, max_iter: int, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=max_iter,
        tol=1e-4,
        random_state=seed,
    )


def transform_dense(
    matrix: np.ndarray, fit_rows: np.ndarray, *row_sets: np.ndarray
) -> tuple[list[np.ndarray], StandardScaler]:
    scaler = StandardScaler().fit(matrix[fit_rows])
    return [scaler.transform(matrix[rows]) for rows in row_sets], scaler


@dataclass(frozen=True)
class FoldSpec:
    protocol: str
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    seed: int


def score_predictions(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict:
    predicted = classes[np.argmax(probabilities, axis=1)]
    return {
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_auroc": float(roc_auc_score(
            y_true, probabilities, labels=classes, multi_class="ovr", average="macro"
        )),
        "top3_accuracy": float(top_k_accuracy_score(
            y_true, probabilities, k=3, labels=classes
        )),
    }


def evaluate_fold(
    spec: FoldSpec, name: str, matrix: np.ndarray | None, sequences: np.ndarray,
    y: np.ndarray, classes: np.ndarray, groups: np.ndarray | None,
    c_grid: tuple[float, ...], max_iter: int,
) -> tuple[dict, pd.DataFrame]:
    tune_train, tune_val = inner_split(spec.train, y, groups, spec.seed + spec.fold * 1009)
    best: tuple[float, float] | None = None
    if name == "kmer_tfidf":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False)
        inner_train_x = vectorizer.fit_transform(sequences[tune_train])
        inner_val_x = vectorizer.transform(sequences[tune_val])
    else:
        (inner_train_x, inner_val_x), _ = transform_dense(
            matrix, tune_train, tune_train, tune_val
        )
    for c_value in c_grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = classifier(c_value, max_iter, spec.seed).fit(inner_train_x, y[tune_train])
        value = float(f1_score(y[tune_val], model.predict(inner_val_x), average="macro"))
        candidate = (value, -c_value)
        if best is None or candidate > best:
            best = candidate
    chosen_c = -best[1]

    if name == "kmer_tfidf":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False)
        train_x = vectorizer.fit_transform(sequences[spec.train])
        test_x = vectorizer.transform(sequences[spec.test])
        dimension = int(train_x.shape[1])
    else:
        (train_x, test_x), _ = transform_dense(matrix, spec.train, spec.train, spec.test)
        dimension = int(matrix.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = classifier(chosen_c, max_iter, spec.seed).fit(train_x, y[spec.train])
    probabilities = model.predict_proba(test_x)
    if not np.array_equal(model.classes_, classes):
        raise ValueError("A fold lacks one or more classes despite split validation.")
    metrics = score_predictions(y[spec.test], probabilities, classes)
    row = {
        "protocol": spec.protocol, "repeat": spec.repeat, "fold": spec.fold,
        "seed": spec.seed, "representation": name, "display_name": DISPLAY[name],
        "dim": dimension, "train_rows": len(spec.train), "test_rows": len(spec.test),
        "selected_c": chosen_c, **metrics,
    }
    predicted = classes[np.argmax(probabilities, axis=1)]
    top3 = classes[np.argpartition(probabilities, -3, axis=1)[:, -3:]]
    prediction = pd.DataFrame({
        "protocol": spec.protocol,
        "repeat": spec.repeat,
        "fold": spec.fold,
        "representation": name,
        "prediction_row_index": spec.test,
        "true_label": y[spec.test],
        "predicted_label": predicted,
        "correct": predicted == y[spec.test],
        "top3_labels": ["|".join(map(str, sorted(values))) for values in top3],
        "true_label_probability": probabilities[
            np.arange(len(spec.test)), np.searchsorted(classes, y[spec.test])
        ],
    })
    return row, prediction


def aggregate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["macro_f1", "balanced_accuracy", "accuracy", "macro_auroc", "top3_accuracy"]
    grouped = raw.groupby(["protocol", "representation", "display_name"], sort=False)
    rows = []
    for keys, frame in grouped:
        row = dict(zip(("protocol", "representation", "display_name"), keys))
        row["dim"] = int(round(float(frame["dim"].median())))
        row["dim_min"] = int(frame["dim"].min())
        row["dim_max"] = int(frame["dim"].max())
        row["observations"] = len(frame)
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_statistics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = ("rp", "tcr_bert", "sceptr_cdr3", "esm2_35m")
    differences, tests = [], []
    for protocol in raw["protocol"].unique():
        subset = raw[raw.protocol == protocol]
        wide = subset.pivot(index=["repeat", "fold", "seed"], columns="representation",
                            values="macro_f1")
        if "rtp" not in wide:
            continue
        for comparator in comparisons:
            if comparator not in wide:
                continue
            paired = wide[["rtp", comparator]].dropna().copy()
            paired["difference"] = paired["rtp"] - paired[comparator]
            for index, values in paired.iterrows():
                differences.append({
                    "protocol": protocol, "repeat": index[0], "fold": index[1],
                    "seed": index[2], "comparison": f"rtp_minus_{comparator}",
                    "rtp_macro_f1": values["rtp"], "comparator_macro_f1": values[comparator],
                    "difference": values["difference"],
                })
            delta = paired["difference"].to_numpy()
            try:
                statistic, p_value = wilcoxon(delta, alternative="two-sided", method="auto")
            except ValueError:
                statistic, p_value = 0.0, 1.0
            tests.append({
                "protocol": protocol, "comparison": f"rtp_vs_{comparator}",
                "observations": len(delta), "mean_paired_difference": float(delta.mean()),
                "median_paired_difference": float(np.median(delta)),
                "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()), "wilcoxon_statistic": float(statistic),
                "wilcoxon_two_sided_p": float(p_value),
            })
    return pd.DataFrame(differences), pd.DataFrame(tests)


def latex_table(aggregate: pd.DataFrame, protocol: str) -> str:
    data = aggregate[aggregate.protocol == protocol].set_index("representation")
    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Representation & Dim & Macro F1 & Balanced accuracy & Accuracy & Macro AUROC \\",
        r"\midrule",
    ]
    for name in PRIMARY_ORDER:
        if name not in data.index:
            continue
        row = data.loc[name]
        cell = lambda metric: f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_std']:.3f}"
        lines.append(
            f"{DISPLAY[name]} & {int(row['dim'])} & {cell('macro_f1')} & "
            f"{cell('balanced_accuracy')} & {cell('accuracy')} & {cell('macro_auroc')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def create_figure(raw: pd.DataFrame, aggregate: pd.DataFrame, output: Path) -> None:
    standard = aggregate[aggregate.protocol == "sequence_stratified"].set_index("representation")
    order = [name for name in PRIMARY_ORDER if name in standard.index]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), gridspec_kw={"width_ratios": [1.55, 1, 1]})
    ax = axes[0]
    positions = np.arange(len(order))
    means = standard.loc[order, "macro_f1_mean"].to_numpy()
    errors = standard.loc[order, "macro_f1_std"].to_numpy()
    colors = ["#d55e00" if name == "rtp" else "#4477aa" if name in NATIVE_NAMES else "#228833"
              for name in order]
    ax.errorbar(positions, means, yerr=errors, fmt="none", ecolor="#333333", capsize=2, lw=.9)
    ax.scatter(positions, means, c=colors, s=34, edgecolor="white", linewidth=.5, zorder=3)
    ax.set_xticks(positions, [DISPLAY[name] for name in order], rotation=50, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("A  Sequence-stratified performance", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=.6)

    def paired_panel(panel, protocol: str, title: str) -> None:
        subset = raw[(raw.protocol == protocol) & raw.representation.isin(["rp", "rtp"])]
        wide = subset.pivot(index=["repeat", "fold"], columns="representation", values="macro_f1").dropna()
        for _, row in wide.iterrows():
            panel.plot([0, 1], [row.rp, row.rtp], color="#999999", alpha=.65, lw=.7)
            panel.scatter([0, 1], [row.rp, row.rtp], c=["#4477aa", "#d55e00"], s=18, zorder=3)
        delta = (wide.rtp - wide.rp).mean()
        panel.set_xticks([0, 1], ["RP", "RTP"])
        panel.set_ylabel("Macro F1")
        panel.set_title(title, loc="left", fontweight="bold")
        panel.text(.5, .02, f"mean Δ={delta:+.3f}", transform=panel.transAxes,
                   ha="center", va="bottom", fontsize=8)
        panel.grid(axis="y", color="#dddddd", linewidth=.6)

    paired_panel(axes[1], "sequence_stratified", "B  TCRemP effect")
    if "redcea_similarity_grouped" in set(raw.protocol):
        paired_panel(axes[2], "redcea_similarity_grouped", "C  Cluster-grouped")
    else:
        axes[2].axis("off")
    fig.suptitle("Frozen TCR representations for multiclass epitope prediction", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "epitope_prediction.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "epitope_prediction.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if len(set(args.seeds)) != len(args.seeds) or args.folds < 2:
        raise ValueError("Seeds must be unique and at least two folds are required.")
    if any(value <= 0 for value in args.c_grid):
        raise ValueError("Every C value must be positive.")
    unknown = set(args.representations).difference(PRIMARY_ORDER)
    if unknown:
        raise ValueError(f"Unknown representations: {sorted(unknown)}")

    frame, original = selected_cohort(args.cohort, args.uniform_evaluation)
    audit, class_sizes = audit_dataset(frame)
    frame.to_csv(args.output_dir / "samples.tsv", sep="\t", index=False, lineterminator="\n")
    class_sizes.to_csv(args.output_dir / "class_sizes.csv", index=False, lineterminator="\n")
    exact_groups = pd.factorize(frame["cdr3"], sort=True)[0]
    similarity_groups = redcea_component_groups(frame)
    label_names = np.array(sorted(frame["label"].astype(str).unique()))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    y_id = np.array([label_to_id[label] for label in frame["label"].astype(str)], dtype=np.int32)
    protocols, split_report = make_splits(
        y_id, exact_groups, similarity_groups, args.seeds, args.folds,
        not args.skip_similarity_grouped,
    )
    write_split_assignments(args.output_dir / "split_assignments.csv", protocols, frame)

    specs = []
    for protocol, folds in protocols.items():
        if protocol == "exact_cdr3_grouped":
            continue  # exact groups are singletons; metrics are copied below with an explicit flag
        group_values = similarity_groups if protocol == "redcea_similarity_grouped" else None
        for repeat, fold, train, test in folds:
            specs.append((FoldSpec(protocol, repeat, fold, train, test, args.seeds[repeat]), group_values))

    all_metrics, all_predictions, rep_manifest = [], [], {}
    sequences = frame["cdr3"].astype(str).to_numpy()
    source_rows = int(pd.read_csv(args.cohort, sep="\t", usecols=["cdr3"]).shape[0])
    for name in args.representations:
        matrix = None if name == "kmer_tfidf" else load_dense_matrix(
            name, original, args.native_evaluation, args.external_representations, source_rows
        )
        rep_manifest[name] = {
            "display_name": DISPLAY[name],
            "dim": "fold-specific vocabulary" if matrix is None else int(matrix.shape[1]),
        }
        tasks = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(evaluate_fold)(
                spec, name, matrix, sequences, y_id, np.arange(len(label_names)), group_values,
                tuple(sorted(set(args.c_grid))), args.max_iter,
            )
            for spec, group_values in specs
        )
        for metrics, predictions in tasks:
            predictions["true_label"] = predictions["true_label"].map(dict(enumerate(label_names)))
            predictions["predicted_label"] = predictions["predicted_label"].map(dict(enumerate(label_names)))
            # top3 values are encoded class IDs; preserve IDs compactly and map is in label_classes.tsv.
            all_metrics.append(metrics)
            all_predictions.append(predictions)

    raw = pd.DataFrame(all_metrics)
    exact = raw[raw.protocol == "sequence_stratified"].copy()
    exact["protocol"] = "exact_cdr3_grouped"
    exact["evaluation_reused"] = True
    raw["evaluation_reused"] = False
    raw = pd.concat([raw, exact], ignore_index=True)
    raw.to_csv(args.output_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    pd.concat(all_predictions, ignore_index=True).to_parquet(
        args.output_dir / "predictions.parquet", index=False, compression="zstd"
    )
    pd.DataFrame({"class_id": np.arange(len(label_names)), "label": label_names}).to_csv(
        args.output_dir / "label_classes.tsv", sep="\t", index=False, lineterminator="\n"
    )
    aggregate = aggregate_metrics(raw)
    aggregate.to_csv(args.output_dir / "aggregated_metrics.csv", index=False, lineterminator="\n")
    differences, tests = paired_statistics(raw)
    differences.to_csv(args.output_dir / "paired_macro_f1_differences.csv", index=False,
                       lineterminator="\n")
    tests.to_csv(args.output_dir / "paired_tests.csv", index=False, lineterminator="\n")
    for protocol in aggregate.protocol.unique():
        (args.output_dir / f"table_{protocol}.tex").write_text(
            latex_table(aggregate, protocol), encoding="utf-8"
        )
    create_figure(raw, aggregate, args.output_dir)

    input_files = {
        "cohort": args.cohort,
        "uniform_pairs": args.uniform_evaluation / "uniform_pairs.npz",
    }
    manifest = {
        "status": "complete",
        "dataset": audit,
        "protocols": {
            "outer": {
                "requested_folds": args.folds, "seeds": args.seeds,
                "standard": "repeated shuffled StratifiedKFold",
                "exact_cdr3_grouped": split_report["exact_cdr3_grouped"],
                "similarity_grouped": split_report.get("redcea_similarity_grouped"),
            },
            "classifier": "L2 multinomial logistic regression; balanced class weights",
            "standardization": "training-fold statistics only",
            "hyperparameter_selection": {
                "c_grid": sorted(set(args.c_grid)),
                "criterion": "inner-training macro F1; smaller C wins ties",
                "inner_split": "stratified 15% validation; group-aware for grouped protocol",
            },
            "tfidf": "character 3-mer TF-IDF fit on training rows only",
            "exact_group_evaluation_reused": True,
        },
        "representations": rep_manifest,
        "inputs": {name: {"path": str(path), "sha256": sha256(path)}
                   for name, path in input_files.items()},
        "outputs": {name: sha256(args.output_dir / name) for name in (
            "samples.tsv", "class_sizes.csv", "split_assignments.csv", "fold_metrics.csv",
            "aggregated_metrics.csv", "paired_macro_f1_differences.csv", "paired_tests.csv",
            "predictions.parquet", "epitope_prediction.png", "epitope_prediction.pdf",
        )},
        "software": {"python": os.sys.version, "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
    }
    stable_json(args.output_dir / "RESULTS.json", manifest)
    print(json.dumps({"status": "complete", "output": str(args.output_dir), **audit}, sort_keys=True))


if __name__ == "__main__":
    main()
