"""Finalize an already-scored VDJdb epitope-prediction output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtp_codec.benchmarks.downstream.evaluate_vdjdb_epitope_prediction import (
    aggregate_metrics,
    create_figure,
    latex_table,
    paired_statistics,
    sha256,
    stable_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    raw_path = output / "fold_metrics.csv"
    predictions = output / "predictions.parquet"
    samples_path = output / "samples.tsv"
    splits_path = output / "split_assignments.csv"
    for path in (raw_path, predictions, samples_path, splits_path):
        if not path.exists():
            raise FileNotFoundError(path)

    raw = pd.read_csv(raw_path)
    expected_representations = {
        "r", "t", "p", "rt", "rp", "tp", "rtp", "tcr_bert", "sceptr_cdr3",
        "esm2_35m", "kmer_tfidf",
    }
    if set(raw["representation"]) != expected_representations:
        raise ValueError("Fold metrics do not contain all eleven planned representations.")
    observed = raw.groupby(["protocol", "representation"]).size()
    expected_observations = {
        "sequence_stratified": 15,
        "exact_cdr3_grouped": 15,
        "redcea_similarity_grouped": 12,
    }
    for (protocol, _), count in observed.items():
        if int(count) != expected_observations[protocol]:
            raise ValueError(f"Incomplete {protocol} representation: {count}")

    aggregate = aggregate_metrics(raw)
    aggregate.to_csv(output / "aggregated_metrics.csv", index=False, lineterminator="\n")
    differences, tests = paired_statistics(raw)
    differences.to_csv(output / "paired_macro_f1_differences.csv", index=False,
                       lineterminator="\n")
    tests.to_csv(output / "paired_tests.csv", index=False, lineterminator="\n")
    for protocol in aggregate["protocol"].unique():
        (output / f"table_{protocol}.tex").write_text(
            latex_table(aggregate, protocol), encoding="utf-8"
        )
    create_figure(raw, aggregate, output)

    samples = pd.read_csv(samples_path, sep="\t")
    sizes = samples.groupby("label").size()
    label_counts = samples.groupby("cdr3")["label"].nunique()
    audit = {
        "sequences": int(len(samples)),
        "epitopes": int(samples["label"].nunique()),
        "class_size_min": int(sizes.min()),
        "class_size_median": float(np.median(sizes)),
        "class_size_max": int(sizes.max()),
        "duplicate_cdr3_rows": int(samples["cdr3"].duplicated(keep=False).sum()),
        "duplicate_cdr3_sequences_exist": bool(samples["cdr3"].duplicated().any()),
        "identical_cdr3_with_different_labels": int((label_counts > 1).sum()),
    }
    outputs = (
        "samples.tsv", "class_sizes.csv", "split_assignments.csv", "fold_metrics.csv",
        "aggregated_metrics.csv", "paired_macro_f1_differences.csv", "paired_tests.csv",
        "predictions.parquet", "epitope_prediction.png", "epitope_prediction.pdf",
        "table_sequence_stratified.tex", "table_exact_cdr3_grouped.tex",
        "table_redcea_similarity_grouped.tex",
    )
    representation_summary = (
        raw.groupby(["representation", "display_name"], sort=False)
        .agg(dim=("dim", "median"), dim_min=("dim", "min"), dim_max=("dim", "max"))
        .reset_index()
    )
    for column in ("dim", "dim_min", "dim_max"):
        representation_summary[column] = representation_summary[column].round().astype(int)

    result = {
        "status": "complete",
        "recovered_from_completed_scoring": True,
        "dataset": audit,
        "protocols": {
            "sequence_stratified": "5 folds x 3 seeds",
            "exact_cdr3_grouped": "alias of sequence-stratified because every CDR3 is unique",
            "redcea_similarity_grouped": "4 folds x 3 seeds; connected source_cid components",
            "inner_tuning": "training-only C in {0.1, 1.0}; macro F1 selection",
            "classifier": "balanced L2 multinomial logistic regression",
        },
        "representations": representation_summary.to_dict(orient="records"),

        "convergence": {
            "fold_fits": int(len(raw)),
            "converged_before_max_iter": int(raw["converged_before_max_iter"].sum())
            if "converged_before_max_iter" in raw else None,
        },
        "outputs": {name: sha256(output / name) for name in outputs},
    }
    stable_json(output / "RESULTS.json", result)
    print(json.dumps({"status": "complete", "output": str(output), **audit}, sort_keys=True))


if __name__ == "__main__":
    main()
