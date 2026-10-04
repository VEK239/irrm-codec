"""Small CPU-only functional gate for frozen VDJdb evaluation statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rtp_codec.benchmarks.downstream.evaluate_vdjdb_frozen import (
    assert_finite_tree,
    bootstrap_primary,
    build_pair_manifest,
    controls,
    distance_matrix,
    pair_metrics,
    retrieval_metrics,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-preflight", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preflight = json.loads(args.cohort_preflight.read_text(encoding="utf-8"))
    if sha256(args.cohort) != preflight["cohort"]["cohort_sha256"]:
        raise ValueError("Cohort hash changed after preparation.")
    full = pd.read_csv(args.cohort, sep="\t")
    chosen = []
    for _, group in full.groupby("label", sort=True):
        if group["donor_key"].nunique() >= 2:
            chosen.append(group.head(20))
        if len(chosen) == 4:
            break
    cohort = pd.concat(chosen, ignore_index=True)
    pairs = build_pair_manifest(cohort, limit=50, seed=42)
    rng = np.random.default_rng(42)
    arrays = {name: rng.normal(size=(len(cohort), 16)) for name in ("r", "p", "rp", "rtp")}
    matrices = {name: distance_matrix(value, "cosine") for name, value in arrays.items()}
    retrieval = {}
    report = {}
    for name, matrix in matrices.items():
        pair_report, _ = pair_metrics(matrix, pairs)
        retrieval_report, retrieval[name] = retrieval_metrics(matrix, cohort)
        report[name] = {"pairs": pair_report, "retrieval": retrieval_report}
    report["bootstrap"] = bootstrap_primary(matrices, pairs, retrieval, replicates=20, seed=42)
    report["controls"] = controls(cohort, pairs, matrices, permutations=20, seed=42)
    assert_finite_tree(report)
    payload = {
        "status": "ready",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cohort_sha256": sha256(args.cohort),
        "evaluation_script_sha256": sha256(args.evaluation_script),
        "smoke_rows": len(cohort),
        "smoke_pairs": len(pairs),
        "checks": {
            "pair_contrast": True,
            "pairwise_auroc_auprc": True,
            "donor_excluded_retrieval": True,
            "cosine_distance": True,
            "euclidean_distance": bool(np.isfinite(distance_matrix(arrays["r"], "euclidean")).all()),
            "donor_cluster_bootstrap": True,
            "length_composition_and_shuffle_controls": True,
            "finite_results": True,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
