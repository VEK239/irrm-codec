"""Collect unpooled reconstruction and pgen runs into auditable tables."""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--pgen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def collect(root, task):
    rows = []
    for path in sorted(Path(root).glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = {
            "representation": payload["representation"],
            "seed": payload["seed"],
            "interface": payload["interface"],
            "parameters": payload["parameters"],
            "best_epoch": payload["best_epoch"],
            "best_val_loss": payload["best_val_loss"],
            "epochs_ran": payload["epochs_ran"],
            "training_seconds": payload["training_seconds"],
            "metrics_path": str(path),
        }
        if task == "reconstruction":
            names = ("loss", "token_accuracy", "exact_match", "normalized_levenshtein", "within_edit_distance_1", "mean_edit_distance")
        else:
            names = ("rmse", "mae", "r2", "pearson_r", "spearman_rho", "bias")
        rows.append({**base, **{name: payload["test"][name] for name in names}})
    if not rows:
        raise ValueError(f"No {task} metrics found under {root}")
    return pd.DataFrame(rows)


def summarize(frame, task):
    if task == "reconstruction":
        names = ("token_accuracy", "exact_match", "normalized_levenshtein", "within_edit_distance_1", "mean_edit_distance")
        ascending = False
        key = "exact_match_mean"
    else:
        names = ("rmse", "mae", "r2", "pearson_r", "spearman_rho", "bias")
        ascending = True
        key = "rmse_mean"
    result = frame.groupby("representation").agg(
        seeds=("seed", "count"),
        parameters=("parameters", "first"),
        training_seconds_mean=("training_seconds", "mean"),
        **{f"{name}_{stat}": (name, stat) for name in names for stat in ("mean", "std")},
    ).reset_index()
    return result.sort_values(key, ascending=ascending)


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reconstruction = collect(args.reconstruction_dir, "reconstruction")
    pgen = collect(args.pgen_dir, "pgen")
    reconstruction.to_csv(output / "reconstruction_residue_runs.tsv", sep="\t", index=False)
    pgen.to_csv(output / "pgen_residue_runs.tsv", sep="\t", index=False)
    summarize(reconstruction, "reconstruction").to_csv(
        output / "reconstruction_residue_summary.tsv", sep="\t", index=False
    )
    summarize(pgen, "pgen").to_csv(output / "pgen_residue_summary.tsv", sep="\t", index=False)
    print(f"status=complete reconstruction_runs={len(reconstruction)} pgen_runs={len(pgen)}")


if __name__ == "__main__":
    main()
