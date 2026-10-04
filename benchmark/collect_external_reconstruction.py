"""Issue 2: aggregate reconstruction runs into the per-run CSV and the summary table.

Reads every ``metrics.json`` written by ``train_decoder`` and produces:
  reconstruction_runs.csv     one row per run (representation x seed)
  reconstruction_summary.csv  mean and std across seeds, per representation
  reconstruction_summary.md   the same table in Markdown, for REPORT.md

Explained variance from the PCA fit is joined in, because it is the main explanation
for differences at a fixed bottleneck width: a representation that survives the
projection intact starts from a different place than one that loses most of its variance.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = (
    "exact_match",
    "token_accuracy",
    "normalized_levenshtein",
    "within_edit_distance_1",
    "mean_edit_distance",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="artifacts/benchmark/reconstruction")
    p.add_argument("--bottleneck-dir", default="data/benchmark/trb/bottleneck")
    p.add_argument("--output-dir", default="results/reconstruction")
    p.add_argument("--bottleneck-dim", type=int, default=64)
    return p.parse_args()


def load_runs(runs_dir, bottleneck_dim):
    rows = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["bottleneck_dim"] != bottleneck_dim:
            continue
        rows.append(
            {
                "representation": run["representation"],
                "seed": run["seed"],
                "bottleneck_dim": run["bottleneck_dim"],
                "train_subset": run["train_subset"],
                "train_size": run["train_size"],
                "epochs_ran": run["epochs_ran"],
                "best_epoch": run["best_epoch"],
                "best_val_loss": run["best_val_loss"],
                "training_seconds": run["training_seconds"],
                "test_loss": run["test"]["loss"],
                **{name: run["test"][name] for name in METRICS},
                "inference_seq_per_second": run["inference"]["sequences_per_second"],
                "inference_ms_per_sequence": run["inference"]["ms_per_sequence"],
                "device": run["inference"]["device"],
            }
        )
    if not rows:
        raise SystemExit(f"No metrics.json found under {runs_dir} for dim={bottleneck_dim}.")
    return pd.DataFrame(rows)


def summarize(runs, bottleneck_dir, bottleneck_dim):
    aggregated = runs.groupby("representation").agg(
        seeds=("seed", "count"),
        **{f"{name}_{stat}": (name, stat) for name in METRICS for stat in ("mean", "std")},
        training_seconds_mean=("training_seconds", "mean"),
        epochs_mean=("epochs_ran", "mean"),
        inference_seq_per_second_mean=("inference_seq_per_second", "mean"),
    )

    summary_path = Path(bottleneck_dir) / "bottleneck_summary.json"
    if summary_path.exists():
        bottleneck = json.loads(summary_path.read_text(encoding="utf-8"))
        aggregated["source_dim"] = [
            bottleneck.get(name, {}).get("source_dim") for name in aggregated.index
        ]
        aggregated["explained_variance"] = [
            bottleneck.get(name, {}).get("explained_variance", {}).get(str(bottleneck_dim))
            for name in aggregated.index
        ]

    return aggregated.sort_values("exact_match_mean", ascending=False).reset_index()


def to_markdown(summary, bottleneck_dim):
    header = (
        f"### Reconstruction at a {bottleneck_dim}-dimensional bottleneck\n\n"
        "| representation | source dim | expl. var | exact match | token acc | "
        "norm. Levenshtein | within ED 1 | train s | seq/s |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )

    def cell(row, name):
        std = row[f"{name}_std"]
        spread = f" ± {std:.4f}" if pd.notna(std) else ""
        return f"{row[f'{name}_mean']:.4f}{spread}"

    lines = []
    for _, row in summary.iterrows():
        expl = row.get("explained_variance")
        source_dim = row.get("source_dim")
        cells = [
            row["representation"],
            f"{source_dim:.0f}" if pd.notna(source_dim) else "-",
            f"{expl:.3f}" if pd.notna(expl) else "-",
            cell(row, "exact_match"),
            cell(row, "token_accuracy"),
            cell(row, "normalized_levenshtein"),
            cell(row, "within_edit_distance_1"),
            f"{row['training_seconds_mean']:.0f}",
            f"{row['inference_seq_per_second_mean']:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs_dir, args.bottleneck_dim)
    runs = runs.sort_values(["representation", "seed"])
    runs.to_csv(output_dir / "reconstruction_runs.csv", index=False)

    summary = summarize(runs, args.bottleneck_dir, args.bottleneck_dim)
    summary.to_csv(output_dir / "reconstruction_summary.csv", index=False)
    (output_dir / "reconstruction_summary.md").write_text(
        to_markdown(summary, args.bottleneck_dim), encoding="utf-8"
    )

    print(f"runs: {len(runs)}  representations: {runs.representation.nunique()}")
    print(
        summary[
            ["representation", "seeds", "exact_match_mean", "exact_match_std", "token_accuracy_mean"]
        ].to_string(index=False)
    )
    print(f"\nwrote {output_dir}/reconstruction_runs.csv")
    print(f"wrote {output_dir}/reconstruction_summary.csv")
    print(f"wrote {output_dir}/reconstruction_summary.md")


if __name__ == "__main__":
    main()
