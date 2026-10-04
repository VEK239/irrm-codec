"""Issue 3: aggregate Pgen runs into the per-run CSV and the summary tables.

Reads every ``metrics.json`` written by ``train_pgen_arm`` and produces:
  pgen_runs.csv       one row per run (arm x target x training size x seed)
  pgen_summary.csv    mean and std across seeds
  pgen_summary.md     Markdown tables for REPORT.md, one per target

The Markdown groups by training size so the low-data question is readable directly:
whether the pretrained IRRM arm's advantage over the from-scratch arm grows as the
training set shrinks.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

METRICS = ("rmse", "mae", "r2", "pearson_r", "spearman_rho", "bias")
SUBSET_ORDER = {"1k": 0, "10k": 1, "all": 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="artifacts/benchmark/pgen")
    p.add_argument("--output-dir", default="results/pgen")
    return p.parse_args()


def load_runs(runs_dir):
    rows = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        memory = run.get("memory") or {}
        rows.append(
            {
                "arm": run["arm"],
                "target": run["target"],
                "train_subset": run["train_subset"],
                "train_size": run["train_size"],
                "seed": run["seed"],
                "training_seconds": run["training_seconds"],
                **{name: run["test"][name] for name in METRICS},
                "inference_seq_per_second": run["inference"]["sequences_per_second"],
                "inference_ms_per_sequence": run["inference"]["ms_per_sequence"],
                "device": run["inference"]["device"],
                "peak_rss_mb": memory.get("peak_rss_mb"),
                "peak_gpu_mb": memory.get("peak_gpu_mb"),
                "epochs_ran": (run.get("arm_details") or {}).get("epochs_ran"),
            }
        )
    if not rows:
        raise SystemExit(f"No metrics.json found under {runs_dir}.")
    frame = pd.DataFrame(rows)
    frame["subset_order"] = frame.train_subset.map(SUBSET_ORDER).fillna(99)
    return frame.sort_values(["target", "subset_order", "arm", "seed"])


def summarize(runs):
    aggregated = (
        runs.groupby(["target", "train_subset", "subset_order", "arm"])
        .agg(
            seeds=("seed", "count"),
            train_size=("train_size", "first"),
            **{f"{name}_{stat}": (name, stat) for name in METRICS for stat in ("mean", "std")},
            training_seconds_mean=("training_seconds", "mean"),
            inference_seq_per_second_mean=("inference_seq_per_second", "mean"),
            peak_rss_mb_mean=("peak_rss_mb", "mean"),
            peak_gpu_mb_mean=("peak_gpu_mb", "mean"),
        )
        .reset_index()
    )
    return aggregated.sort_values(["target", "subset_order", "rmse_mean"])


def to_markdown(summary):
    blocks = []
    for target in summary.target.unique():
        blocks.append(f"### {target}\n")
        for subset in summary[summary.target == target].train_subset.unique():
            rows = summary[(summary.target == target) & (summary.train_subset == subset)]
            size = int(rows.train_size.iloc[0])
            blocks.append(f"**train = {subset} ({size:,} sequences)**\n")
            blocks.append(
                "| arm | RMSE | MAE | R² | Pearson r | Spearman ρ | bias | train s | seq/s | peak RSS MB |\n"
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for _, row in rows.iterrows():
                std = row["rmse_std"]
                rmse = f"{row['rmse_mean']:.4f}" + (f" ± {std:.4f}" if pd.notna(std) else "")
                cells = [
                    row["arm"],
                    rmse,
                    f"{row['mae_mean']:.4f}",
                    f"{row['r2_mean']:.4f}",
                    f"{row['pearson_r_mean']:.4f}",
                    f"{row['spearman_rho_mean']:.4f}",
                    f"{row['bias_mean']:+.4f}",
                    f"{row['training_seconds_mean']:.0f}",
                    f"{row['inference_seq_per_second_mean']:.0f}",
                    f"{row['peak_rss_mb_mean']:.0f}" if pd.notna(row["peak_rss_mb_mean"]) else "-",
                ]
                blocks.append("| " + " | ".join(cells) + " |")
            blocks.append("")
    return "\n".join(blocks) + "\n"


def pretraining_delta(runs):
    """Does IRRM pretraining help? RMSE difference with a significance test.

    Seed spread for these two arms is comparable to the effect being measured, so the
    difference of means alone cannot answer the question. Welch's t-test does not assume
    the two arms have equal variance, which they do not.
    """
    rows = []
    for (target, subset, order), group in runs.groupby(["target", "train_subset", "subset_order"]):
        scratch = group.loc[group.arm == "irrm_scratch", "rmse"].to_numpy()
        pretrained = group.loc[group.arm == "irrm_pretrained", "rmse"].to_numpy()
        if len(scratch) < 2 or len(pretrained) < 2:
            continue
        # Lower RMSE is better, so a positive delta means pretraining won.
        delta = scratch.mean() - pretrained.mean()
        _, p_value = stats.ttest_ind(scratch, pretrained, equal_var=False)
        rows.append(
            {
                "target": target,
                "train_subset": subset,
                "subset_order": order,
                "n_scratch": len(scratch),
                "n_pretrained": len(pretrained),
                "irrm_scratch": scratch.mean(),
                "irrm_pretrained": pretrained.mean(),
                "rmse_delta": delta,
                "relative_improvement": delta / scratch.mean(),
                "p_value": p_value,
                "verdict": (
                    "no detectable effect"
                    if p_value >= 0.05
                    else ("pretraining helps" if delta > 0 else "pretraining hurts")
                ),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["target", "subset_order"])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs_dir)
    runs.drop(columns=["subset_order"]).to_csv(output_dir / "pgen_runs.csv", index=False)

    summary = summarize(runs)
    summary.drop(columns=["subset_order"]).to_csv(output_dir / "pgen_summary.csv", index=False)
    (output_dir / "pgen_summary.md").write_text(to_markdown(summary), encoding="utf-8")

    print(f"runs: {len(runs)}  arms: {runs.arm.nunique()}  targets: {runs.target.nunique()}")
    delta = pretraining_delta(runs)
    if delta is not None:
        delta.drop(columns=["subset_order"]).to_csv(output_dir / "pretraining_delta.csv", index=False)
        print("\nDoes IRRM pretraining help? (positive rmse_delta means pretraining wins)")
        print(
            delta[
                ["target", "train_subset", "n_scratch", "n_pretrained", "irrm_scratch",
                 "irrm_pretrained", "rmse_delta", "p_value", "verdict"]
            ].to_string(index=False)
        )
    print(f"\nwrote {output_dir}/pgen_runs.csv, pgen_summary.csv, pgen_summary.md")


if __name__ == "__main__":
    main()
