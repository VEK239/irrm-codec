"""Run matched char/WordPiece multitask experiments and collect test metrics."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/benchmark/trb")
    parser.add_argument("--output-root", default="artifacts/multitask_comparison")
    parser.add_argument("--wordpiece-tokenizers", nargs="+", required=True)
    parser.add_argument("--train-subset", choices=["1k", "10k", "all"], default="10k")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments after this flag are forwarded to train_multitask.",
    )
    return parser.parse_args()


def config_name(tokenizer_path: str | None) -> str:
    if tokenizer_path is None:
        return "char"
    path = Path(tokenizer_path)
    return path.parent.name or path.stem


def run_one(args, tokenizer_path: str | None, seed: int) -> tuple[str, dict]:
    name = config_name(tokenizer_path)
    run_dir = Path(args.output_root) / name / f"seed_{seed}"
    metrics_path = run_dir / "test_metrics.json"
    if args.force or not metrics_path.exists():
        command = [
            sys.executable,
            "-m",
            "irrm_codec.train_multitask",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(run_dir),
            "--train-subset",
            args.train_subset,
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
        if tokenizer_path is None:
            command.extend(["--tokenizer-type", "char"])
        else:
            command.extend(
                [
                    "--tokenizer-type",
                    "wordpiece",
                    "--tokenizer-path",
                    tokenizer_path,
                ]
            )
        command.extend(args.extra_args)
        subprocess.run(command, check=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return name, metrics


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    configs = [None, *args.wordpiece_tokenizers]
    rows = []
    for tokenizer_path in configs:
        for seed in args.seeds:
            name, metrics = run_one(args, tokenizer_path, seed)
            rows.append({"config": name, "seed": seed, **metrics})

    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"config", "seed"} and isinstance(value, (int, float))
        }
    )
    with (output_root / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "seed", *metric_names])
        writer.writeheader()
        writer.writerows(
            {key: row.get(key) for key in writer.fieldnames}
            for row in rows
        )

    summary = {}
    for name in sorted({row["config"] for row in rows}):
        matching = [row for row in rows if row["config"] == name]
        summary[name] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in matching if metric in row]
            if values:
                summary[name][metric] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
