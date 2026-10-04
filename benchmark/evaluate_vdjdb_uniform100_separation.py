"""AUROC and standardized distance separation on the fixed uniform-100 pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--native-evaluation", type=Path, required=True)
    parser.add_argument("--external-representations", type=Path, required=True)
    parser.add_argument("--uniform-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def separation_metrics(
    matrix: np.ndarray, within: np.ndarray, between: np.ndarray, labels: np.ndarray,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(values).all():
        raise ValueError("Embedding matrix contains a zero-norm or non-finite vector.")
    unit = values / norms
    n_within, n_between = within.shape[1], between.shape[1]
    auroc = np.empty(len(unit), dtype=np.float32)
    cohen_d = np.empty(len(unit), dtype=np.float32)
    for start in range(0, len(unit), batch_size):
        stop = min(start + batch_size, len(unit))
        query = unit[start:stop]
        d_within = 1.0 - np.sum(query[:, None] * unit[within[start:stop]], axis=2)
        d_between = 1.0 - np.sum(query[:, None] * unit[between[start:stop]], axis=2)
        ordered = np.argsort(np.concatenate([d_within, d_between], axis=1), axis=1)
        ranks = np.empty_like(ordered)
        ranks[np.arange(stop-start)[:, None], ordered] = np.arange(1, n_within+n_between+1)
        positive_rank_sum = ranks[:, :n_within].sum(axis=1)
        auroc[start:stop] = (
            n_within * n_between - (positive_rank_sum - n_within * (n_within + 1) / 2)
        ) / (n_within * n_between)
        pooled_sd = np.sqrt((d_within.var(axis=1, ddof=1) + d_between.var(axis=1, ddof=1)) / 2)
        cohen_d[start:stop] = (d_between.mean(axis=1) - d_within.mean(axis=1)) / pooled_sd
    per_query = pd.DataFrame({"epitope": labels, "auroc": auroc, "cohen_d": cohen_d})
    per_epitope = per_query.groupby("epitope", sort=True).agg(
        queries=("epitope", "size"), auroc=("auroc", "mean"), cohen_d=("cohen_d", "mean"),
    ).reset_index()
    return per_query, per_epitope


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    cohort = pd.read_csv(args.cohort, sep="\t")
    pairs = np.load(args.uniform_evaluation / "uniform_pairs.npz")
    original = pairs["original_indices"]
    within, between = pairs["within"], pairs["between"]
    labels = cohort.iloc[original]["label"].astype(str).to_numpy()
    catalog = json.loads((args.external_representations / "representations.json").read_text(encoding="utf-8"))
    paths = {
        "RTP": args.native_evaluation / "embeddings_rtp.npy",
        "RP": args.native_evaluation / "embeddings_rp.npy",
        "T": args.native_evaluation / "embeddings_t.npy",
        "ESM2-35M": Path(catalog["esm2_35m"]["path"]),
        "TCR-BERT": Path(catalog["tcr_bert"]["path"]),
        "SCEPTR (CDR3-only)": Path(catalog["sceptr_cdr3"]["path"]),
    }
    summary, per_epitope = [], []
    for name, path in paths.items():
        query, epitope = separation_metrics(np.load(path, mmap_mode="r")[original], within, between, labels)
        query.insert(0, "model", name)
        query.to_parquet(args.output_dir / f"per_query_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.parquet", index=False)
        epitope.insert(0, "model", name)
        per_epitope.append(epitope)
        summary.append({"model": name, "macro_auroc": float(epitope["auroc"].mean()), "macro_cohen_d": float(epitope["cohen_d"].mean()), "queries":len(query), "epitopes":len(epitope)})
    pd.DataFrame(summary).to_csv(args.output_dir / "summary.tsv", sep="\t", index=False, lineterminator="\n")
    pd.concat(per_epitope, ignore_index=True).to_csv(args.output_dir / "per_epitope.tsv", sep="\t", index=False, lineterminator="\n")
    (args.output_dir / "RESULTS.json").write_text(json.dumps({
        "status":"complete", "pairs":str(args.uniform_evaluation / "uniform_pairs.npz"),
        "metrics":{"auroc":"P(within distance < between distance)", "cohen_d":"(mean between - mean within) / pooled within/between distance SD"},
        "queries":len(labels), "epitopes":int(pd.Series(labels).nunique()), "models":list(paths),
    }, indent=2, sort_keys=True)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
