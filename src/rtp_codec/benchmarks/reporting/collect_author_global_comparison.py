"""Assemble the preregistered author-native global embedding comparison table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = [
    ("rtp", "DATA-ANCHOR RTP", "sequence-only", 128),
    ("tcremp", "TCRemP prototype distances", "sequence-derived alignment", 9000),
    ("tcr_bert", "TCR-BERT final residue mean", "sequence-only", 768),
    ("esm2_35m", "ESM2-35M final residue mean", "sequence-only", 480),
    ("esm2_8m", "ESM2-8M final residue mean", "sequence-only sensitivity", 320),
    ("sceptr_cdr3", "SCEPTR CDR3-only native CLS", "sequence-only", 64),
    ("sceptr", "SCEPTR default native CLS", "annotation-aware V+CDR3+J", 64),
    ("tfidf", "CDR3 TF-IDF ridge", "sequence control", None),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_runs(root: Path, key: str, expected: set[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        data = load_json(path)
        name = data[key]
        if name not in expected:
            continue
        row = {"name": name, "seed": int(data["seed"])}
        row.update({f"test_{metric}": value for metric, value in data["test"].items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    for name in expected:
        seeds = sorted(frame.loc[frame["name"] == name, "seed"].tolist())
        if seeds != [42, 43, 44]:
            raise ValueError(f"{name} seeds are {seeds}, expected [42, 43, 44]")
    return frame


def format_cell(mean, std, digits=4):
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "method", "modality", "reconstruction_exact", "pgen_rmse",
        "vdjdb_p_at_1", "vdjdb_p_at_5", "vdjdb_p_at_10",
        "vdjdb_map", "vdjdb_within_between_ratio",
    ]
    labels = [
        "Method", "Modality", "Reconstruction exact", "pgen RMSE",
        "VDJdb P@1", "P@5", "P@10", "MAP", "Within/between",
    ]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    for _, row in frame.iterrows():
        values = [str(row[col]) if not pd.isna(row[col]) else "NA" for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-reconstruction-summary", type=Path, required=True)
    parser.add_argument("--old-pgen-summary", type=Path, required=True)
    parser.add_argument("--new-reconstruction-dir", type=Path, required=True)
    parser.add_argument("--new-pgen-dir", type=Path, required=True)
    parser.add_argument("--control-pgen-dir", type=Path, required=True)
    parser.add_argument("--reference-vdjdb-metrics", type=Path, required=True)
    parser.add_argument("--external-vdjdb-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    reconstruction = pd.read_csv(args.old_reconstruction_summary)
    new_recon = aggregate_runs(
        args.new_reconstruction_dir, "representation", {"esm2_35m", "sceptr_cdr3"}
    )
    recon_new_rows = []
    for name, block in new_recon.groupby("name"):
        recon_new_rows.append({
            "representation": name,
            "seeds": len(block),
            "exact_match_mean": block["test_exact_match"].mean(),
            "exact_match_std": block["test_exact_match"].std(ddof=1),
            "token_accuracy_mean": block["test_token_accuracy"].mean(),
            "token_accuracy_std": block["test_token_accuracy"].std(ddof=1),
        })
    reconstruction = pd.concat([reconstruction, pd.DataFrame(recon_new_rows)], ignore_index=True)

    pgen = pd.read_csv(args.old_pgen_summary)
    primary_pgen = aggregate_runs(
        args.new_pgen_dir,
        "arm",
        {"esm2_35m_mlp", "sceptr_cdr3_mlp", "tcremp_mlp"},
    )
    control_pgen = aggregate_runs(args.control_pgen_dir, "arm", {"onehot_mlp"})
    new_pgen = pd.concat([primary_pgen, control_pgen], ignore_index=True)
    pgen_new_rows = []
    for arm, block in new_pgen.groupby("name"):
        pgen_new_rows.append({
            "arm": arm,
            "seeds": len(block),
            "rmse_mean": block["test_rmse"].mean(),
            "rmse_std": block["test_rmse"].std(ddof=1),
            "r2_mean": block["test_r2"].mean(),
            "r2_std": block["test_r2"].std(ddof=1),
            "spearman_rho_mean": block["test_spearman_rho"].mean(),
            "spearman_rho_std": block["test_spearman_rho"].std(ddof=1),
        })
    pgen = pd.concat([pgen, pd.DataFrame(pgen_new_rows)], ignore_index=True)

    external = pd.read_csv(args.external_vdjdb_summary, sep="\t").set_index("model")
    reference = load_json(args.reference_vdjdb_metrics)["rtp"]["cosine"]
    rows = []
    for key, label, modality, native_dim in METHODS:
        recon = reconstruction[reconstruction["representation"] == key]
        pgen_arm = "tfidf_ridge" if key == "tfidf" else f"{key}_mlp"
        pgen_row = pgen[pgen["arm"] == pgen_arm]
        row = {
            "condition": key,
            "method": label,
            "modality": modality,
            "native_global_dim": native_dim,
            "reconstruction_common_dim": 64 if len(recon) else np.nan,
            "reconstruction_exact_mean": recon["exact_match_mean"].iloc[0] if len(recon) else np.nan,
            "reconstruction_exact_std": recon["exact_match_std"].iloc[0] if len(recon) else np.nan,
            "reconstruction_token_accuracy_mean": recon["token_accuracy_mean"].iloc[0] if len(recon) else np.nan,
            "pgen_rmse_mean": pgen_row["rmse_mean"].iloc[0] if len(pgen_row) else np.nan,
            "pgen_rmse_std": pgen_row["rmse_std"].iloc[0] if len(pgen_row) else np.nan,
            "pgen_r2_mean": pgen_row["r2_mean"].iloc[0] if len(pgen_row) else np.nan,
            "pgen_spearman_mean": pgen_row["spearman_rho_mean"].iloc[0] if len(pgen_row) else np.nan,
        }
        if key == "rtp":
            macro = reference["macro_epitope"]
            knn = reference["knn_classification"]
        elif key in external.index:
            macro = {
                metric: external.loc[key, f"cosine_{metric}"]
                for metric in ("precision_at_1", "precision_at_5", "precision_at_10",
                               "average_precision", "within_between_ratio")
            }
            knn = None
        else:
            macro = knn = None
        for source, target in (
            ("precision_at_1", "vdjdb_cosine_precision_at_1"),
            ("precision_at_5", "vdjdb_cosine_precision_at_5"),
            ("precision_at_10", "vdjdb_cosine_precision_at_10"),
            ("average_precision", "vdjdb_cosine_map"),
            ("within_between_ratio", "vdjdb_cosine_within_between_ratio"),
        ):
            row[target] = (
                float(macro[source])
                if macro is not None
                else np.nan
            )
        row["vdjdb_cosine_knn_macro_f1_at_5"] = (
            float(knn["k_5"]["macro_f1"]) if knn is not None
            else (
                float(external.loc[key, "cosine_knn_macro_f1_at_5"])
                if key in external.index else np.nan
            )
        )
        rows.append(row)

    final = pd.DataFrame(rows)
    final["reconstruction_exact"] = [
        format_cell(mean, std) for mean, std in zip(
            final["reconstruction_exact_mean"], final["reconstruction_exact_std"]
        )
    ]
    final["pgen_rmse"] = [
        format_cell(mean, std) for mean, std in zip(final["pgen_rmse_mean"], final["pgen_rmse_std"])
    ]
    for src, dst in (
        ("vdjdb_cosine_precision_at_1", "vdjdb_p_at_1"),
        ("vdjdb_cosine_precision_at_5", "vdjdb_p_at_5"),
        ("vdjdb_cosine_precision_at_10", "vdjdb_p_at_10"),
        ("vdjdb_cosine_map", "vdjdb_map"),
        ("vdjdb_cosine_within_between_ratio", "vdjdb_within_between_ratio"),
    ):
        final[dst] = final[src].map(lambda value: "NA" if pd.isna(value) else f"{value:.4f}")

    table_path = args.output_dir / "final_author_global_comparison.tsv"
    final.to_csv(table_path, sep="\t", index=False, lineterminator="\n")
    md_path = args.output_dir / "final_author_global_comparison.md"
    md_path.write_text(markdown_table(final), encoding="utf-8")
    manifest = {
        "status": "complete",
        "protocol": {
            "reconstruction": "train-only PCA64 plus identical learned decoder; three probe seeds",
            "pgen": "native author-global vector plus identical MLP; three probe seeds",
            "vdjdb": "native author-global vector on the fixed REDCEA gallery; macro over 87 epitopes",
        },
        "inputs": {
            str(path): sha256(path)
            for path in (
                args.old_reconstruction_summary,
                args.old_pgen_summary,
                args.reference_vdjdb_metrics,
                args.external_vdjdb_summary,
            )
        },
        "outputs": {table_path.name: sha256(table_path), md_path.name: sha256(md_path)},
        "limitations": [
            "SCEPTR-default is annotation-aware and is not equivalent to sequence-only methods.",
            "SCEPTR-default was not run on full REDCEA because unsupported V/J rows would require method-specific filtering.",
            "REDCEA membership is sequence/TCRemP-cluster-conditioned, so retrieval is motif-conditioned.",
            "Aligned one-hot is retained as an internal high-dimensional sequence control but omitted from the manuscript-facing comparison.",
            "Probe mean/std reflects probe seeds, not pretrained encoder training-seed uncertainty.",
        ],
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
