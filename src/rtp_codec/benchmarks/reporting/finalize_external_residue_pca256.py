"""Fail-closed catalog construction for train-only residue PCA vectors."""

import argparse
import json
from pathlib import Path

import numpy as np


NAMES = ("esm2_8m_residue", "tcr_bert_residue", "sceptr_residue")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows", type=int, default=99430)
    parser.add_argument("--components", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    catalog, summary = {}, {}
    for name in NAMES:
        report_path = output / f"{name}_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        array_path = output / f"{name}_{args.components}.npy"
        matrix = np.load(array_path, mmap_mode="r")
        if report.get("status") != "accepted" or report.get("validation_or_test_used_for_fit"):
            raise ValueError(f"Rejected or leaky projection report for {name}")
        if tuple(matrix.shape) != (args.rows, args.components):
            raise ValueError(f"Wrong projected shape for {name}: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite projection for {name}")
        catalog[name] = {
            "path": str(array_path),
            "shape": list(matrix.shape),
            "sha256": report["output_sha256"],
            "interface": f"flattened_residue_states_train_only_pca{args.components}",
        }
        summary[name] = {
            "source_dim": report["flattened_dim"],
            "components_fitted": args.components,
            "explained_variance": {str(args.components): report["explained_variance_ratio"]},
            "solver": report["solver"],
            "fit_split": "train",
            "fit_rows_sha256": report["fit_rows_sha256"],
            "capped_by_source_dim": False,
        }
    (output / "representations.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    (output / "bottleneck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"status=accepted representations={len(catalog)} rows={args.rows} dim={args.components}")


if __name__ == "__main__":
    main()
