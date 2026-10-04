"""Evaluate author-native frozen global embeddings on the fixed REDCEA gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.evaluate_vdjdb_redcea_clonotypes import (
    METRIC_COLUMNS,
    candidate_distances,
    summarize_candidates,
)

EXTERNAL = ("esm2_35m", "tcr_bert", "sceptr_cdr3")
MODALITY = {
    "esm2_35m": "sequence-only",
    "tcr_bert": "sequence-only",
    "sceptr_cdr3": "sequence-only",
    "sceptr": "annotation-aware V+CDR3+J",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--representations-dir", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    gate = json.loads(args.preflight.read_text(encoding="utf-8"))
    if gate["status"] != "accepted":
        raise ValueError("Author-global preflight did not accept.")

    cohort = pd.read_csv(args.cohort, sep="\t")
    if sha256(args.cohort) != gate["input_hashes"]["redcea_cohort"]:
        raise ValueError("Cohort hash differs from accepted preflight.")
    gallery_file = args.reference_evaluation / "candidate_gallery.npz"
    gallery = np.load(gallery_file)
    candidates = gallery["candidates"]
    relevant = gallery["relevant"]
    if candidates.shape[0] != len(cohort):
        raise ValueError("Candidate gallery row count mismatch.")

    labels_text = cohort["label"].astype(str)
    label_names = sorted(labels_text.unique())
    label_to_id = {label: index for index, label in enumerate(label_names)}
    label_ids = np.array([label_to_id[value] for value in labels_text], dtype=np.int32)
    sequence_sha = hashlib.sha256(
        "\n".join(cohort["cdr3"].astype(str)).encode("utf-8")
    ).hexdigest()
    catalog = json.loads(
        (args.representations_dir / "representations.json").read_text(encoding="utf-8")
    )

    summaries = {}
    per_epitope = []
    rows = []
    for name in EXTERNAL:
        entry = catalog[name]
        if entry["rows"] != len(cohort) or entry["sequence_sha256"] != sequence_sha:
            raise ValueError(f"{name} catalog provenance mismatch")
        source = Path(entry["path"])
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"{name} representation hash mismatch")
        matrix = np.load(source, mmap_mode="r")
        if matrix.shape != (len(cohort), entry["dim"]) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} shape/finiteness mismatch: {matrix.shape}")
        summaries[name] = {}
        row = {"model": name, "modality": MODALITY[name], "native_dim": int(matrix.shape[1])}
        for distance in ("cosine", "euclidean"):
            distances, norms = candidate_distances(matrix, candidates, distance)
            summary, _, per_label = summarize_candidates(
                distances, candidates, relevant, label_ids, label_names
            )
            summary["embedding_norms"] = norms
            summaries[name][distance] = summary
            per_label.insert(0, "model", name)
            per_label.insert(1, "modality", MODALITY[name])
            per_label.insert(2, "distance", distance)
            per_epitope.append(per_label)
            for metric in METRIC_COLUMNS:
                row[f"{distance}_{metric}"] = summary["macro_epitope"][metric]
            for k in (1, 5, 10):
                row[f"{distance}_knn_macro_f1_at_{k}"] = (
                    summary["knn_classification"][f"k_{k}"]["macro_f1"]
                )
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        args.output_dir / "external_retrieval_summary.tsv",
        sep="\t", index=False, lineterminator="\n",
    )
    pd.concat(per_epitope, ignore_index=True).to_csv(
        args.output_dir / "external_per_epitope.tsv",
        sep="\t", index=False, lineterminator="\n",
    )
    (args.output_dir / "external_metrics.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "status": "complete",
        "cohort": {
            "sha256": sha256(args.cohort),
            "clonotypes": len(cohort),
            "epitopes": len(label_names),
        },
        "candidate_gallery": {
            "path": str(gallery_file),
            "sha256": sha256(gallery_file),
            "shape": list(candidates.shape),
        },
        "representations": {
            name: {
                "sha256": catalog[name]["sha256"],
                "dim": catalog[name]["dim"],
                "modality": MODALITY[name],
                "model": catalog[name]["model"],
            }
            for name in EXTERNAL
        },
        "selection": "No VDJdb labels were used to fit, pool, tune, or select an encoder.",
        "unavailable": {
            "sceptr": (
                "Annotation-aware SCEPTR-default rejects unsupported/nonfunctional "
                "VDJdb V/J alleles; the fixed cohort was not filtered per method."
            )
        },
        "uncertainty": "Per-epitope distributions retained; no bootstrap requested.",
        "circularity_caveat": (
            "REDCEA membership is sequence/TCRemP-cluster-conditioned; this is "
            "motif-conditioned retrieval geometry, not independent binding evidence."
        ),
    }
    (args.output_dir / "RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
