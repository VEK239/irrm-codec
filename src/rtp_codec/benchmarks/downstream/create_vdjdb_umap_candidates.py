"""Create matched, exploratory RP/RTP UMAPs from saved VDJdb embeddings only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


TARGETS = ["SLLMWITQV", "GLCTLVAML", "VEALYLVCG", "LLLDRLNQL"]
SEED = 1729
N_TARGET = 1500
N_BACKGROUND = 1500


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--native-evaluation", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def sample_indices(labels: np.ndarray, target: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(labels == target)
    background = np.flatnonzero(labels != target)
    return (rng.choice(positive, size=min(N_TARGET, len(positive)), replace=False),
            rng.choice(background, size=min(N_BACKGROUND, len(background)), replace=False))


def project(values: np.ndarray) -> np.ndarray:
    # Projection is intentionally fit separately in each representation space.
    # This preserves native neighborhoods but does not make coordinate systems comparable.
    return umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=SEED,
                     n_components=2, transform_seed=SEED).fit_transform(values)


def main() -> None:
    a = args()
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    cohort = pd.read_csv(a.cohort, sep="\t")
    labels = cohort.label.astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    models = {"RP": np.load(a.native_evaluation / "embeddings_rp.npy", mmap_mode="r"),
              "RTP": np.load(a.native_evaluation / "embeddings_rtp.npy", mmap_mode="r")}
    if any(len(matrix) != len(cohort) for matrix in models.values()):
        raise ValueError("Embedding rows do not match cohort rows")
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(7.0, 10.0), constrained_layout=True)
    all_rows = []
    for row, target in enumerate(TARGETS):
        positive, background = sample_indices(labels, target, rng)
        indices = np.concatenate([background, positive])
        is_target = np.concatenate([np.zeros(len(background), dtype=bool), np.ones(len(positive), dtype=bool)])
        for column, (name, matrix) in enumerate(models.items()):
            coords = project(np.asarray(matrix[indices], dtype=np.float32))
            frame = pd.DataFrame({"index": indices, "epitope": labels[indices], "selected_epitope": target,
                                  "is_selected_epitope": is_target, "model": name,
                                  "umap_1": coords[:, 0], "umap_2": coords[:, 1]})
            all_rows.append(frame)
            ax = axes[row, column]
            ax.scatter(coords[~is_target, 0], coords[~is_target, 1], s=3.2, c="#bdbdbd", alpha=.33, linewidths=0)
            ax.scatter(coords[is_target, 0], coords[is_target, 1], s=6.5, c="#0072B2", alpha=.76, linewidths=0)
            ax.set_title(f"{target} (n={len(positive):,}) — {name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(.02, .03, "separate UMAP fit", transform=ax.transAxes, fontsize=6.5, color="#555555")
            for spine in ax.spines.values(): spine.set_linewidth(.5)
    fig.text(.5, .002, "Same sequence subset in each RP/RTP pair; UMAP: cosine, n_neighbors=30, min_dist=0.15, seed=1729. Coordinates are not directly comparable across separate fits.", ha="center", fontsize=7)
    fig.savefig(a.output_dir / "vdjdb_umap_contact_sheet.png", dpi=450, bbox_inches="tight")
    fig.savefig(a.output_dir / "vdjdb_umap_contact_sheet.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.concat(all_rows, ignore_index=True).to_csv(a.output_dir / "umap_coordinates.tsv", sep="\t", index=False)
    (a.output_dir / "README.md").write_text(json.dumps({
        "source_embeddings": {name: str(a.native_evaluation / f"embeddings_{name.lower()}.npy") for name in models},
        "cohort": str(a.cohort), "targets": TARGETS, "sampling": {"target_max": N_TARGET, "background": N_BACKGROUND, "seed": SEED},
        "umap": {"metric": "cosine", "n_neighbors": 30, "min_dist": .15, "random_state": SEED},
        "caveat": "RP and RTP use matched sequence subsets but separate UMAP fits; compare qualitative within-panel structure only."
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
