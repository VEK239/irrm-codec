"""Fit one UMAP per saved RP/RTP embedding set across the complete VDJdb cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import umap


HIGHLIGHTS = {
    "SLLMWITQV": ("SLLMWITQV", "#0072B2"),
    "GLCTLVAML": ("GLCTLVAML", "#D55E00"),
    "VEALYLVCG": ("VEALYLVCG", "#009E73"),
    "LLLDRLNQL": ("LLLDRLNQL", "#CC79A7"),
}
SEED = 1729


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--native-evaluation", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def fit(matrix: np.ndarray) -> np.ndarray:
    return umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", n_components=2,
                     random_state=SEED, transform_seed=SEED).fit_transform(matrix)


def main() -> None:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    cohort = pd.read_csv(a.cohort, sep="\t")
    labels = cohort.label.astype(str).to_numpy()
    matrices = {"RP": np.load(a.native_evaluation / "embeddings_rp.npy", mmap_mode="r"),
                "RTP": np.load(a.native_evaluation / "embeddings_rtp.npy", mmap_mode="r")}
    if any(len(x) != len(cohort) for x in matrices.values()):
        raise ValueError("Embedding rows do not match the full cohort")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    frames = []
    highlighted = np.isin(labels, list(HIGHLIGHTS))
    for ax, (name, matrix) in zip(axes, matrices.items()):
        coords = fit(np.asarray(matrix, dtype=np.float32))
        ax.scatter(coords[~highlighted, 0], coords[~highlighted, 1], s=.45, c="#a9a9a9", alpha=.16, linewidths=0, rasterized=True)
        for epitope, (display, color) in HIGHLIGHTS.items():
            mask = labels == epitope
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3.8, c=color, alpha=.85, linewidths=0, rasterized=True, label=f"{display} (n={mask.sum():,})")
        ax.set_title(name, fontsize=12, weight="bold")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.tick_params(length=2, labelsize=7)
        frames.append(pd.DataFrame({"index": np.arange(len(cohort)), "epitope": labels, "model": name,
                                    "umap_1": coords[:, 0], "umap_2": coords[:, 1],
                                    "highlighted": highlighted}))
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="#a9a9a9", markeredgewidth=0, markersize=4, alpha=.55, label="Other VDJdb epitopes")]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor=color, markeredgewidth=0, markersize=5, label=f"{display} (n={(labels == ep).sum():,})") for ep, (display, color) in HIGHLIGHTS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.5, -.085))
    fig.text(.5, .005, "Full 65,756-sequence VDJdb cohort. UMAP: cosine, n_neighbors=30, min_dist=0.15, seed=1729. RP and RTP are separate fits; coordinates are not directly comparable.", ha="center", fontsize=7.5)
    fig.savefig(a.output_dir / "vdjdb_whole_cohort_rp_rtp_umap.png", dpi=450, bbox_inches="tight")
    fig.savefig(a.output_dir / "vdjdb_whole_cohort_rp_rtp_umap.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.concat(frames, ignore_index=True).to_parquet(a.output_dir / "umap_coordinates.parquet", index=False)
    (a.output_dir / "README.md").write_text(json.dumps({
        "cohort_size": int(len(cohort)), "source_embeddings": {name: str(a.native_evaluation / f"embeddings_{name.lower()}.npy") for name in matrices},
        "highlights": {ep: {"n": int((labels == ep).sum()), "color": color} for ep, (_, color) in HIGHLIGHTS.items()},
        "umap": {"metric": "cosine", "n_neighbors": 30, "min_dist": .15, "random_state": SEED},
        "caveat": "Both panels use the full identical cohort; RP/RTP are separately fitted UMAPs, so compare qualitative within-panel structure only."
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
