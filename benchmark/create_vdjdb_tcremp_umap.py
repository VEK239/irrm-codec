"""Fit and render a native-TCRemP UMAP for the full VDJdb cohort."""
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
    "SSYRRPVGI": "#4477AA",
    "STPESANL": "#EE6677",
    "SLLMWITQV": "#228833",
    "KYNKANVFL": "#AA3377",
    "SIINFEKL": "#CCBB44",
}
SEED = 1729
MAX_BACKGROUND = 5_000
MAX_PER_HIGHLIGHT = 1_800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tcremp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cap(frame: pd.DataFrame, maximum: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.iloc[rng.choice(len(frame), size=maximum, replace=False)]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    cohort = pd.read_csv(args.cohort, sep="\t")
    labels = cohort.label.astype(str).to_numpy()
    matrix = np.load(args.tcremp, mmap_mode="r")
    if matrix.shape[0] != len(cohort):
        raise ValueError(f"TCRemP rows ({matrix.shape[0]}) do not match cohort ({len(cohort)})")
    if not np.isfinite(matrix).all():
        raise ValueError("TCRemP matrix contains non-finite values")

    coordinates = umap.UMAP(
        n_neighbors=30,
        min_dist=0.15,
        metric="cosine",
        n_components=2,
        random_state=SEED,
        transform_seed=SEED,
    ).fit_transform(np.asarray(matrix, dtype=np.float32))
    frame = pd.DataFrame({"index": np.arange(len(cohort)), "epitope": labels,
                          "umap_1": coordinates[:, 0], "umap_2": coordinates[:, 1]})
    frame.to_parquet(args.output_dir / "umap_coordinates.parquet", index=False)

    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(3.38, 3.38))
    other = cap(frame[~frame.epitope.isin(HIGHLIGHTS)], MAX_BACKGROUND, rng)
    ax.scatter(other.umap_1, other.umap_2, s=1.1, c="#9e9e9e", alpha=.20,
               linewidths=0, rasterized=True)
    for epitope, color in HIGHLIGHTS.items():
        selected = cap(frame[frame.epitope == epitope], MAX_PER_HIGHLIGHT, rng)
        ax.scatter(selected.umap_1, selected.umap_2, s=8.0, c=color, alpha=.76,
                   linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("TCRemP prototype embedding", fontsize=9, fontweight="bold", pad=5)
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="#9e9e9e",
                      markeredgewidth=0, markersize=3.9, alpha=.7, label="Other epitopes")]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor=color,
                       markeredgewidth=0, markersize=4.5, label=epitope)
                for epitope, color in HIGHLIGHTS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.8,
               bbox_to_anchor=(.5, -.045), handletextpad=.25, columnspacing=.7, borderaxespad=0)
    fig.subplots_adjust(left=.012, right=.995, top=.89, bottom=.105)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig((args.output_dir / "vdjdb_whole_cohort_tcremp_umap").with_suffix(suffix),
                    bbox_inches="tight", pad_inches=.01, **kwargs)
    plt.close(fig)

    metadata = {
        "cohort_size": int(len(cohort)),
        "source_embeddings": {"TCRemP": str(args.tcremp)},
        "matrix_shape": [int(dim) for dim in matrix.shape],
        "highlights": {epitope: {"n_full_cohort": int((labels == epitope).sum()), "color": color}
                       for epitope, color in HIGHLIGHTS.items()},
        "umap": {"metric": "cosine", "n_neighbors": 30, "min_dist": 0.15,
                 "n_components": 2, "random_state": SEED, "transform_seed": SEED},
        "display_sampling": {"seed": SEED, "max_background": MAX_BACKGROUND,
                             "max_per_highlight": MAX_PER_HIGHLIGHT},
        "caveat": "The native 9,000-dimensional TCRemP representation is projected separately; compare only qualitative within-panel structure.",
    }
    (args.output_dir / "README.md").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
