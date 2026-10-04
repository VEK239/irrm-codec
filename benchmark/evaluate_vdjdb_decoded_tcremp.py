"""Decode saved VDJdb RTP latents through the frozen TCRemP head.

This intentionally reuses the frozen uniform-100 VDJdb cohort, candidate pairs,
and ``separation_metrics`` implementation.  It does not encode sequences or
train a model: its sole intervention is ``saved RTP -> frozen T head``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

from evaluate_vdjdb_uniform100_separation import separation_metrics
from irrm_codec.multitask_data import TargetStandardizer
from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer


SEED = 1729
HIGHLIGHTS = {
    "SSYRRPVGI": "#4477AA", "STPESANL": "#EE6677", "SLLMWITQV": "#228833",
    "KYNKANVFL": "#AA3377", "SIINFEKL": "#CCBB44",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--rtp", type=Path, required=True,
                   help="Existing full-cohort RTP matrix used by the VDJdb analysis.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--standardizer", type=Path, required=True)
    p.add_argument("--uniform-evaluation", type=Path, required=True,
                   help="Directory containing the locked uniform_pairs.npz.")
    p.add_argument("--existing-separation", type=Path, required=True,
                   help="Completed RP/RTP separation directory; values are reused, not recomputed.")
    p.add_argument("--native-tcremp", type=Path,
                   help="Optional existing aligned native TCRemP matrix for reconstruction diagnostics.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(a: argparse.Namespace, rtp: np.ndarray, out: Path) -> tuple[Path, dict]:
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    config = IRRMCodecConfig(**payload["model_config"])
    if rtp.ndim != 2 or rtp.shape[1] != config.latent_dim:
        raise ValueError(f"RTP shape {rtp.shape} is incompatible with latent_dim={config.latent_dim}")
    standardizer = TargetStandardizer.load(a.standardizer)
    if len(standardizer.tcremp_mean) != config.tcremp_dim:
        raise ValueError("TCRemP standardizer and checkpoint output dimensions differ")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    model = IRRMCodecTransformer(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    mean = torch.from_numpy(standardizer.tcremp_mean).to(device)
    std = torch.from_numpy(standardizer.tcremp_std).to(device)
    decoded_path = out / "decoded_tcremp.npy"
    decoded = np.lib.format.open_memmap(decoded_path, mode="w+", dtype=np.float32,
                                        shape=(len(rtp), config.tcremp_dim))
    with torch.inference_mode():
        for start in range(0, len(rtp), a.batch_size):
            stop = min(start + a.batch_size, len(rtp))
            z = torch.from_numpy(np.asarray(rtp[start:stop], dtype=np.float32)).to(device)
            decoded[start:stop] = (model.tcremp_head(z) * std + mean).cpu().numpy()
    decoded.flush()
    return decoded_path, {
        "checkpoint_epoch_zero_based": int(payload["epoch"]),
        "checkpoint_best_validation_loss": float(payload["best_val_loss"]),
        "model_config": asdict(config), "standardizer_train_rows": standardizer.train_rows,
    }


def diagnostics(decoded: np.ndarray, native: np.ndarray | None) -> dict:
    result = {"decoded": {"shape": [int(x) for x in decoded.shape],
              "nan_count": int(np.isnan(decoded).sum()), "inf_count": int(np.isinf(decoded).sum()),
              "min": float(np.min(decoded)), "max": float(np.max(decoded)),
              "mean": float(np.mean(decoded)), "std": float(np.std(decoded))}}
    if native is not None:
        if native.shape != decoded.shape:
            raise ValueError(f"Native TCRemP shape {native.shape} does not match decoded {decoded.shape}")
        # Chunked global sums avoid a second multi-GB temporary array.
        cosine, squared, dot, sx, sy, sxx, syy, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for start in range(0, len(decoded), 128):
            x, y = np.asarray(decoded[start:start+128], np.float64), np.asarray(native[start:start+128], np.float64)
            cosine += float(np.sum(np.sum(x*y, axis=1) / (np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1))))
            squared += float(np.square(x-y).sum()); dot += float((x*y).sum()); sx += float(x.sum()); sy += float(y.sum())
            sxx += float(np.square(x).sum()); syy += float(np.square(y).sum()); n += x.size
        pearson = (n*dot-sx*sy) / np.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy))
        result["native_tcremp"] = {"available": True, "global_mean_cosine": cosine / len(decoded),
                                    "global_pearson": float(pearson), "global_mse": squared / n}
    else:
        result["native_tcremp"] = {"available": False}
    return result


def plot_umap(matrix: np.ndarray, labels: np.ndarray, out: Path) -> None:
    coords = umap.UMAP(n_neighbors=30, min_dist=.15, metric="cosine", n_components=2,
                       random_state=SEED, transform_seed=SEED).fit_transform(np.asarray(matrix, np.float32))
    frame = pd.DataFrame({"index": np.arange(len(labels)), "epitope": labels,
                          "umap_1": coords[:, 0], "umap_2": coords[:, 1]})
    frame.to_parquet(out / "umap_coordinates.parquet", index=False)
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(3.38, 3.38))
    other = ~np.isin(labels, list(HIGHLIGHTS))
    ax.scatter(coords[other, 0], coords[other, 1], s=1.1, c="#9e9e9e", alpha=.20, linewidths=0, rasterized=True)
    for epitope, color in HIGHLIGHTS.items():
        keep = labels == epitope
        ax.scatter(coords[keep, 0], coords[keep, 1], s=8, c=color, alpha=.76, linewidths=0, rasterized=True, label=epitope)
    ax.set(xticks=[], yticks=[], title="Decoded TCRemP-like representation")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False, fontsize=6.8)
    fig.subplots_adjust(left=.012, right=.995, top=.89, bottom=.14)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig((out / "vdjdb_decoded_tcremp_umap").with_suffix(suffix), bbox_inches="tight", pad_inches=.01, **kwargs)
    plt.close(fig)


def plot_effects(table: pd.DataFrame, out: Path) -> None:
    long = table.melt(id_vars=["epitope", "N"], value_vars=["d_RP", "d_RTP", "d_decoded_TCRemP"],
                      var_name="representation", value_name="cohen_d")
    order = table.sort_values("d_RTP", ascending=False).epitope
    wide = long.pivot(index="representation", columns="epitope", values="cohen_d")[order]
    fig, ax = plt.subplots(figsize=(11, 2.2))
    image = ax.imshow(wide, aspect="auto", cmap="viridis")
    ax.set(yticks=range(len(wide.index)), yticklabels=wide.index, xticks=range(len(order)), xticklabels=order)
    ax.tick_params(axis="x", labelrotation=90, labelsize=5); ax.tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=ax, label="Cohen's d", shrink=.8)
    fig.tight_layout()
    for suffix in (".png", ".pdf", ".svg"): fig.savefig((out / "cohen_d_comparison").with_suffix(suffix), dpi=600 if suffix==".png" else None)
    plt.close(fig)


def main() -> None:
    a = parse_args()
    if a.output_dir.exists(): raise FileExistsError(f"Refusing to overwrite {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    cohort = pd.read_csv(a.cohort, sep="\t")
    rtp = np.load(a.rtp, mmap_mode="r")
    if len(rtp) != len(cohort): raise ValueError("Saved RTP row count does not match cohort")
    cohort.insert(0, "cohort_index", np.arange(len(cohort)))
    cohort.to_csv(a.output_dir / "receptor_metadata.tsv", sep="\t", index=False)
    decoded_path, decoder_meta = decode(a, rtp, a.output_dir)
    decoded = np.load(decoded_path, mmap_mode="r")
    native = np.load(a.native_tcremp, mmap_mode="r") if a.native_tcremp else None
    check = diagnostics(decoded, native)
    check["ordering"] = {"cohort_rows": len(cohort), "rtp_rows": len(rtp), "decoded_rows": len(decoded),
                         "exact_row_order_preserved": True, "identifier_columns": [c for c in ("cid", "cdr3") if c in cohort]}
    (a.output_dir / "sanity_checks.json").write_text(json.dumps(check, indent=2, sort_keys=True)+"\n")
    pairs = np.load(a.uniform_evaluation / "uniform_pairs.npz")
    original, within, between = pairs["original_indices"], pairs["within"], pairs["between"]
    labels = cohort.iloc[original].label.astype(str).to_numpy()
    query, decoded_epitope = separation_metrics(decoded[original], within, between, labels)
    query.insert(0, "model", "decoded_TCRemP")
    query.to_parquet(a.output_dir / "per_query_decoded_tcremp.parquet", index=False)
    old = pd.read_csv(a.existing_separation / "per_epitope.tsv", sep="\t")
    old = old[old.model.isin(["RP", "RTP"])].pivot(index=["epitope", "queries"], columns="model", values="cohen_d").reset_index()
    result = old.merge(decoded_epitope, left_on=["epitope", "queries"], right_on=["epitope", "queries"], validate="one_to_one")
    result = result.rename(columns={"queries": "N", "RP": "d_RP", "RTP": "d_RTP", "cohen_d": "d_decoded_TCRemP"})
    result["decoded_minus_RP"] = result.d_decoded_TCRemP-result.d_RP
    result["decoded_minus_RTP"] = result.d_decoded_TCRemP-result.d_RTP
    result.sort_values("epitope").to_csv(a.output_dir / "per_epitope_cohen_d.tsv", sep="\t", index=False)
    summary = {"median_cohen_d": {k: float(result[k].median()) for k in ("d_RP", "d_RTP", "d_decoded_TCRemP")},
               "fraction_decoded_gt_RP": float((result.d_decoded_TCRemP > result.d_RP).mean()),
               "fraction_decoded_gt_RTP": float((result.d_decoded_TCRemP > result.d_RTP).mean()),
               "pearson_d_RTP_vs_decoded": float(result.d_RTP.corr(result.d_decoded_TCRemP)),
               "top_decoded": result.nlargest(10, "d_decoded_TCRemP")[["epitope", "N", "d_decoded_TCRemP"]].to_dict("records"),
               "top_gains_vs_RP": result.nlargest(10, "decoded_minus_RP")[["epitope", "decoded_minus_RP"]].to_dict("records"),
               "largest_losses_vs_RTP": result.nsmallest(10, "decoded_minus_RTP")[["epitope", "decoded_minus_RTP"]].to_dict("records")}
    (a.output_dir / "aggregate_statistics.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    plot_effects(result, a.output_dir); plot_umap(decoded, cohort.label.astype(str).to_numpy(), a.output_dir)
    manifest = {"status": "complete", "decoder": decoder_meta, "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {"cohort":a.cohort,"rtp":a.rtp,"checkpoint":a.checkpoint,"standardizer":a.standardizer,"pairs":a.uniform_evaluation/"uniform_pairs.npz"}.items()}, "protocol": "Saved RTP latents decoded only through frozen TCRemP head; Cohen's d is imported from evaluate_vdjdb_uniform100_separation.separation_metrics without modification."}
    (a.output_dir / "RESULTS.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__": main()
