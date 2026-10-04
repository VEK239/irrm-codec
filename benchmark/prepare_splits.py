"""Issue 1: build the common human TRB dataset shared by all benchmark experiments.

Steps:
  1. load the AIRR table and drop invalid / duplicate CDR3 amino-acid sequences,
     then standardize V/J gene calls and drop rows no representation can encode
  2. match sequences to their TCRemP embeddings and verify the alignment
  3. compute and cache log10(pgen) and log10(pgen_1mm)
  4. build one reproducible 80/10/10 split keyed on the unique CDR3 sequence
  5. emit nested 1k / 10k / all training subsets
  6. write split manifests and a dataset summary
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tidytcells as tt

from irrm_codec.dataio import normalize_locus_name
from irrm_codec.tokenization import VALID_AA
from irrm_codec.utils import setup_logging

SUBSET_SIZES = (1_000, 10_000)

PGEN_COLUMNS = ("log10_pgen", "log10_pgen_1mm")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--airr-path", default="data/zenodo/trb_background_100k.tsv")
    p.add_argument("--embeddings-path", default="data/zenodo/trb_background_embeddings.parquet")
    p.add_argument("--output-dir", default="data/benchmark/trb")
    p.add_argument("--locus", default="beta")
    p.add_argument("--chain", default="TRB")
    p.add_argument("--species", default="human")
    p.add_argument("--max-len", type=int, default=40, help="Maximum CDR3 length the models can encode.")
    p.add_argument("--min-len", type=int, default=1)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42, help="Seed for the split permutation.")
    p.add_argument("--pgen-threads", type=int, default=8)
    p.add_argument("--pgen-chunk-size", type=int, default=1000)
    p.add_argument("--skip-pgen", action="store_true", help="Reuse a cached pgen table only; never compute.")
    p.add_argument("--alignment-min-corr", type=float, default=0.3)
    return p.parse_args()


def clean_sequences(df, cdr3_col, locus, min_len, max_len, log):
    """Drop rows the models cannot consume, then de-duplicate on the CDR3 sequence."""
    report = {"rows_input": int(len(df))}

    locus_norm = normalize_locus_name(locus)
    locus_series = df["locus"].astype(str).str.strip().str.lower().map(normalize_locus_name)
    df = df[locus_series == locus_norm].copy()
    report["rows_after_locus_filter"] = int(len(df))

    df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()

    checks = {
        "dropped_missing": df[cdr3_col].isin(["", "NAN", "NONE"]),
        "dropped_invalid_chars": ~df[cdr3_col].map(lambda s: bool(s) and set(s) <= VALID_AA),
        "dropped_too_short": df[cdr3_col].str.len() < min_len,
        "dropped_too_long": df[cdr3_col].str.len() > max_len,
    }
    drop = pd.Series(False, index=df.index)
    for name, mask in checks.items():
        mask = mask & ~drop
        report[name] = int(mask.sum())
        drop |= mask
    df = df[~drop].copy()

    # SCEPTR consumes V/J gene calls and rejects anything non-standard or
    # non-functional, while the other representations only read the CDR3. Resolving the
    # gene names here keeps a single row set that every representation can encode.
    for column, gene_type in (("v_call", "V"), ("j_call", "J")):
        df[column] = df[column].map(
            lambda value: tt.tr.standardize(
                value, species="homosapiens", enforce_functional=True, log_failures=False
            )
        )
        unusable = df[column].isna()
        report[f"dropped_non_functional_{gene_type.lower()}_call"] = int(unusable.sum())
        df = df[~unusable].copy()

    duplicated = df[cdr3_col].duplicated(keep="first")
    report["dropped_duplicate_cdr3"] = int(duplicated.sum())
    df = df[~duplicated].copy()

    report["rows_kept"] = int(len(df))
    if df.empty:
        raise ValueError("No sequences left after cleaning.")
    log.info("cleaning: %s", json.dumps(report))
    return df.reset_index(drop=True), df.index.copy(), report


def load_embeddings(path, keep_positions, n_source_rows, log):
    parquet = pq.ParquetFile(path)
    n_emb = parquet.metadata.num_rows
    if n_emb != n_source_rows:
        raise ValueError(
            f"Embeddings have {n_emb} rows but the AIRR table had {n_source_rows} before cleaning; "
            "row-order alignment is unsafe. Provide a clone_id column to merge by id instead."
        )
    log.info("reading embeddings rows=%d cols=%d", n_emb, parquet.metadata.num_columns)
    matrix = parquet.read().to_pandas().to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or infinite values.")
    return matrix[keep_positions]


def check_alignment(emb, sequences, min_corr, log):
    lengths = np.array([len(s) for s in sequences], dtype=np.float64)
    probe = emb[:, : min(600, emb.shape[1])].mean(axis=1)
    corr = float(np.corrcoef(probe, lengths)[0, 1])
    shuffled = float(
        np.corrcoef(probe, np.random.default_rng(0).permutation(lengths))[0, 1]
    )
    log.info("alignment check corr=%.4f shuffled_corr=%.4f", corr, shuffled)
    if not np.isfinite(corr) or abs(corr) < min_corr:
        raise ValueError(
            f"Embedding/sequence alignment check failed: |corr|={corr:.4f} < {min_corr}. "
            "The embedding rows are probably not aligned with the AIRR rows."
        )
    return {"length_corr": corr, "shuffled_length_corr": shuffled, "min_corr_threshold": min_corr}


def _pgen_lookup(cache_path, needed):
    """Return per-sequence pgen values from the cache, or None if it cannot serve them.

    Values are keyed on the CDR3 sequence rather than on row position, so tightening
    the cleaning filters later reuses the cache instead of recomputing it.
    """
    if not cache_path.exists():
        return None
    cached = pd.read_csv(cache_path, sep="\t")
    if not all(c in cached.columns for c in ("junction_aa", *PGEN_COLUMNS)):
        return None
    cached = cached.drop_duplicates("junction_aa").set_index("junction_aa")
    if not needed.isin(cached.index).all():
        return None
    return cached.loc[needed, list(PGEN_COLUMNS)].reset_index(drop=True)


def ensure_pgen(df, args, output_dir, log):
    cache_path = output_dir / "pgen.tsv"
    clean_airr_path = output_dir / "cleaned_airr.tsv"
    df.to_csv(clean_airr_path, sep="\t", index=False)

    hit = _pgen_lookup(cache_path, df["junction_aa"])
    if hit is not None:
        log.info("reusing cached pgen values for all %d sequences from %s", len(hit), cache_path)
        return hit, {"source": "cache", "path": str(cache_path)}
    if cache_path.exists():
        log.warning("cached pgen table does not cover the cleaned data; recomputing")

    if args.skip_pgen:
        raise FileNotFoundError(
            f"--skip-pgen was set but no matching pgen cache exists at {cache_path}."
        )

    cmd = [
        sys.executable, "-m", "irrm_codec.calc_pgen_1mm",
        "--airr-path", str(clean_airr_path),
        "--output-path", str(cache_path),
        "--chain", args.chain,
        "--species", args.species,
        "--locus", args.locus,
        "--threads", str(args.pgen_threads),
        "--chunk-size", str(args.pgen_chunk_size),
    ]
    log.info("computing pgen: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path.cwd())

    hit = _pgen_lookup(cache_path, df["junction_aa"])
    if hit is None:
        raise ValueError("pgen output does not cover every cleaned sequence.")
    return hit, {"source": "computed", "path": str(cache_path)}


def make_splits(n, train_fraction, val_fraction, seed):
    """One reproducible permutation shared by every experiment.

    Splitting on de-duplicated sequences means a CDR3 can appear in only one split.
    """
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("Fractions must lie in (0, 1) and [0, 1).")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1.")
    order = np.random.default_rng(seed).permutation(n)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    return order[:train_end], order[train_end:val_end], order[val_end:]


def summarize_target(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n_finite": 0}
    return {
        "n_finite": int(finite.size),
        "n_non_finite": int(values.size - finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "prepare_splits.log")

    raw = pd.read_csv(args.airr_path, sep="\t")
    n_source_rows = len(raw)
    log.info("loaded AIRR rows=%d from %s", n_source_rows, args.airr_path)

    clean, clean_index, clean_report = clean_sequences(
        raw, "junction_aa", args.locus, args.min_len, args.max_len, log
    )
    keep_positions = raw.index.get_indexer(clean_index)
    if (keep_positions < 0).any() or len(keep_positions) != len(clean):
        raise ValueError("Internal error: kept-row bookkeeping disagrees with the cleaned table.")

    emb = load_embeddings(args.embeddings_path, keep_positions, n_source_rows, log)
    alignment = check_alignment(emb, clean["junction_aa"].tolist(), args.alignment_min_corr, log)

    pgen_table, pgen_meta = ensure_pgen(clean, args, output_dir, log)
    for column in PGEN_COLUMNS:
        clean[column] = pgen_table[column].to_numpy(dtype=np.float64)

    # OLGA returns pgen=0 for a few sequences its generative model cannot account for,
    # which becomes -inf after the log. Such a target has no finite loss or gradient and
    # would turn every model weight into NaN, so the rows are dropped before splitting.
    # They are removed for both targets at once to keep one row set shared by every
    # model and target, as the benchmark requires.
    finite = np.ones(len(clean), dtype=bool)
    for column in PGEN_COLUMNS:
        finite &= np.isfinite(clean[column].to_numpy())
    dropped_non_finite = int((~finite).sum())
    if dropped_non_finite:
        log.warning(
            "dropping %d row(s) with non-finite pgen targets: %s",
            dropped_non_finite,
            clean.loc[~finite, ["junction_aa", *PGEN_COLUMNS]].to_dict("records"),
        )
        clean = clean[finite].reset_index(drop=True)
        emb = emb[finite]
    clean_report["dropped_non_finite_pgen"] = dropped_non_finite
    clean_report["rows_kept"] = int(len(clean))

    train_idx, val_idx, test_idx = make_splits(
        len(clean), args.train_fraction, args.val_fraction, args.seed
    )

    clean["split"] = ""
    clean.loc[clean.index[train_idx], "split"] = "train"
    clean.loc[clean.index[val_idx], "split"] = "val"
    clean.loc[clean.index[test_idx], "split"] = "test"
    if (clean["split"] == "").any():
        raise ValueError("Some rows were not assigned to a split.")

    clean.insert(0, "row_index", np.arange(len(clean), dtype=np.int64))

    subset_sizes = [s for s in SUBSET_SIZES if s < len(train_idx)] + [len(train_idx)]
    subsets = {}
    for size in subset_sizes:
        name = "all" if size == len(train_idx) else f"{size // 1000}k"
        subsets[name] = train_idx[:size]

    np.save(output_dir / "embeddings.npy", emb)
    clean.to_parquet(output_dir / "dataset.parquet", index=False)

    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        clean.iloc[np.sort(idx)][["row_index", "junction_aa", "v_call", "j_call", *PGEN_COLUMNS]].to_csv(
            manifest_dir / f"{name}.tsv", sep="\t", index=False
        )
    for name, idx in subsets.items():
        pd.DataFrame({"row_index": np.sort(idx)}).to_csv(
            manifest_dir / f"train_{name}.tsv", sep="\t", index=False
        )

    overlaps = {
        "train_val": len(set(clean.iloc[train_idx].junction_aa) & set(clean.iloc[val_idx].junction_aa)),
        "train_test": len(set(clean.iloc[train_idx].junction_aa) & set(clean.iloc[test_idx].junction_aa)),
        "val_test": len(set(clean.iloc[val_idx].junction_aa) & set(clean.iloc[test_idx].junction_aa)),
    }
    if any(overlaps.values()):
        raise ValueError(f"CDR3 sequences leak across splits: {overlaps}")

    for name, idx in subsets.items():
        if not set(idx).issubset(set(train_idx)):
            raise ValueError(f"Training subset {name} is not contained in the train split.")
    ordered = [subsets[n] for n in subsets]
    for smaller, larger in zip(ordered, ordered[1:]):
        if not set(smaller).issubset(set(larger)):
            raise ValueError("Training subsets are not nested.")

    summary = {
        "airr_path": str(Path(args.airr_path).resolve()),
        "embeddings_path": str(Path(args.embeddings_path).resolve()),
        "locus": args.locus,
        "chain": args.chain,
        "species": args.species,
        "seed": args.seed,
        "max_len": args.max_len,
        "cleaning": clean_report,
        "embedding_dim": int(emb.shape[1]),
        "embedding_alignment": alignment,
        "alignment_mode": "row_order",
        "pgen": pgen_meta,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_fractions": {
            "train": args.train_fraction,
            "val": args.val_fraction,
            "test": round(1.0 - args.train_fraction - args.val_fraction, 6),
        },
        "train_subsets": {name: int(len(idx)) for name, idx in subsets.items()},
        "cdr3_overlap_between_splits": overlaps,
        "targets": {
            column: {
                split: summarize_target(clean.loc[clean.split == split, column].to_numpy())
                for split in ("train", "val", "test")
            }
            for column in PGEN_COLUMNS
        },
        "cdr3_length": {
            "min": int(clean.junction_aa.str.len().min()),
            "median": float(clean.junction_aa.str.len().median()),
            "max": int(clean.junction_aa.str.len().max()),
        },
        "outputs": {
            "dataset": str((output_dir / "dataset.parquet").resolve()),
            "embeddings": str((output_dir / "embeddings.npy").resolve()),
            "manifests": str(manifest_dir.resolve()),
        },
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info("kept %d unique CDR3 sequences (embedding_dim=%d)", len(clean), emb.shape[1])
    log.info(
        "split train=%d val=%d test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    log.info("train subsets: %s", ", ".join(f"{k}={len(v)}" for k, v in subsets.items()))
    log.info("no CDR3 overlap between splits: %s", overlaps)
    log.info("wrote %s", output_dir / "dataset_summary.json")


if __name__ == "__main__":
    main()
