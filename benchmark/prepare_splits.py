"""Issue 1: build the common human TRB dataset shared by all benchmark experiments.

Steps:
  1. load the AIRR table and drop invalid / duplicate CDR3 amino-acid sequences
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

from irrm_codec.dataio import iter_embedding_batches, normalize_locus_name
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
    p.add_argument("--embedding-batch-size", type=int, default=1024)
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

    duplicated = df[cdr3_col].duplicated(keep="first")
    report["dropped_duplicate_cdr3"] = int(duplicated.sum())
    df = df[~duplicated].copy()

    report["rows_kept"] = int(len(df))
    if df.empty:
        raise ValueError("No sequences left after cleaning.")
    log.info("cleaning: %s", json.dumps(report))
    return df.reset_index(drop=True), df.index.copy(), report


def load_embeddings(
    path,
    keep_positions,
    n_source_rows,
    output_path,
    log,
    batch_size=1024,
    expected_clone_ids=None,
    clone_id_col="clone_id",
):
    parquet = pq.ParquetFile(path)
    n_emb = parquet.metadata.num_rows
    has_clone_ids = clone_id_col in parquet.schema_arrow.names
    if not has_clone_ids and n_emb != n_source_rows:
        raise ValueError(
            f"Embeddings have {n_emb} rows but the AIRR table had {n_source_rows} before cleaning; "
            f"row-order alignment is unsafe. Add a {clone_id_col!r} column to merge by id instead."
        )
    if batch_size < 1:
        raise ValueError("embedding batch size must be positive.")
    keep_positions = np.asarray(keep_positions, dtype=np.int64)
    if len(keep_positions) == 0 or np.any(np.diff(keep_positions) <= 0):
        raise ValueError("Kept embedding positions must be non-empty, sorted, and unique.")
    clone_id_to_output = None
    if has_clone_ids:
        if expected_clone_ids is None:
            raise ValueError(
                f"Embeddings contain {clone_id_col!r}; matching AIRR clone ids are required."
            )
        expected_clone_ids = list(expected_clone_ids)
        if len(expected_clone_ids) != len(keep_positions):
            raise ValueError("Expected clone ids must match the number of kept AIRR rows.")
        if any(pd.isna(clone_id) for clone_id in expected_clone_ids):
            raise ValueError(f"AIRR table contains missing {clone_id_col} values.")
        clone_id_to_output = {
            clone_id: output_index
            for output_index, clone_id in enumerate(expected_clone_ids)
        }
        if len(clone_id_to_output) != len(expected_clone_ids):
            raise ValueError(f"AIRR table contains duplicate {clone_id_col} values.")

    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    output_matrix = None
    source_offset = 0
    output_offset = 0
    matched_clone_ids = set()
    log.info(
        "streaming embeddings source_rows=%d kept_rows=%d parquet_cols=%d batch_size=%d",
        n_emb,
        len(keep_positions),
        parquet.metadata.num_columns,
        batch_size,
    )
    try:
        for clone_ids, batch in iter_embedding_batches(
            path,
            batch_size=batch_size,
            clone_id_col=clone_id_col,
            include_clone_id=has_clone_ids,
        ):
            if output_matrix is None:
                output_matrix = np.lib.format.open_memmap(
                    temporary_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(keep_positions), batch.shape[1]),
                )
            elif batch.shape[1] != output_matrix.shape[1]:
                raise ValueError("Embedding dimension changed between Parquet batches.")

            if has_clone_ids:
                if clone_ids is None or len(clone_ids) != len(batch):
                    raise ValueError("Embedding clone ids are missing or do not match the batch rows.")
                for batch_index, clone_id in enumerate(clone_ids):
                    destination = clone_id_to_output.get(clone_id)
                    if destination is None:
                        continue
                    if clone_id in matched_clone_ids:
                        raise ValueError(f"Embeddings contain duplicate {clone_id_col}={clone_id!r}.")
                    selected = np.asarray(batch[batch_index], dtype=np.float32)
                    if not np.isfinite(selected).all():
                        raise ValueError("Selected embeddings contain NaN or infinite values.")
                    output_matrix[destination] = selected
                    matched_clone_ids.add(clone_id)
                    output_offset += 1
            else:
                source_end = source_offset + len(batch)
                left = np.searchsorted(keep_positions, source_offset, side="left")
                right = np.searchsorted(keep_positions, source_end, side="left")
                if right > left:
                    local_positions = keep_positions[left:right] - source_offset
                    selected = np.asarray(batch[local_positions], dtype=np.float32)
                    if not np.isfinite(selected).all():
                        raise ValueError("Selected embeddings contain NaN or infinite values.")
                    output_matrix[output_offset : output_offset + len(selected)] = selected
                    output_offset += len(selected)
            source_offset += len(batch)
        if output_matrix is None:
            raise ValueError("Embeddings Parquet contains no rows.")
        if source_offset != n_emb or output_offset != len(keep_positions):
            missing_ids = []
            if has_clone_ids:
                missing_ids = [
                    clone_id
                    for clone_id in expected_clone_ids
                    if clone_id not in matched_clone_ids
                ][:10]
            raise ValueError(
                "Streaming embedding row counts do not match the AIRR selection: "
                f"source={source_offset}/{n_emb}, kept={output_offset}/{len(keep_positions)}, "
                f"missing_clone_ids={missing_ids}."
            )
        output_matrix.flush()
        output_matrix._mmap.close()
        output_matrix = None
        temporary_path.replace(output_path)
    except Exception:
        if output_matrix is not None:
            output_matrix._mmap.close()
            output_matrix = None
        temporary_path.unlink(missing_ok=True)
        raise
    return np.load(output_path, mmap_mode="r")


def filter_embedding_rows(embeddings, keep_mask, output_path, chunk_size=1024):
    """Rewrite a prepared .npy matrix without materializing it in memory."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if len(keep_mask) != len(embeddings):
        raise ValueError("Embedding filter mask length does not match the matrix.")
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".filtered.tmp")
    temporary_path.unlink(missing_ok=True)
    expected_rows = int(keep_mask.sum())
    filtered = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(expected_rows, embeddings.shape[1]),
    )
    output_offset = 0
    try:
        for start in range(0, len(embeddings), chunk_size):
            end = min(start + chunk_size, len(embeddings))
            selected = np.asarray(embeddings[start:end][keep_mask[start:end]], dtype=np.float32)
            filtered[output_offset : output_offset + len(selected)] = selected
            output_offset += len(selected)
        if output_offset != expected_rows:
            raise ValueError("Filtered embedding row count is inconsistent.")
        filtered.flush()
        filtered._mmap.close()
        filtered = None
        if isinstance(embeddings, np.memmap):
            embeddings._mmap.close()
        temporary_path.replace(output_path)
    except Exception:
        if filtered is not None:
            filtered._mmap.close()
            filtered = None
        temporary_path.unlink(missing_ok=True)
        raise
    return np.load(output_path, mmap_mode="r")


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


def ensure_pgen(df, args, output_dir, log):
    cache_path = output_dir / "pgen.tsv"
    clean_airr_path = output_dir / "cleaned_airr.tsv"
    df.to_csv(clean_airr_path, sep="\t", index=False)

    if cache_path.exists():
        cached = pd.read_csv(cache_path, sep="\t")
        same_rows = len(cached) == len(df)
        same_seqs = same_rows and np.array_equal(
            cached["junction_aa"].astype(str).to_numpy(), df["junction_aa"].to_numpy()
        )
        if same_seqs and all(c in cached.columns for c in PGEN_COLUMNS):
            log.info("reusing cached pgen table %s", cache_path)
            return cached, {"source": "cache", "path": str(cache_path)}
        log.warning("cached pgen table does not match the cleaned data; recomputing")

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

    table = pd.read_csv(cache_path, sep="\t")
    if not np.array_equal(table["junction_aa"].astype(str).to_numpy(), df["junction_aa"].to_numpy()):
        raise ValueError("pgen output rows do not line up with the cleaned AIRR rows.")
    return table, {"source": "computed", "path": str(cache_path)}


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

    embeddings_output_path = output_dir / "embeddings.npy"
    embeddings_have_clone_ids = "clone_id" in pq.ParquetFile(
        args.embeddings_path
    ).schema_arrow.names
    expected_clone_ids = None
    if embeddings_have_clone_ids:
        if "clone_id" not in clean.columns:
            raise ValueError(
                "Embeddings contain 'clone_id', but the AIRR table does not; safe id alignment is impossible."
            )
        expected_clone_ids = clean["clone_id"].tolist()
    emb = load_embeddings(
        args.embeddings_path,
        keep_positions,
        n_source_rows,
        embeddings_output_path,
        log,
        batch_size=args.embedding_batch_size,
        expected_clone_ids=expected_clone_ids,
    )
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
        emb = filter_embedding_rows(
            emb,
            finite,
            embeddings_output_path,
            chunk_size=args.embedding_batch_size,
        )
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
        "alignment_mode": "clone_id" if embeddings_have_clone_ids else "row_order",
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
