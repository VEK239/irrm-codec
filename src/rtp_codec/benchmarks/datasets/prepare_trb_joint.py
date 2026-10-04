"""Prepare the identity-aligned TRB benchmark used by joint codec training.

The TCRemP Parquet intentionally contains only numeric embedding columns.  Its
authoritative row identities live in the representations TSV written beside it
from the same ``analysis_repertoire`` by TCRemP.  This preparation therefore
uses that sidecar's ``clone_id`` values, verifies every AIRR and pgen record by
biological identity, and refuses a bare row-count-only alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
IDENTITY_COLUMNS = ["junction_aa", "v_identity", "j_identity", "locus_identity"]
PGEN_COLUMNS = ["log10_pgen", "log10_pgen_1mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--representations-path", required=True)
    parser.add_argument("--pgen-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tcremp-writer-source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--standardizer-chunk-size", type=int, default=1024)
    return parser.parse_args()


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    return logging.getLogger("prepare_trb_joint")


def normalize_gene(value: object) -> str:
    value = str(value).strip().upper()
    return value.split("*", 1)[0]


def normalize_locus(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {"trb": "beta", "b": "beta"}
    return aliases.get(normalized, normalized)


def add_identity_columns(
    table: pd.DataFrame,
    *,
    cdr3_col: str,
    v_col: str,
    j_col: str,
    locus_col: str | None = None,
    locus_value: str = "beta",
) -> pd.DataFrame:
    required = {cdr3_col, v_col, j_col}
    if locus_col is not None:
        required.add(locus_col)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Source table is missing identity columns: {sorted(missing)}")
    result = table.copy()
    result["junction_aa"] = result[cdr3_col].astype(str).str.strip().str.upper()
    result["v_identity"] = result[v_col].map(normalize_gene)
    result["j_identity"] = result[j_col].map(normalize_gene)
    if locus_col is None:
        result["locus_identity"] = normalize_locus(locus_value)
    else:
        result["locus_identity"] = result[locus_col].map(normalize_locus)
    return result


def identity_strings(table: pd.DataFrame) -> pd.Series:
    return table[IDENTITY_COLUMNS].astype(str).agg("\x1f".join, axis=1)


def require_unique_identity(table: pd.DataFrame, label: str) -> None:
    duplicated = table.duplicated(IDENTITY_COLUMNS, keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, IDENTITY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{label} contains duplicate biological identities: {examples}")


def validate_writer_contract(path: str | None) -> dict:
    if not path:
        raise ValueError(
            "--tcremp-writer-source is required: the clone-id sidecar contract must be auditable."
        )
    source_path = Path(path)
    source = source_path.read_text(encoding="utf-8")
    required_fragments = [
        "clone_representations = get_representations_df(analysis_repertoire",
        "embeddings = run_tcremp_embedding(\n        analysis_repertoire",
        "embeddings.to_parquet",
        "index=False",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise ValueError(f"TCRemP writer contract is not recognizable; missing {missing}")
    return {
        "path": str(source_path),
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "evidence": (
            "representations clone_id and embedding rows are emitted from the same "
            "analysis_repertoire before index-free Parquet serialization"
        ),
    }


def prepare_identity_table(
    airr: pd.DataFrame,
    pgen: pd.DataFrame,
    representations: pd.DataFrame,
    *,
    min_len: int,
    max_len: int,
) -> tuple[pd.DataFrame, dict]:
    airr = add_identity_columns(
        airr,
        cdr3_col="junction_aa",
        v_col="v_call",
        j_col="j_call",
        locus_col="locus",
    )
    pgen = add_identity_columns(
        pgen,
        cdr3_col="junction_aa",
        v_col="v_call",
        j_col="j_call",
        locus_col="locus",
    )
    representations = add_identity_columns(
        representations,
        cdr3_col="cdr3aa_TRB",
        v_col="v_TRB",
        j_col="j_TRB",
        locus_value="beta",
    )
    if "clone_id" not in representations.columns:
        raise ValueError("TCRemP representations must contain clone_id.")
    if representations["clone_id"].isna().any() or representations["clone_id"].duplicated().any():
        raise ValueError("TCRemP clone_id values must be complete and unique.")
    for column in PGEN_COLUMNS:
        if column not in pgen.columns:
            raise ValueError(f"pgen table is missing required target {column!r}.")

    require_unique_identity(airr, "AIRR")
    require_unique_identity(pgen, "pgen")
    require_unique_identity(representations, "TCRemP representations")

    representation_identity = representations[["clone_id", *IDENTITY_COLUMNS]].copy()
    joined = airr.merge(
        representation_identity,
        on=IDENTITY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(airr) or len(joined) != len(representations):
        raise ValueError(
            "AIRR and TCRemP representations do not match one-to-one by CDR3/V/J/locus: "
            f"airr={len(airr)} representations={len(representations)} matched={len(joined)}"
        )
    joined = joined.merge(
        pgen[[*IDENTITY_COLUMNS, *PGEN_COLUMNS]],
        on=IDENTITY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(airr) or len(joined) != len(pgen):
        raise ValueError(
            "AIRR and pgen targets do not match one-to-one by CDR3/V/J/locus: "
            f"airr={len(airr)} pgen={len(pgen)} matched={len(joined)}"
        )

    report = {
        "rows_input": int(len(joined)),
        "rows_after_locus_filter": 0,
        "dropped_missing": 0,
        "dropped_invalid_chars": 0,
        "dropped_too_short": 0,
        "dropped_too_long": 0,
        "dropped_duplicate_cdr3": 0,
        "dropped_non_finite_pgen": 0,
    }
    keep = pd.Series(True, index=joined.index)
    locus_ok = joined["locus_identity"].eq("beta")
    report["rows_after_locus_filter"] = int(locus_ok.sum())
    keep &= locus_ok

    missing = joined["junction_aa"].isin(["", "NAN", "NONE"])
    report["dropped_missing"] = int((keep & missing).sum())
    keep &= ~missing
    invalid = ~joined["junction_aa"].map(lambda sequence: set(sequence) <= VALID_AA)
    report["dropped_invalid_chars"] = int((keep & invalid).sum())
    keep &= ~invalid
    too_short = joined["junction_aa"].str.len().lt(min_len)
    report["dropped_too_short"] = int((keep & too_short).sum())
    keep &= ~too_short
    too_long = joined["junction_aa"].str.len().gt(max_len)
    report["dropped_too_long"] = int((keep & too_long).sum())
    keep &= ~too_long
    joined = joined.loc[keep].copy()

    duplicated = joined["junction_aa"].duplicated(keep="first")
    report["dropped_duplicate_cdr3"] = int(duplicated.sum())
    joined = joined.loc[~duplicated].copy()

    target_values = joined[PGEN_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(target_values.to_numpy(dtype=np.float64)).all(axis=1)
    report["dropped_non_finite_pgen"] = int((~finite).sum())
    joined = joined.loc[finite].copy()
    joined[PGEN_COLUMNS] = target_values.loc[finite].to_numpy(dtype=np.float64)
    if joined.empty:
        raise ValueError("No rows remain after benchmark validation.")

    record_ids = identity_strings(joined).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    joined.insert(0, "record_id", record_ids.to_numpy())
    joined = joined.reset_index(drop=True)
    joined.insert(0, "row_index", np.arange(len(joined), dtype=np.int64))
    report["rows_kept"] = int(len(joined))
    return joined, report


def make_splits(
    table: pd.DataFrame,
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("Split fractions are invalid.")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than one.")
    order = np.random.default_rng(seed).permutation(len(table))
    train_end = int(len(table) * train_fraction)
    val_end = train_end + int(len(table) * val_fraction)
    splits = {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:],
    }
    table["split"] = ""
    for name, indices in splits.items():
        table.loc[indices, "split"] = name
    sets = {
        name: set(table.iloc[indices]["junction_aa"].tolist())
        for name, indices in splits.items()
    }
    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()) or (table["split"] == "").any():
        raise ValueError(f"Split construction failed: overlaps={overlaps}")
    return splits, overlaps


def parquet_embedding_columns(parquet: pq.ParquetFile) -> list[str]:
    columns = [
        field.name
        for field in parquet.schema_arrow
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
    ]
    if not columns:
        raise ValueError("TCRemP Parquet has no numeric embedding columns.")
    if "clone_id" in parquet.schema_arrow.names:
        raise ValueError(
            "This preparation expects identity in the authoritative sidecar; "
            "a physical clone_id column requires a dedicated direct-id reader."
        )
    return columns


def materialize_embeddings(
    parquet_path: str,
    representations: pd.DataFrame,
    table: pd.DataFrame,
    output_path: Path,
    *,
    batch_size: int,
) -> tuple[np.memmap, dict]:
    parquet = pq.ParquetFile(parquet_path)
    columns = parquet_embedding_columns(parquet)
    if parquet.metadata.num_rows != len(representations):
        raise ValueError(
            "TCRemP Parquet and authoritative clone-id sidecar row counts differ: "
            f"parquet={parquet.metadata.num_rows} sidecar={len(representations)}"
        )
    source_clone_ids = representations["clone_id"].astype(str).to_numpy()
    destination_by_clone = {
        str(clone_id): int(row_index)
        for clone_id, row_index in zip(table["clone_id"], table["row_index"])
    }
    temporary_path = output_path.with_suffix(".npy.tmp")
    temporary_path.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(table), len(columns)),
    )
    written = np.zeros(len(table), dtype=bool)
    source_offset = 0
    try:
        for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            matrix = record_batch.to_pandas().to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(matrix).all():
                raise ValueError("TCRemP source embeddings contain NaN or infinite values.")
            batch_ids = source_clone_ids[source_offset : source_offset + len(matrix)]
            for source_row, clone_id in enumerate(batch_ids):
                destination = destination_by_clone.get(clone_id)
                if destination is not None:
                    if written[destination]:
                        raise ValueError(f"Duplicate embedding clone_id {clone_id!r}.")
                    output[destination] = matrix[source_row]
                    written[destination] = True
            source_offset += len(matrix)
        if source_offset != len(representations) or not written.all():
            missing = table.loc[~written, "clone_id"].head(10).tolist()
            raise ValueError(
                "Embedding identity materialization was incomplete: "
                f"source={source_offset}/{len(representations)} "
                f"written={int(written.sum())}/{len(table)} missing={missing}"
            )
        output.flush()
        output._mmap.close()
        temporary_path.replace(output_path)
    except Exception:
        output._mmap.close()
        temporary_path.unlink(missing_ok=True)
        raise
    matrix = np.load(output_path, mmap_mode="r")
    return matrix, {
        "source_rows": int(parquet.metadata.num_rows),
        "selected_rows": int(len(table)),
        "dimension": int(len(columns)),
        "dtype": str(matrix.dtype),
        "all_finite": True,
        "mapping": "TCRemP representations clone_id sidecar writer contract",
    }


def combine_moments(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    batch: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    batch_count = len(batch)
    batch_mean = batch.mean(axis=0, dtype=np.float64)
    batch_m2 = batch.var(axis=0, dtype=np.float64) * batch_count
    if count == 0:
        return batch_count, batch_mean, batch_m2
    delta = batch_mean - mean
    total = count + batch_count
    return (
        total,
        mean + delta * (batch_count / total),
        m2 + batch_m2 + delta * delta * (count * batch_count / total),
    )


def write_standardizer(
    path: Path,
    table: pd.DataFrame,
    embeddings: np.ndarray,
    train_indices: np.ndarray,
    *,
    chunk_size: int,
) -> dict:
    count = 0
    mean = np.zeros(embeddings.shape[1], dtype=np.float64)
    m2 = np.zeros(embeddings.shape[1], dtype=np.float64)
    for start in range(0, len(train_indices), chunk_size):
        indices = train_indices[start : start + chunk_size]
        batch = np.asarray(embeddings[indices], dtype=np.float32)
        if not np.isfinite(batch).all():
            raise ValueError("TCRemP train targets contain non-finite values.")
        count, mean, m2 = combine_moments(count, mean, m2, batch)
    raw_std = np.sqrt(m2 / count)
    zero_variance = raw_std < 1e-8
    std = np.where(zero_variance, 1.0, raw_std)

    pgen_stats = {}
    for column in PGEN_COLUMNS:
        values = table.iloc[train_indices][column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Training target {column} contains non-finite values.")
        target_std = float(values.std())
        if target_std < 1e-8:
            target_std = 1.0
        pgen_stats[column] = {"mean": float(values.mean()), "std": target_std}
    default = pgen_stats["log10_pgen_1mm"]
    np.savez(
        path,
        tcremp_mean=mean.astype(np.float32),
        tcremp_std=std.astype(np.float32),
        pgen_mean=np.float32(default["mean"]),
        pgen_std=np.float32(default["std"]),
        pgen_target=np.asarray("log10_pgen_1mm"),
        log10_pgen_mean=np.float32(pgen_stats["log10_pgen"]["mean"]),
        log10_pgen_std=np.float32(pgen_stats["log10_pgen"]["std"]),
        log10_pgen_1mm_mean=np.float32(default["mean"]),
        log10_pgen_1mm_std=np.float32(default["std"]),
        train_rows=np.int64(count),
    )
    return {
        "fit_split": "train",
        "train_rows": int(count),
        "embedding_dimension": int(embeddings.shape[1]),
        "zero_variance_embedding_dimensions": int(zero_variance.sum()),
        "pgen": pgen_stats,
        "default_pgen_target": "log10_pgen_1mm",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_info(path: str, *, hash_file: bool) -> dict:
    source = Path(path)
    stat = source.stat()
    result = {"path": str(source), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if hash_file:
        result["sha256"] = sha256_file(source)
    return result


def summarize_target(table: pd.DataFrame, indices: np.ndarray, column: str) -> dict:
    values = table.iloc[indices][column].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(values)),
        "finite": int(np.isfinite(values).sum()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_path = output_dir / "READY.json"
    if ready_path.exists():
        raise FileExistsError(f"Refusing to overwrite an already ready benchmark: {ready_path}")
    log = setup_logging(output_dir / "prepare.log")
    log.info("starting identity-aligned TRB benchmark preparation")
    writer_contract = validate_writer_contract(args.tcremp_writer_source)

    airr = pd.read_csv(args.airr_path, sep="\t")
    pgen = pd.read_csv(args.pgen_path, sep="\t")
    representations = pd.read_csv(args.representations_path, sep="\t")
    table, cleaning = prepare_identity_table(
        airr,
        pgen,
        representations,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    splits, overlaps = make_splits(
        table,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    embeddings, embedding_summary = materialize_embeddings(
        args.embeddings_path,
        representations,
        table,
        output_dir / "embeddings.npy",
        batch_size=args.embedding_batch_size,
    )
    normalizer = write_standardizer(
        output_dir / "target_standardizer.npz",
        table,
        embeddings,
        splits["train"],
        chunk_size=args.standardizer_chunk_size,
    )

    output_columns = [
        "row_index",
        "record_id",
        "clone_id",
        "junction_aa",
        "v_call",
        "j_call",
        "locus",
        *PGEN_COLUMNS,
        "split",
    ]
    table[output_columns].to_parquet(output_dir / "dataset.parquet", index=False)
    table[output_columns].to_csv(output_dir / "dataset.tsv", sep="\t", index=False)
    table[["row_index", "record_id", "clone_id"]].to_csv(
        output_dir / "embedding_row_id.tsv", sep="\t", index=False
    )

    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    manifest_columns = [
        "row_index",
        "record_id",
        "clone_id",
        "junction_aa",
        "v_call",
        "j_call",
        *PGEN_COLUMNS,
    ]
    for name, indices in splits.items():
        table.iloc[np.sort(indices)][manifest_columns].to_csv(
            manifest_dir / f"{name}.tsv", sep="\t", index=False
        )
    train = splits["train"]
    for name, indices in {
        "1k": train[: min(1_000, len(train))],
        "10k": train[: min(10_000, len(train))],
        "all": train,
    }.items():
        pd.DataFrame({"row_index": np.sort(indices)}).to_csv(
            manifest_dir / f"train_{name}.tsv", sep="\t", index=False
        )

    summary = {
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "status": "ready",
        "benchmark": "TRB joint RTP-CODEC",
        "seed": args.seed,
        "max_len": args.max_len,
        "sources": {
            "airr": source_info(args.airr_path, hash_file=True),
            "tcremp_embeddings": source_info(args.embeddings_path, hash_file=False),
            "tcremp_representations": source_info(args.representations_path, hash_file=True),
            "pgen": source_info(args.pgen_path, hash_file=True),
            "tcremp_writer": writer_contract,
        },
        "alignment": {
            "method": (
                "TCRemP authoritative representations clone_id sidecar; AIRR and pgen "
                "joined one-to-one by normalized CDR3/V/J/locus identity"
            ),
            "row_order_only": False,
            "airr_rows": int(len(airr)),
            "representations_rows": int(len(representations)),
            "pgen_rows": int(len(pgen)),
            "identity_matches_before_cleaning": int(cleaning["rows_input"]),
            "selected_embedding_rows": int(embedding_summary["selected_rows"]),
        },
        "cleaning": cleaning,
        "dataset": {
            "rows": int(len(table)),
            "unique_record_ids": int(table["record_id"].nunique()),
            "unique_clone_ids": int(table["clone_id"].nunique()),
            "unique_cdr3": int(table["junction_aa"].nunique()),
            "cdr3_length_min": int(table["junction_aa"].str.len().min()),
            "cdr3_length_max": int(table["junction_aa"].str.len().max()),
        },
        "embeddings": embedding_summary,
        "splits": {
            "sizes": {name: int(len(indices)) for name, indices in splits.items()},
            "fractions": {
                "train": args.train_fraction,
                "val": args.val_fraction,
                "test": 1.0 - args.train_fraction - args.val_fraction,
            },
            "cdr3_overlap": overlaps,
        },
        "targets": {
            column: {
                name: summarize_target(table, indices, column)
                for name, indices in splits.items()
            }
            for column in PGEN_COLUMNS
        },
        "standardizer": normalizer,
        "checks": {
            "identity_alignment": True,
            "valid_amino_acids": True,
            "length_at_most_40": bool(table["junction_aa"].str.len().le(args.max_len).all()),
            "unique_cdr3": bool(table["junction_aa"].is_unique),
            "finite_pgen_targets": bool(
                np.isfinite(table[PGEN_COLUMNS].to_numpy(dtype=np.float64)).all()
            ),
            "finite_embeddings": bool(embedding_summary["all_finite"]),
            "non_overlapping_splits": not any(overlaps.values()),
            "train_only_standardization": normalizer["fit_split"] == "train",
        },
        "outputs": {
            "dataset_parquet": str(output_dir / "dataset.parquet"),
            "dataset_tsv": str(output_dir / "dataset.tsv"),
            "embeddings_npy": str(output_dir / "embeddings.npy"),
            "embedding_row_ids": str(output_dir / "embedding_row_id.tsv"),
            "manifests": str(manifest_dir),
            "standardizer": str(output_dir / "target_standardizer.npz"),
            "log": str(output_dir / "prepare.log"),
        },
    }
    if not all(summary["checks"].values()):
        raise ValueError(f"Final readiness checks failed: {summary['checks']}")
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    ready_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("benchmark ready at %s", output_dir)
    log.info("rows=%d embedding_shape=%s splits=%s", len(table), embeddings.shape, summary["splits"])
    if isinstance(embeddings, np.memmap):
        embeddings._mmap.close()


if __name__ == "__main__":
    main()
