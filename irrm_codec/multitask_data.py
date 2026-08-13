"""Prepared-benchmark data pipeline for joint IRRM-CODEC training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from irrm_codec.tokenization import (
    AA_VOCAB,
    PAD_ID,
    encode_raw,
    encode_reconstruction_pair,
)
from irrm_codec.wordpiece_tokenization import (
    WordpieceUnpaddedEncodeFn,
    load_wordpiece_tokenizer,
    wordpiece_vocab_size,
)


REQUIRED_DATASET_COLUMNS = {
    "row_index",
    "junction_aa",
    "split",
    "log10_pgen",
    "log10_pgen_1mm",
}


@dataclass(frozen=True)
class TargetStandardizer:
    tcremp_mean: np.ndarray
    tcremp_std: np.ndarray
    pgen_mean: float
    pgen_std: float
    pgen_target: str
    train_rows: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            tcremp_mean=self.tcremp_mean.astype(np.float32),
            tcremp_std=self.tcremp_std.astype(np.float32),
            pgen_mean=np.float32(self.pgen_mean),
            pgen_std=np.float32(self.pgen_std),
            pgen_target=np.asarray(self.pgen_target),
            train_rows=np.int64(self.train_rows),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TargetStandardizer":
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                tcremp_mean=payload["tcremp_mean"].astype(np.float32),
                tcremp_std=payload["tcremp_std"].astype(np.float32),
                pgen_mean=float(payload["pgen_mean"]),
                pgen_std=float(payload["pgen_std"]),
                pgen_target=str(payload["pgen_target"]),
                train_rows=int(payload["train_rows"]),
            )

    def as_torch(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "tcremp_mean": torch.from_numpy(self.tcremp_mean).to(device),
            "tcremp_std": torch.from_numpy(self.tcremp_std).to(device),
            "pgen_mean": torch.tensor(self.pgen_mean, dtype=torch.float32, device=device),
            "pgen_std": torch.tensor(self.pgen_std, dtype=torch.float32, device=device),
        }


@dataclass(frozen=True)
class EncoderTokenizer:
    name: str
    vocab_size: int
    path: str | None
    encode: Callable[[str, int], list[int]]


def resolve_encoder_tokenizer(
    tokenizer_type: str,
    tokenizer_path: str | None = None,
) -> EncoderTokenizer:
    if tokenizer_type == "char":
        if tokenizer_path:
            raise ValueError("--tokenizer-path is only valid for WordPiece input.")
        return EncoderTokenizer(
            name="char",
            vocab_size=len(AA_VOCAB),
            path=None,
            encode=encode_raw,
        )
    if tokenizer_type != "wordpiece":
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type!r}.")
    if not tokenizer_path:
        raise ValueError("--tokenizer-path is required for WordPiece input.")

    tokenizer = load_wordpiece_tokenizer(tokenizer_path)

    return EncoderTokenizer(
        name="wordpiece",
        vocab_size=wordpiece_vocab_size(tokenizer),
        path=str(Path(tokenizer_path)),
        encode=WordpieceUnpaddedEncodeFn(tokenizer),
    )


def load_prepared_benchmark(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    data_dir = Path(data_dir)
    dataset_path = data_dir / "dataset.parquet"
    embeddings_path = data_dir / "embeddings.npy"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared benchmark table is missing: {dataset_path}. "
            "Run python -m benchmark.prepare_splits first."
        )
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Prepared TCRemP matrix is missing: {embeddings_path}. "
            "Run python -m benchmark.prepare_splits first."
        )

    table = pd.read_parquet(dataset_path)
    missing = REQUIRED_DATASET_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"Prepared dataset is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("Prepared benchmark is empty.")
    if table["row_index"].duplicated().any():
        raise ValueError("row_index must be unique in the prepared benchmark.")
    expected = np.arange(len(table), dtype=np.int64)
    actual = table["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError(
            "dataset.parquet must be ordered by contiguous row_index so it aligns "
            "with embeddings.npy."
        )
    if not set(table["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError("split contains values other than train/val/test.")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D TCRemP matrix, got {embeddings.shape}.")
    if embeddings.shape[0] != len(table):
        raise ValueError(
            f"TCRemP rows ({embeddings.shape[0]}) do not match dataset rows ({len(table)})."
        )
    if embeddings.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"TCRemP matrix must be numeric, got dtype={embeddings.dtype}.")

    for column in ("log10_pgen", "log10_pgen_1mm"):
        values = table[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Prepared target {column} contains non-finite values.")
    return table, embeddings


def read_manifest_indices(data_dir: str | Path, name: str) -> np.ndarray:
    path = Path(data_dir) / "manifests" / f"{name}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest is missing: {path}")
    manifest = pd.read_csv(path, sep="\t", usecols=["row_index"])
    indices = manifest["row_index"].to_numpy(dtype=np.int64)
    if len(indices) == 0:
        raise ValueError(f"Benchmark manifest is empty: {path}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Benchmark manifest contains duplicate row_index values: {path}")
    return indices


def select_split_indices(
    table: pd.DataFrame,
    data_dir: str | Path,
    split: str,
    train_subset: str = "all",
) -> np.ndarray:
    if split == "train":
        manifest_name = "train" if train_subset == "all" else f"train_{train_subset}"
    elif split in {"val", "test"}:
        manifest_name = split
    else:
        raise ValueError(f"Unsupported split: {split!r}.")

    indices = read_manifest_indices(data_dir, manifest_name)
    if indices.min() < 0 or indices.max() >= len(table):
        raise ValueError(f"Manifest {manifest_name} contains out-of-range row indices.")
    observed = set(table.iloc[indices]["split"].unique())
    if observed != {split}:
        raise ValueError(
            f"Manifest {manifest_name} points to split values {sorted(observed)}, expected {split}."
        )
    return indices


def _combine_moments(
    count_a: int,
    mean_a: np.ndarray,
    m2_a: np.ndarray,
    batch: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    batch_count = len(batch)
    batch_mean = batch.mean(axis=0, dtype=np.float64)
    batch_var = batch.var(axis=0, dtype=np.float64)
    batch_m2 = batch_var * batch_count
    if count_a == 0:
        return batch_count, batch_mean, batch_m2
    delta = batch_mean - mean_a
    total = count_a + batch_count
    mean = mean_a + delta * (batch_count / total)
    m2 = m2_a + batch_m2 + delta * delta * (count_a * batch_count / total)
    return total, mean, m2


def compute_target_standardizer(
    table: pd.DataFrame,
    embeddings: np.ndarray,
    train_indices: np.ndarray,
    pgen_target: str,
    chunk_size: int = 1024,
) -> TargetStandardizer:
    if pgen_target not in {"log10_pgen", "log10_pgen_1mm"}:
        raise ValueError(f"Unsupported pgen target: {pgen_target!r}.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    count = 0
    mean = np.zeros(embeddings.shape[1], dtype=np.float64)
    m2 = np.zeros(embeddings.shape[1], dtype=np.float64)
    for start in range(0, len(train_indices), chunk_size):
        indices = train_indices[start : start + chunk_size]
        batch = np.asarray(embeddings[indices], dtype=np.float32)
        if not np.isfinite(batch).all():
            raise ValueError("TCRemP train targets contain NaN or infinite values.")
        count, mean, m2 = _combine_moments(count, mean, m2, batch)
    if count != len(train_indices) or count < 1:
        raise ValueError("Could not compute train-target statistics.")
    std = np.sqrt(m2 / count)
    std = np.where(std < 1e-8, 1.0, std)

    pgen = table.iloc[train_indices][pgen_target].to_numpy(dtype=np.float64)
    pgen_mean = float(pgen.mean())
    pgen_std = float(pgen.std())
    if pgen_std < 1e-8:
        pgen_std = 1.0
    return TargetStandardizer(
        tcremp_mean=mean.astype(np.float32),
        tcremp_std=std.astype(np.float32),
        pgen_mean=pgen_mean,
        pgen_std=pgen_std,
        pgen_target=pgen_target,
        train_rows=count,
    )


class MultiTaskBenchmarkDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        embeddings: np.ndarray,
        indices: np.ndarray,
        encoder_tokenizer: EncoderTokenizer,
        pgen_target: str,
        max_sequence_len: int = 40,
    ):
        self.table = table
        self.embeddings = embeddings
        self.indices = np.asarray(indices, dtype=np.int64)
        self.encoder_tokenizer = encoder_tokenizer
        self.pgen_target = pgen_target
        self.max_sequence_len = max_sequence_len

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        row_index = int(self.indices[index])
        row = self.table.iloc[row_index]
        sequence = str(row["junction_aa"])
        encoder_tokens = self.encoder_tokenizer.encode(sequence, self.max_sequence_len)
        decoder_input, reconstruction_target = encode_reconstruction_pair(
            sequence,
            max_len=self.max_sequence_len,
        )
        return {
            "row_index": row_index,
            "sequence": sequence,
            "encoder_tokens": torch.tensor(encoder_tokens, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "reconstruction_target": torch.tensor(
                reconstruction_target,
                dtype=torch.long,
            ),
            "tcremp_target": torch.from_numpy(
                np.asarray(self.embeddings[row_index], dtype=np.float32).copy()
            ),
            "pgen_target": torch.tensor(float(row[self.pgen_target]), dtype=torch.float32),
        }


def collate_multitask(batch: list[dict]) -> dict:
    encoder_tokens = torch.nn.utils.rnn.pad_sequence(
        [item["encoder_tokens"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    decoder_input = torch.nn.utils.rnn.pad_sequence(
        [item["decoder_input"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    reconstruction_target = torch.nn.utils.rnn.pad_sequence(
        [item["reconstruction_target"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    return {
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        "sequence": [item["sequence"] for item in batch],
        "encoder_tokens": encoder_tokens,
        "encoder_mask": encoder_tokens.ne(PAD_ID),
        "decoder_input": decoder_input,
        "reconstruction_target": reconstruction_target,
        "tcremp_target": torch.stack([item["tcremp_target"] for item in batch]),
        "pgen_target": torch.stack([item["pgen_target"] for item in batch]),
    }


def build_multitask_dataloader(
    dataset: MultiTaskBenchmarkDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_multitask,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
