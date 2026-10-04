"""Issue 2/3: generate and cache frozen representations for the benchmark dataset.

One pretrained checkpoint per encoder, no fine-tuning. Every matrix is written in the
row order of ``dataset.parquet`` so that ``row_index`` identifies the same sequence in
every representation, including the TCRemP matrix produced by ``prepare_splits``.

Encoders:
  onehot    aligned one-hot from the repo tokenizer (gap-padded to 40 x vocab)
  esm2_8m   ESM-2 8M (esm2_t6_8M_UR50D), mean-pooled over residues
  tcr_bert  TCR-BERT (wukevin/tcr-bert), mean-pooled over residues
  sceptr    SCEPTR default variant, already one vector per receptor

Transformer encoders are mean-pooled over real residues only, excluding BOS/EOS and
padding, so the pooling rule is identical across models.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irrm_codec.tokenization import AA_VOCAB, encode
from irrm_codec.utils import setup_logging

ENCODERS = ("onehot", "esm2_8m", "esm2_35m", "tcr_bert", "sceptr", "sceptr_cdr3")
TCR_BERT_REVISION = "ef65ddc"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="data/benchmark/trb")
    p.add_argument(
        "--dataset-file",
        type=Path,
        help="Optional parquet/TSV cohort. Sequence column may be junction_aa or cdr3.",
    )
    p.add_argument("--output-dir", default="data/benchmark/trb/representations")
    p.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--rtp-embeddings", type=Path)
    p.add_argument("--overwrite", action="store_true", help="Recompute even if the cache exists.")
    return p.parse_args()


def encode_onehot(sequences, args, log):
    """Aligned one-hot: the repo's gap padding gives every sequence the same layout."""
    vocab_size = len(AA_VOCAB)
    out = np.zeros((len(sequences), args.max_len * vocab_size), dtype=np.float32)
    positions = np.arange(args.max_len)
    for row, seq in enumerate(sequences):
        block = np.zeros((args.max_len, vocab_size), dtype=np.float32)
        block[positions, encode(seq, max_len=args.max_len)] = 1.0
        out[row] = block.ravel()
    log.info("onehot done dim=%d", out.shape[1])
    return out


def _mean_pool(states, keep):
    """Average hidden states over the positions flagged in ``keep``."""
    mask = keep.unsqueeze(-1).to(states.dtype)
    return (states * mask).sum(1) / mask.sum(1).clamp_min(1e-9)


def encode_esm2(sequences, args, log, model_name="esm2_8m"):
    import esm

    factories = {
        "esm2_8m": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_35m": esm.pretrained.esm2_t12_35M_UR50D,
    }
    model, alphabet = factories[model_name]()
    device = choose_device(args.device)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    last_layer = model.num_layers
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            _, _, tokens = batch_converter([(str(i), s) for i, s in enumerate(batch)])
            tokens = tokens.to(device)
            states = model(tokens, repr_layers=[last_layer])["representations"][last_layer]
            keep = (
                (tokens != alphabet.padding_idx)
                & (tokens != alphabet.cls_idx)
                & (tokens != alphabet.eos_idx)
            )
            chunks.append(_mean_pool(states, keep).cpu().numpy().astype(np.float32))
            if start % (args.batch_size * 100) == 0:
                log.info("%s %d/%d", model_name, start, len(sequences))
    return np.concatenate(chunks)


def encode_tcr_bert(sequences, args, log):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert", revision=TCR_BERT_REVISION)
    model = AutoModel.from_pretrained("wukevin/tcr-bert", revision=TCR_BERT_REVISION)
    device = choose_device(args.device)
    model = model.eval().to(device)
    special = set(tokenizer.all_special_ids)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            # TCR-BERT expects residues separated by spaces.
            enc = tokenizer([" ".join(s) for s in batch], return_tensors="pt", padding=True)
            enc = {name: value.to(device) for name, value in enc.items()}
            states = model(**enc).last_hidden_state
            keep = enc["attention_mask"].bool()
            for token_id in special:
                keep &= enc["input_ids"] != token_id
            chunks.append(_mean_pool(states, keep).cpu().numpy().astype(np.float32))
            if start % (args.batch_size * 100) == 0:
                log.info("tcr_bert %d/%d", start, len(sequences))
    return np.concatenate(chunks)


def encode_sceptr(sequences, args, log, frame=None, cdr3_only=False):
    import logging
    import sceptr

    if choose_device(args.device).type == "cuda" and hasattr(sceptr, "enable_hardware_acceleration"):
        sceptr.enable_hardware_acceleration()
    if cdr3_only:
        model = sceptr.variant.cdr3_only()
        instances = pd.DataFrame({"CDR3B": frame["junction_aa"].to_numpy()})
    else:
        model = sceptr
        # The default author model derives beta CDR1/2 from V/J and embeds them
        # together with CDR3B. Keep this annotation-aware modality explicit.
        instances = pd.DataFrame(
            {
                "TRBV": frame["v_call"].to_numpy(),
                "CDR3B": frame["junction_aa"].to_numpy(),
                "TRBJ": frame["j_call"].to_numpy(),
            }
        )
    chunks = []
    stride = max(args.batch_size, 1024)
    # libtcrlm logs an INFO line for every accepted ambiguous terminal residue; on a
    # 100k-row cohort this can dominate runtime and create multi-megabyte Slurm logs.
    # Suppression changes no token/model output and is limited to the SCEPTR call.
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    try:
        root_logger.setLevel(logging.WARNING)
        for start in range(0, len(instances), stride):
            chunks.append(
                model.calc_vector_representations(instances.iloc[start : start + stride]).astype(np.float32)
            )
    finally:
        root_logger.setLevel(previous_level)
    log.info("sceptr encoded %d rows in %d chunks", len(instances), len(chunks))
    return np.concatenate(chunks)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is visible")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def sha256_file(path, chunk_size=1024 * 1024):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def update_catalog(meta_path, name, entry):
    """Merge one completed encoder atomically; Slurm array tasks may finish together."""
    import fcntl

    lock_path = meta_path.with_suffix(meta_path.suffix + ".lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        current[name] = entry
        temporary = meta_path.with_suffix(meta_path.suffix + f".{name}.tmp")
        temporary.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(meta_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return current


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "cache_embeddings.log")

    dataset_file = args.dataset_file or dataset_dir / "dataset.parquet"
    if dataset_file.suffix.lower() in {".tsv", ".txt"}:
        frame = pd.read_csv(dataset_file, sep="\t")
    else:
        frame = pd.read_parquet(dataset_file)
    if "junction_aa" not in frame and "cdr3" in frame:
        frame = frame.rename(columns={"cdr3": "junction_aa"})
    required = {"junction_aa"}
    if any(name == "sceptr" for name in args.encoders):
        required.update({"v_call", "j_call"})
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input cohort is missing columns: {sorted(missing)}")
    sequences = frame["junction_aa"].tolist()
    log.info("dataset rows=%d from %s", len(frame), dataset_file)

    builders = {
        "onehot": lambda: encode_onehot(sequences, args, log),
        "esm2_8m": lambda: encode_esm2(sequences, args, log, "esm2_8m"),
        "esm2_35m": lambda: encode_esm2(sequences, args, log, "esm2_35m"),
        "tcr_bert": lambda: encode_tcr_bert(sequences, args, log),
        "sceptr": lambda: encode_sceptr(sequences, args, log, frame=frame),
        "sceptr_cdr3": lambda: encode_sceptr(
            sequences, args, log, frame=frame, cdr3_only=True
        ),
    }

    meta_path = output_dir / "representations.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    for name in args.encoders:
        path = output_dir / f"{name}.npy"
        if path.exists() and not args.overwrite:
            cached = np.load(path, mmap_mode="r")
            if cached.shape[0] == len(frame):
                log.info("%s cached dim=%d, skipping", name, cached.shape[1])
                continue
            log.warning("%s cache has %d rows, expected %d; recomputing", name, cached.shape[0], len(frame))

        log.info("encoding %s", name)
        started = time.perf_counter()
        matrix = builders[name]()
        elapsed = time.perf_counter() - started

        if matrix.shape[0] != len(frame):
            raise ValueError(f"{name} produced {matrix.shape[0]} rows, expected {len(frame)}.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} produced non-finite values.")

        np.save(path, matrix)
        entry = {
            "dim": int(matrix.shape[1]),
            "rows": int(matrix.shape[0]),
            "seconds": round(elapsed, 2),
            "ms_per_sequence": round(elapsed / len(frame) * 1000, 4),
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "sequence_sha256": __import__("hashlib").sha256(
                "\n".join(sequences).encode("utf-8")
            ).hexdigest(),
        }
        entry["model"] = {
            "esm2_8m": {"id": "esm2_t6_8M_UR50D", "library": "fair-esm==2.0.0"},
            "esm2_35m": {"id": "esm2_t12_35M_UR50D", "library": "fair-esm==2.0.0"},
            "tcr_bert": {"id": "wukevin/tcr-bert", "revision": TCR_BERT_REVISION},
            "sceptr": {
                "id": "SCEPTR default native CLS64",
                "library": "sceptr==1.2.0",
                "modality": "CDR3+V+J",
                "pooling": "author-native CLS",
            },
            "sceptr_cdr3": {
                "id": "SCEPTR cdr3_only native CLS64",
                "library": "sceptr==1.2.0",
                "modality": "CDR3 sequence only",
                "pooling": "author-native CLS",
            },
            "onehot": {"id": "aligned repository one-hot", "modality": "sequence"},
        }[name]
        meta = update_catalog(meta_path, name, entry)
        log.info("%s done dim=%d in %.1f s", name, matrix.shape[1], elapsed)

    # TCRemP comes from prepare_splits and needs no encoder pass.
    tcremp = dataset_dir / "embeddings.npy"
    if tcremp.exists():
        matrix = np.load(tcremp, mmap_mode="r")
        entry = {
            "dim": int(matrix.shape[1]),
            "rows": int(matrix.shape[0]),
            "seconds": None,
            "ms_per_sequence": None,
            "path": str(tcremp.resolve()),
            "sha256": sha256_file(tcremp),
        }
        meta = update_catalog(meta_path, "tcremp", entry)

    if args.rtp_embeddings:
        matrix = np.load(args.rtp_embeddings, mmap_mode="r")
        if matrix.shape[0] != len(frame) or not np.isfinite(matrix).all():
            raise ValueError(f"RTP matrix failed shape/finiteness gate: {matrix.shape}")
        entry = {
            "dim": int(matrix.shape[1]),
            "rows": int(matrix.shape[0]),
            "seconds": 0.0,
            "ms_per_sequence": 0.0,
            "path": str(args.rtp_embeddings.resolve()),
            "sha256": sha256_file(args.rtp_embeddings),
            "checkpoint_role": "DATA-ANCHOR R+T+P seed42 validation-selected frozen encoder",
        }
        meta = update_catalog(meta_path, "rtp", entry)

    log.info("=" * 60)
    for name, info in meta.items():
        log.info("%-10s dim=%-5d rows=%d", name, info["dim"], info["rows"])
    log.info("wrote %s", meta_path)


if __name__ == "__main__":
    main()
