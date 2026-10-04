"""Cache frozen per-residue states without global pooling.

Outputs are float16 ``[N, 40, D]`` arrays plus boolean masks.  The float16
storage is an I/O choice only; heads cast each batch to float32 before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irrm_codec.utils import setup_logging


ENCODERS = ("esm2_8m", "tcr_bert", "sceptr")
DIMENSIONS = {"esm2_8m": 320, "tcr_bert": 768, "sceptr": 64}
TCR_BERT_REVISION = "ef65ddc"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/benchmark/trb")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def open_outputs(output_dir, name, n_rows, max_len, dim):
    feature_tmp = output_dir / f"{name}_residue.tmp.npy"
    mask_tmp = output_dir / f"{name}_residue_mask.tmp.npy"
    features = np.lib.format.open_memmap(
        feature_tmp, mode="w+", dtype=np.float16, shape=(n_rows, max_len, dim)
    )
    mask = np.lib.format.open_memmap(
        mask_tmp, mode="w+", dtype=np.bool_, shape=(n_rows, max_len)
    )
    features[:] = 0
    mask[:] = False
    return features, mask, feature_tmp, mask_tmp


def assign_rows(features, mask, start, states, keep, sequences):
    for offset, sequence in enumerate(sequences):
        selected = states[offset][keep[offset]].detach().cpu().numpy()
        if len(selected) != len(sequence):
            raise ValueError(
                f"Residue-state count {len(selected)} != sequence length {len(sequence)}"
            )
        features[start + offset, : len(sequence)] = selected.astype(np.float16)
        mask[start + offset, : len(sequence)] = True


def encode_esm2(frame, args, features, mask, log):
    import esm

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    device = choose_device(args.device)
    model = model.eval().to(device)
    converter = alphabet.get_batch_converter()
    sequences = frame.junction_aa.tolist()
    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            _, _, tokens = converter([(str(i), seq) for i, seq in enumerate(batch)])
            tokens = tokens.to(device)
            states = model(tokens, repr_layers=[model.num_layers])["representations"][model.num_layers]
            keep = (
                (tokens != alphabet.padding_idx)
                & (tokens != alphabet.cls_idx)
                & (tokens != alphabet.eos_idx)
            )
            assign_rows(features, mask, start, states, keep, batch)
            if start % (args.batch_size * 100) == 0:
                log.info("esm2_8m residue states %d/%d", start, len(sequences))


def encode_tcr_bert(frame, args, features, mask, log):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert", revision=TCR_BERT_REVISION)
    model = AutoModel.from_pretrained("wukevin/tcr-bert", revision=TCR_BERT_REVISION)
    device = choose_device(args.device)
    model = model.eval().to(device)
    specials = set(tokenizer.all_special_ids)
    sequences = frame.junction_aa.tolist()
    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            encoded = tokenizer([" ".join(seq) for seq in batch], return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            states = model(**encoded).last_hidden_state
            keep = encoded["attention_mask"].bool()
            for token_id in specials:
                keep &= encoded["input_ids"] != token_id
            assign_rows(features, mask, start, states, keep, batch)
            if start % (args.batch_size * 100) == 0:
                log.info("tcr_bert residue states %d/%d", start, len(sequences))


def encode_sceptr(frame, args, features, mask, log):
    import logging
    import sceptr

    if choose_device(args.device).type == "cuda":
        sceptr.enable_hardware_acceleration()
    else:
        sceptr.disable_hardware_acceleration()
    instances = pd.DataFrame(
        {
            "TRBV": frame.v_call.to_numpy(),
            "CDR3B": frame.junction_aa.to_numpy(),
            "TRBJ": frame.j_call.to_numpy(),
        }
    )
    sequences = frame.junction_aa.tolist()
    # libtcrlm emits one INFO message for many accepted ambiguous terminal residues.
    # Suppress only that internal chatter; it does not change tokenisation or outputs.
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    try:
        root_logger.setLevel(logging.WARNING)
        for start in range(0, len(instances), args.batch_size):
            batch = instances.iloc[start : start + args.batch_size]
            result = sceptr.calc_residue_representations(batch)
            for offset, sequence in enumerate(sequences[start : start + len(batch)]):
                selected = result.representation_array[offset][result.compartment_mask[offset] == 6]
                if len(selected) != len(sequence):
                    raise ValueError(
                        f"SCEPTR CDR3B state count {len(selected)} != sequence length {len(sequence)}"
                    )
                features[start + offset, : len(sequence)] = selected.astype(np.float16)
                mask[start + offset, : len(sequence)] = True
            if start % (args.batch_size * 100) == 0:
                log.warning("sceptr residue states %d/%d", start, len(instances))
    finally:
        root_logger.setLevel(previous_level)


def update_catalog(path, name, entry):
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        current[name] = entry
        temporary = path.with_suffix(path.suffix + f".{name}.tmp")
        temporary.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "cache_residue_embeddings.log")
    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    if args.limit is not None:
        frame = frame.iloc[: args.limit].copy()
    if frame.junction_aa.str.len().max() > args.max_len:
        raise ValueError("A sequence exceeds max_len")
    sequence_sha = hashlib.sha256("\n".join(frame.junction_aa).encode()).hexdigest()
    builders = {"esm2_8m": encode_esm2, "tcr_bert": encode_tcr_bert, "sceptr": encode_sceptr}
    catalog_path = output_dir / "residue_representations.json"

    for name in args.encoders:
        feature_path = output_dir / f"{name}_residue.npy"
        mask_path = output_dir / f"{name}_residue_mask.npy"
        if feature_path.exists() and mask_path.exists() and not args.overwrite:
            cached = np.load(feature_path, mmap_mode="r")
            if cached.shape == (len(frame), args.max_len, DIMENSIONS[name]):
                log.info("%s residue cache already complete", name)
                continue
        started = time.perf_counter()
        features, mask, feature_tmp, mask_tmp = open_outputs(
            output_dir, name, len(frame), args.max_len, DIMENSIONS[name]
        )
        builders[name](frame, args, features, mask, log)
        features.flush()
        mask.flush()
        del features, mask
        feature_tmp.replace(feature_path)
        mask_tmp.replace(mask_path)
        features = np.load(feature_path, mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")
        lengths = frame.junction_aa.str.len().to_numpy()
        if not np.array_equal(mask.sum(axis=1), lengths):
            raise ValueError(f"{name} mask counts do not equal CDR3 lengths")
        if not np.isfinite(features).all():
            raise ValueError(f"{name} residue cache contains non-finite values")
        update_catalog(
            catalog_path,
            name,
            {
                "model": {
                    "esm2_8m": "esm2_t6_8M_UR50D",
                    "tcr_bert": f"wukevin/tcr-bert@{TCR_BERT_REVISION}",
                    "sceptr": "sceptr==1.2.0 default penultimate residue states; CDR3B mask=6",
                }[name],
                "rows": len(frame),
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "mask_true": int(mask.sum()),
                "sequence_sha256": sequence_sha,
                "features": str(feature_path.resolve()),
                "mask": str(mask_path.resolve()),
                "features_sha256": sha256_file(feature_path),
                "mask_sha256": sha256_file(mask_path),
                "seconds": round(time.perf_counter() - started, 2),
                "pooling": "none",
            },
        )
        log.info("%s accepted shape=%s", name, features.shape)


if __name__ == "__main__":
    main()
