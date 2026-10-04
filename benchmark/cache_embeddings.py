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

ENCODERS = ("onehot", "esm2_8m", "tcr_bert", "sceptr")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="data/benchmark/trb")
    p.add_argument("--output-dir", default="data/benchmark/trb/representations")
    p.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--threads", type=int, default=8)
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


def encode_esm2(sequences, args, log):
    import esm

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    last_layer = model.num_layers
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            _, _, tokens = batch_converter([(str(i), s) for i, s in enumerate(batch)])
            states = model(tokens, repr_layers=[last_layer])["representations"][last_layer]
            keep = (
                (tokens != alphabet.padding_idx)
                & (tokens != alphabet.cls_idx)
                & (tokens != alphabet.eos_idx)
            )
            chunks.append(_mean_pool(states, keep).numpy().astype(np.float32))
            if start % (args.batch_size * 100) == 0:
                log.info("esm2_8m %d/%d", start, len(sequences))
    return np.concatenate(chunks)


def encode_tcr_bert(sequences, args, log):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert")
    model = AutoModel.from_pretrained("wukevin/tcr-bert")
    model.eval()
    special = set(tokenizer.all_special_ids)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            # TCR-BERT expects residues separated by spaces.
            enc = tokenizer([" ".join(s) for s in batch], return_tensors="pt", padding=True)
            states = model(**enc).last_hidden_state
            keep = enc["attention_mask"].bool()
            for token_id in special:
                keep &= enc["input_ids"] != token_id
            chunks.append(_mean_pool(states, keep).numpy().astype(np.float32))
            if start % (args.batch_size * 100) == 0:
                log.info("tcr_bert %d/%d", start, len(sequences))
    return np.concatenate(chunks)


def encode_sceptr(sequences, args, log, frame=None):
    import sceptr

    # SCEPTR reads V and J gene calls in addition to the CDR3, so it needs the full table.
    instances = pd.DataFrame(
        {
            "TRBV": frame["v_call"].to_numpy(),
            "CDR3B": frame["junction_aa"].to_numpy(),
            "TRBJ": frame["j_call"].to_numpy(),
        }
    )
    chunks = []
    stride = max(args.batch_size, 1024)
    for start in range(0, len(instances), stride):
        chunks.append(
            sceptr.calc_vector_representations(instances.iloc[start : start + stride]).astype(np.float32)
        )
        if start % (stride * 10) == 0:
            log.info("sceptr %d/%d", start, len(instances))
    return np.concatenate(chunks)


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "cache_embeddings.log")

    frame = pd.read_parquet(dataset_dir / "dataset.parquet")
    sequences = frame["junction_aa"].tolist()
    log.info("dataset rows=%d from %s", len(frame), dataset_dir / "dataset.parquet")

    builders = {
        "onehot": lambda: encode_onehot(sequences, args, log),
        "esm2_8m": lambda: encode_esm2(sequences, args, log),
        "tcr_bert": lambda: encode_tcr_bert(sequences, args, log),
        "sceptr": lambda: encode_sceptr(sequences, args, log, frame=frame),
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
        meta[name] = {
            "dim": int(matrix.shape[1]),
            "rows": int(matrix.shape[0]),
            "seconds": round(elapsed, 2),
            "ms_per_sequence": round(elapsed / len(frame) * 1000, 4),
            "path": str(path.resolve()),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("%s done dim=%d in %.1f s", name, matrix.shape[1], elapsed)

    # TCRemP comes from prepare_splits and needs no encoder pass.
    tcremp = dataset_dir / "embeddings.npy"
    if tcremp.exists():
        matrix = np.load(tcremp, mmap_mode="r")
        meta["tcremp"] = {
            "dim": int(matrix.shape[1]),
            "rows": int(matrix.shape[0]),
            "seconds": None,
            "ms_per_sequence": None,
            "path": str(tcremp.resolve()),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("=" * 60)
    for name, info in meta.items():
        log.info("%-10s dim=%-5d rows=%d", name, info["dim"], info["rows"])
    log.info("wrote %s", meta_path)


if __name__ == "__main__":
    main()
