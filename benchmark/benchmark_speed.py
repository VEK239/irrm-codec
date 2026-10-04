"""End-to-end inference speed for every Pgen arm, measured on one device.

The per-arm speeds recorded during training cover only the trained head. For the frozen
arms that omits the encoder, which is the dominant cost and the reason those numbers
looked orders of magnitude faster than they are in practice. Encoder timings were also
collected on a different machine than the heads, so the two cannot be combined.

This script times the full path a new sequence actually travels — string in, log10 Pgen
out — for every arm on whichever device is requested, so the quality-versus-speed
comparison rests on one set of measurements.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irrm_codec.pgen_model import PgenModel
from irrm_codec.tokenization import PAD_ID, encode
from irrm_codec.utils import setup_logging

ARMS = ("irrm", "tfidf_ridge", "sceptr_mlp", "tcr_bert_mlp", "esm2_8m_mlp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="data/benchmark/trb")
    p.add_argument("--output-dir", default="results/pgen")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--n-sequences", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    return p.parse_args()


def timed(fn, sequences, repeats):
    """Fastest of several passes, so a stray scheduling hiccup does not set the number."""
    fn(sequences[: min(64, len(sequences))])  # warm up lazy initialization
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn(sequences)
        best = min(best, time.perf_counter() - started)
    return {
        "sequences_per_second": len(sequences) / best,
        "ms_per_sequence": best / len(sequences) * 1000,
        "seconds_total": best,
    }


def make_irrm(args, device):
    """Tokenize and run the sequence encoder: the whole pipeline for this arm."""
    model = PgenModel(max_len=40).to(device).eval()

    @torch.no_grad()
    def run(sequences):
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            tokens = torch.tensor([encode(s, max_len=40) for s in batch], dtype=torch.long).to(device)
            model(tokens, tokens.ne(PAD_ID))
        if device.type == "cuda":
            torch.cuda.synchronize()

    return run


def make_tfidf_ridge(args, device, train_sequences, train_targets):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    features = vectorizer.fit_transform(train_sequences)
    model = Ridge(alpha=1.0).fit(features, train_targets)

    def run(sequences):
        model.predict(vectorizer.transform(sequences))

    return run


def make_frozen(args, device, representation):
    """Encoder plus the regression head, which is what predicting a new sequence costs."""
    from benchmark.train_pgen_arm import RegressionHead

    if representation == "sceptr_mlp":
        import sceptr

        head = RegressionHead(64).to(device).eval()

        def encode_batch(batch):
            frame = pd.DataFrame({"TRBV": ["TRBV20-1"] * len(batch), "CDR3B": batch, "TRBJ": ["TRBJ2-7"] * len(batch)})
            return torch.from_numpy(sceptr.calc_vector_representations(frame))

    elif representation == "esm2_8m_mlp":
        import esm

        encoder, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        encoder = encoder.to(device).eval()
        converter = alphabet.get_batch_converter()
        head = RegressionHead(encoder.embed_dim).to(device).eval()
        last = encoder.num_layers

        def encode_batch(batch):
            _, _, tokens = converter([(str(i), s) for i, s in enumerate(batch)])
            tokens = tokens.to(device)
            states = encoder(tokens, repr_layers=[last])["representations"][last]
            keep = (
                (tokens != alphabet.padding_idx)
                & (tokens != alphabet.cls_idx)
                & (tokens != alphabet.eos_idx)
            ).unsqueeze(-1).to(states.dtype)
            return (states * keep).sum(1) / keep.sum(1).clamp_min(1e-9)

    else:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert")
        encoder = AutoModel.from_pretrained("wukevin/tcr-bert").to(device).eval()
        head = RegressionHead(encoder.config.hidden_size).to(device).eval()
        special = set(tokenizer.all_special_ids)

        def encode_batch(batch):
            enc = tokenizer([" ".join(s) for s in batch], return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            states = encoder(**enc).last_hidden_state
            keep = enc["attention_mask"].bool()
            for token_id in special:
                keep &= enc["input_ids"] != token_id
            keep = keep.unsqueeze(-1).to(states.dtype)
            return (states * keep).sum(1) / keep.sum(1).clamp_min(1e-9)

    @torch.no_grad()
    def run(sequences):
        for start in range(0, len(sequences), args.batch_size):
            head(encode_batch(sequences[start : start + args.batch_size]).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()

    return run


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / f"benchmark_speed_{device.type}.log")

    frame = pd.read_parquet(Path(args.dataset_dir) / "dataset.parquet")
    test_rows = pd.read_csv(Path(args.dataset_dir) / "manifests" / "test.tsv", sep="\t")["row_index"]
    sequences = frame.junction_aa.iloc[test_rows].tolist()[: args.n_sequences]
    train_rows = pd.read_csv(Path(args.dataset_dir) / "manifests" / "train_10k.tsv", sep="\t")["row_index"]
    log.info("device=%s sequences=%d batch=%d", device, len(sequences), args.batch_size)

    builders = {
        "irrm": lambda: make_irrm(args, device),
        "tfidf_ridge": lambda: make_tfidf_ridge(
            args, device,
            frame.junction_aa.iloc[train_rows].tolist(),
            frame.log10_pgen_1mm.to_numpy()[train_rows],
        ),
        "sceptr_mlp": lambda: make_frozen(args, device, "sceptr_mlp"),
        "tcr_bert_mlp": lambda: make_frozen(args, device, "tcr_bert_mlp"),
        "esm2_8m_mlp": lambda: make_frozen(args, device, "esm2_8m_mlp"),
    }

    results = []
    for name in args.arms:
        log.info("timing %s", name)
        measurement = timed(builders[name](), sequences, args.repeats)
        results.append({"arm": name, "device": device.type, **measurement})
        log.info(
            "%-13s %9.1f seq/s  %7.3f ms/seq",
            name, measurement["sequences_per_second"], measurement["ms_per_sequence"],
        )

    table = pd.DataFrame(results).sort_values("sequences_per_second", ascending=False)
    path = output_dir / f"speed_end_to_end_{device.type}.csv"
    table.to_csv(path, index=False)

    log.info("=" * 56)
    for _, row in table.iterrows():
        log.info("%-13s %9.1f seq/s  %7.3f ms/seq", row.arm, row.sequences_per_second, row.ms_per_sequence)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
