"""Prepare three leakage-safe sequence-only terminal-anchor tokenizers.

This script is intentionally a CPU batch workload: it streams an independently
generated OLGA TRB corpus, removes every locked benchmark sequence, mines anchor
libraries from train-only evidence, fits central WordPiece models, and audits
exact round trips on every locked split.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from irrm_codec.anchored_tokenization import AnchoredTokenizer, SPECIAL_TOKENS
from irrm_codec.multitask_data import load_prepared_benchmark, select_split_indices
from irrm_codec.tokenization import BOS_ID, EOS_ID, PAD_ID, UNK_ID, VALID_AA
from irrm_codec.wordpiece_tokenization import validate_wordpiece_tokenizer


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
CENTRAL_VOCAB_SIZE = 256
EDGE_K = 3
MAX_ANCHOR_LEN = 8
MIN_DATA_TOTAL_SUPPORT = 50
MIN_DATA_GENE_SUPPORT = 25
MIN_GERMLINE_SUPPORT = 20
SPECIAL_IDS = {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unk": UNK_ID}
CODONS = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def translate(nt: str) -> str:
    return "".join(CODONS.get(nt[index:index + 3].upper(), "X") for index in range(0, len(nt) - 2, 3))


def canonical_call(value: object) -> str:
    return str(value).split(",", 1)[0].split("*", 1)[0].strip()


def select_pair(sequence: str, n_anchors: tuple[str, ...], c_anchors: tuple[str, ...]) -> tuple[str, str, bool]:
    n_matches = [anchor for anchor in n_anchors if sequence.startswith(anchor)]
    c_matches = [anchor for anchor in c_anchors if sequence.endswith(anchor)]
    feasible = [(n, c) for n in n_matches for c in c_matches if len(n) + len(c) <= len(sequence)]
    if not feasible:
        raise ValueError(f"No feasible anchor pair for {sequence!r}.")
    pair = min(feasible, key=lambda value: (-(len(value[0]) + len(value[1])), -len(value[0]), -len(value[1]), value))
    raw_longest_overlap = bool(n_matches and c_matches and len(n_matches[0]) + len(c_matches[0]) > len(sequence))
    return pair[0], pair[1], raw_longest_overlap


def build_data_anchors(train) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    side_payload = {}
    libraries = []
    for side, gene_column in (("N", "v_call"), ("C", "j_call")):
        total = Counter()
        by_gene: dict[str, Counter] = defaultdict(Counter)
        for sequence, raw_gene in zip(train["junction_aa"], train[gene_column], strict=True):
            gene = canonical_call(raw_gene)
            for length in range(1, min(MAX_ANCHOR_LEN, len(sequence)) + 1):
                motif = sequence[:length] if side == "N" else sequence[-length:]
                total[motif] += 1
                by_gene[motif][gene] += 1
        retained = set(ALPHABET)
        evidence = {}
        for motif, count in total.items():
            top_gene, top_count = by_gene[motif].most_common(1)[0]
            if len(motif) >= 2 and count >= MIN_DATA_TOTAL_SUPPORT and top_count >= MIN_DATA_GENE_SUPPORT:
                retained.add(motif)
                evidence[motif] = {
                    "train_count": count,
                    "top_gene": top_gene,
                    "top_gene_count": top_count,
                    "top_gene_fraction": top_count / count,
                }
        library = tuple(sorted(retained, key=lambda value: (-len(value), value)))
        libraries.append(library)
        side_payload[side] = {
            "candidate_rule": (
                f"train-only literal {side} flanks length 2..{MAX_ANCHOR_LEN}; total support >= "
                f"{MIN_DATA_TOTAL_SUPPORT} and support within at least one annotated gene >= "
                f"{MIN_DATA_GENE_SUPPORT}; exhaustive one-residue identity fallbacks"
            ),
            "retained": len(library),
            "evidence": evidence,
        }
    return libraries[0], libraries[1], side_payload


def read_olga_genes(model_params: Path, section: str) -> dict[str, str]:
    genes = {}
    active = False
    marker = f"#GeneChoice;{section}_gene;"
    for raw_line in model_params.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("#"):
            active = raw_line.startswith(marker)
            continue
        if active and raw_line.startswith("%"):
            fields = raw_line[1:].strip().split(";")
            if len(fields) >= 2:
                genes[fields[0].strip()] = fields[1].strip().upper()
    if not genes:
        raise ValueError(f"No {section} genes found in {model_params}.")
    return genes


def read_anchor_csv(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["gene"]: int(row["anchor_index"])
        for row in rows
        if row["function"].strip("()") == "F"
    }


def build_germline_anchors(train, model_dir: Path) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    params = model_dir / "model_params.txt"
    v_anchor_path = model_dir / "V_gene_CDR3_anchors.csv"
    j_anchor_path = model_dir / "J_gene_CDR3_anchors.csv"
    v_genes = read_olga_genes(params, "V")
    j_genes = read_olga_genes(params, "J")
    v_indices = read_anchor_csv(v_anchor_path)
    j_indices = read_anchor_csv(j_anchor_path)
    reference = {"N": defaultdict(set), "C": defaultdict(set)}
    for gene, index in v_indices.items():
        if gene not in v_genes:
            continue
        amino = translate(v_genes[gene][index:])
        for length in range(1, min(MAX_ANCHOR_LEN, len(amino)) + 1):
            motif = amino[:length]
            if set(motif) <= VALID_AA:
                reference["N"][motif].add(gene)
    for gene, index in j_indices.items():
        if gene not in j_genes:
            continue
        frame_start = index % 3
        amino = translate(j_genes[gene][frame_start:index + 3])
        for length in range(1, min(MAX_ANCHOR_LEN, len(amino)) + 1):
            motif = amino[-length:]
            if set(motif) <= VALID_AA:
                reference["C"][motif].add(gene)

    payload = {}
    libraries = []
    for side in ("N", "C"):
        support = Counter()
        for sequence in train["junction_aa"]:
            for motif in reference[side]:
                if (sequence.startswith(motif) if side == "N" else sequence.endswith(motif)):
                    support[motif] += 1
        retained = {
            motif for motif, count in support.items()
            if count >= MIN_GERMLINE_SUPPORT and len(motif) >= 2
        }
        # Exhaustive one-residue identity fallbacks are constants, not learned motifs.
        retained.update(ALPHABET)
        library = tuple(sorted(retained, key=lambda value: (-len(value), value)))
        libraries.append(library)
        payload[side] = {
            "retained": len(library),
            "minimum_train_support": MIN_GERMLINE_SUPPORT,
            "motifs": {
                motif: {"train_support": support[motif], "source_genes": sorted(reference[side][motif])}
                for motif in library
            },
        }
    payload["source_files"] = {
        path.name: sha256(path) for path in (params, v_anchor_path, j_anchor_path)
    }
    payload["derivation"] = (
        "Functional OLGA TRB_orig V sequences translated from each V CDR3 anchor; functional J "
        "sequences translated in the frame ending at each J conserved-F anchor; literal prefixes/"
        "suffixes length 1..8 retained only with train support >=20."
    )
    return libraries[0], libraries[1], payload


def stream_external_corpus(source: Path, output: Path, benchmark_sequences: set[str], rows: int) -> dict:
    accepted = 0
    source_rows = 0
    rejected_overlap = 0
    rejected_invalid = 0
    rejected_duplicate = 0
    seen = set()
    with gzip.open(source, "rt", encoding="utf-8") as input_handle, output.open("w", encoding="utf-8") as out:
        header = input_handle.readline().rstrip("\n").split("\t")
        aa_index = header.index("junction_aa")
        for line in input_handle:
            source_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= aa_index:
                rejected_invalid += 1
                continue
            sequence = fields[aa_index].strip().upper()
            if not (8 <= len(sequence) <= 40 and set(sequence) <= VALID_AA and sequence.startswith("C") and sequence.endswith("F")):
                rejected_invalid += 1
                continue
            if sequence in benchmark_sequences:
                rejected_overlap += 1
                continue
            if sequence in seen:
                rejected_duplicate += 1
                continue
            seen.add(sequence)
            out.write(f"{sequence}\n")
            accepted += 1
            if accepted == rows:
                break
    if accepted != rows:
        raise ValueError(f"Requested {rows} disjoint external rows but obtained {accepted}.")
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_rows_scanned": source_rows,
        "accepted_unique_rows": accepted,
        "benchmark_overlap_rejected": rejected_overlap,
        "invalid_rejected": rejected_invalid,
        "duplicate_rejected": rejected_duplicate,
        "output_sha256": sha256(output),
        "overlap_after_filter": len(seen.intersection(benchmark_sequences)),
        "selection": "first valid unique rows in source order after locked-benchmark exclusion",
    }


def fit_central_wordpiece(middle_path: Path, output_path: Path) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder(prefix="##")
    tokenizer.train(
        [str(middle_path)],
        WordPieceTrainer(
            vocab_size=CENTRAL_VOCAB_SIZE,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix="##",
            show_progress=False,
        ),
    )
    validate_wordpiece_tokenizer(tokenizer, middle_path)
    if tokenizer.get_vocab_size() != CENTRAL_VOCAB_SIZE:
        raise ValueError(
            f"Central WordPiece requested {CENTRAL_VOCAB_SIZE}, got {tokenizer.get_vocab_size()}."
        )
    tokenizer.save(str(output_path))
    return tokenizer


def write_bundle(directory: Path, kind: str, n_anchors: tuple[str, ...], c_anchors: tuple[str, ...], central: Tokenizer, provenance: dict) -> Path:
    vocab = {token: identifier for identifier, token in enumerate(SPECIAL_TOKENS)}
    for anchor in sorted(n_anchors):
        vocab[f"[N:{anchor}]"] = len(vocab)
    for anchor in sorted(c_anchors):
        vocab[f"[C:{anchor}]"] = len(vocab)
    central_by_id = {identifier: token for token, identifier in central.get_vocab().items()}
    central_to_input = {}
    for central_id in range(len(SPECIAL_TOKENS), central.get_vocab_size()):
        token = f"[M:{central_by_id[central_id]}]"
        central_to_input[str(central_id)] = len(vocab)
        vocab[token] = len(vocab)
    bundle = {
        "format_version": 1,
        "tokenizer_type": kind,
        "inference_modality": "sequence_only",
        "accepts_v_call_or_j_call": False,
        "k": EDGE_K if kind == "edge_k" else None,
        "special_token_ids": SPECIAL_IDS,
        "central_tokenizer_file": "central_tokenizer.json",
        "central_to_input": central_to_input,
        "n_anchors": list(n_anchors),
        "c_anchors": list(c_anchors),
        "vocab": vocab,
        "vocab_size": len(vocab),
        "boundary_types": ["N_ANCHOR", "MIDDLE_WORDPIECE", "C_ANCHOR"],
        "selection_rule": (
            f"literal first/last {EDGE_K} residues" if kind == "edge_k" else
            "longest non-overlapping prefix/suffix pair; maximum total length, then N length, then C length, then lexical"
        ),
        "unknown_behavior": (
            f"fail closed only for invalid amino acids or length <{2 * EDGE_K}; every standard-amino-acid EDGE-{EDGE_K} literal is exhaustively enumerated"
            if kind == "edge_k" else
            "exhaustive one-residue identity fallbacks guarantee valid-amino-acid coverage without heldout fitting; otherwise fail closed"
        ),
        "provenance": provenance,
    }
    path = directory / "anchored_tokenizer.json"
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path


def audit(kind: str, bundle_path: Path, sequences: dict[str, list[str]], train_pgen: np.ndarray) -> tuple[dict, list[dict]]:
    tokenizer = AnchoredTokenizer(bundle_path)
    counts = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    presence = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    position_sum = np.zeros(tokenizer.vocab_size, dtype=np.float64)
    pgen_sum = np.zeros(tokenizer.vocab_size, dtype=np.float64)
    reports = {}
    for split, values in sequences.items():
        lengths, n_lengths, c_lengths = [], [], []
        mismatches = boundary_failures = overlap_adjustments = 0
        n_used, c_used = Counter(), Counter()
        for row_number, sequence in enumerate(values):
            encoded = tokenizer.encode_with_boundaries(sequence, 40)
            mismatches += int(tokenizer.decode(encoded.ids) != sequence)
            boundary_failures += int(
                encoded.boundary_types[0] != "N_ANCHOR"
                or encoded.boundary_types[-1] != "C_ANCHOR"
                or encoded.n_anchor != sequence[:len(encoded.n_anchor)]
                or encoded.c_anchor != sequence[-len(encoded.c_anchor):]
            )
            lengths.append(len(encoded.ids))
            n_lengths.append(len(encoded.n_anchor))
            c_lengths.append(len(encoded.c_anchor))
            n_used[encoded.n_anchor] += 1
            c_used[encoded.c_anchor] += 1
            if kind != "edge_k":
                _n, _c, adjusted = select_pair(sequence, tokenizer.n_anchors, tokenizer.c_anchors)
                overlap_adjustments += int(adjusted)
            if split == "train":
                unique = np.unique(encoded.ids)
                counts += np.bincount(encoded.ids, minlength=tokenizer.vocab_size)
                presence[unique] += 1
                denominator = max(len(encoded.ids) - 1, 1)
                for position, identifier in enumerate(encoded.ids):
                    position_sum[identifier] += position / denominator
                pgen_sum[unique] += float(train_pgen[row_number])
        reports[split] = {
            "rows": len(values),
            "roundtrip_mismatches": mismatches,
            "indel_or_substitution_failures": mismatches,
            "boundary_failures": boundary_failures,
            "token_length_mean": float(np.mean(lengths)),
            "token_length_p95": float(np.quantile(lengths, 0.95)),
            "token_length_max": int(max(lengths)),
            "mean_n_anchor_length": float(np.mean(n_lengths)),
            "mean_c_anchor_length": float(np.mean(c_lengths)),
            "n_anchor_length_gt1_coverage": float(np.mean(np.asarray(n_lengths) > 1)),
            "c_anchor_length_gt1_coverage": float(np.mean(np.asarray(c_lengths) > 1)),
            "unique_n_anchors_used": len(n_used),
            "unique_c_anchors_used": len(c_used),
            "longest_pair_overlap_adjustments": overlap_adjustments,
            "same_length_literal_collisions": 0,
        }
    global_mean = float(train_pgen.mean())
    global_std = float(train_pgen.std())
    rows = []
    for identifier in range(tokenizer.vocab_size):
        token = tokenizer.id_to_token[identifier]
        present = int(presence[identifier])
        absent = len(sequences["train"]) - present
        present_mean = float(pgen_sum[identifier] / present) if present else math.nan
        absent_mean = float((train_pgen.sum() - pgen_sum[identifier]) / absent) if absent else math.nan
        fraction = present / len(sequences["train"])
        association = (
            (present_mean - global_mean) * math.sqrt(fraction * (1 - fraction)) / global_std
            if present and absent and global_std > 0 else math.nan
        )
        boundary = "SPECIAL" if identifier < 4 else (
            "N_ANCHOR" if token.startswith("[N:") else "C_ANCHOR" if token.startswith("[C:") else "MIDDLE_WORDPIECE"
        )
        rows.append({
            "token_id": identifier,
            "token": token,
            "boundary_type": boundary,
            "train_occurrence_count": int(counts[identifier]),
            "train_sequence_count": present,
            "train_mean_normalized_position": (
                float(position_sum[identifier] / counts[identifier]) if counts[identifier] else ""
            ),
            "train_log10_pgen_1mm_mean_present": present_mean if present else "",
            "train_log10_pgen_1mm_mean_absent": absent_mean if absent else "",
            "train_pgen_presence_correlation": association if math.isfinite(association) else "",
        })
    return reports, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--external-corpus", required=True)
    parser.add_argument("--generation-script", required=True)
    parser.add_argument("--olga-model-dir", required=True)
    parser.add_argument("--external-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42 or args.external_rows < 100_000:
        raise ValueError("Anchored study requires seed 42 and a genuinely larger external corpus.")

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    if (output_root / "READY.json").exists():
        raise FileExistsError(f"Refusing to overwrite immutable tokenizer root {output_root}.")
    output_root.mkdir(parents=True, exist_ok=True)
    table, _ = load_prepared_benchmark(data_dir)
    indices = {split: select_split_indices(table, data_dir, split) for split in ("train", "val", "test")}
    sequences = {split: table.iloc[value]["junction_aa"].astype(str).tolist() for split, value in indices.items()}
    train = table.iloc[indices["train"]].reset_index(drop=True)
    benchmark_sequences = set(table["junction_aa"].astype(str))
    train_manifest = data_dir / "manifests" / "train.tsv"

    corpus_path = output_root / "olga_trb_disjoint_500k.txt"
    corpus_report = stream_external_corpus(
        Path(args.external_corpus), corpus_path, benchmark_sequences, args.external_rows
    )
    corpus_report.update({
        "generation_script": str(Path(args.generation_script)),
        "generation_script_sha256": sha256(Path(args.generation_script)),
        "declared_generator": "olga-generate_sequences --VDJ_model_folder=../model/Homo+sapiens/TRB_orig/",
        "fit_usage": "central WordPiece only; never anchors or evaluation metrics",
    })

    edge_n = tuple("".join(value) for value in itertools.product(ALPHABET, repeat=EDGE_K))
    edge_c = tuple("".join(value) for value in itertools.product(ALPHABET, repeat=EDGE_K))
    data_n, data_c, data_provenance = build_data_anchors(train)
    germ_n, germ_c, germ_provenance = build_germline_anchors(train, Path(args.olga_model_dir))
    candidates = {
        "edge_k": (edge_n, edge_c, {
            "anchor_fit": "exhaustive standard-amino-acid K-mer literal space on each side; no observed sequences used",
            "k": EDGE_K,
            "canonical_anchor_count_per_side": len(edge_n),
        }),
        "data_anchor": (data_n, data_c, {
            "anchor_fit": "locked train rows only; V/J calls restrict candidate support but are not runtime inputs",
            "train_manifest_sha256": sha256(train_manifest),
            "mining": data_provenance,
        }),
        "germline_anchor": (germ_n, germ_c, {
            "anchor_fit": "OLGA TRB_orig germline-compatible motifs retained by locked train support only",
            "train_manifest_sha256": sha256(train_manifest),
            "germline": germ_provenance,
        }),
    }

    top_manifest = {
        "status": "preparation_complete",
        "inference_modality": "sequence_only",
        "forbidden_model_inputs": ["v_call", "j_call"],
        "seed": args.seed,
        "locked_split_rows": {split: len(value) for split, value in indices.items()},
        "train_manifest_sha256": sha256(train_manifest),
        "external_corpus": corpus_report,
        "central_wordpiece_vocab_size": CENTRAL_VOCAB_SIZE,
        "candidates": {},
    }
    train_pgen = train["log10_pgen_1mm"].to_numpy(np.float64)
    for kind, (n_anchors, c_anchors, provenance) in candidates.items():
        final_dir = output_root / kind
        if final_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {final_dir}.")
        with tempfile.TemporaryDirectory(prefix=f".{kind}-", dir=output_root) as temporary:
            candidate_dir = Path(temporary)
            middle_path = candidate_dir / "central_corpus.txt"
            with corpus_path.open(encoding="utf-8") as input_handle, middle_path.open("w", encoding="utf-8") as out:
                for line in input_handle:
                    sequence = line.strip()
                    if kind == "edge_k":
                        n_anchor, c_anchor = sequence[:EDGE_K], sequence[-EDGE_K:]
                    else:
                        n_anchor, c_anchor, _ = select_pair(sequence, n_anchors, c_anchors)
                    middle = sequence[len(n_anchor):len(sequence) - len(c_anchor)]
                    if middle:
                        out.write(f"{middle}\n")
            central_path = candidate_dir / "central_tokenizer.json"
            central = fit_central_wordpiece(middle_path, central_path)
            provenance = dict(provenance)
            provenance.update({
                "central_corpus_sha256": sha256(middle_path),
                "external_raw_corpus_sha256": sha256(corpus_path),
                "central_fit_rows_source": args.external_rows,
                "heldout_raw_sequences_used_for_fit": False,
            })
            bundle_path = write_bundle(candidate_dir, kind, n_anchors, c_anchors, central, provenance)
            reports, token_rows = audit(kind, bundle_path, sequences, train_pgen)
            if any(
                report[check]
                for report in reports.values()
                for check in ("roundtrip_mismatches", "boundary_failures")
            ):
                raise ValueError(f"{kind} failed exact round-trip/boundary audit: {reports}")
            if max(report["token_length_max"] for report in reports.values()) > 40:
                raise ValueError(f"{kind} exceeds model max input length: {reports}")
            with (candidate_dir / "token_statistics.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(token_rows[0]))
                writer.writeheader()
                writer.writerows(token_rows)
            validation = {
                "status": "prepared",
                "checks": {
                    "sequence_only_inference": True,
                    "special_ids_0_1_2_3": True,
                    "train_only_anchor_fit": True,
                    "no_val_test_vocab_or_anchor_fit": True,
                    "external_corpus_benchmark_overlap_zero": corpus_report["overlap_after_filter"] == 0,
                    "exact_roundtrip_all_splits": True,
                    "no_indels_substitutions_all_splits": True,
                    "literal_protected_boundaries_all_splits": True,
                    "max_encoded_length_at_most_40": True,
                },
                "encoding": reports,
            }
            (candidate_dir / "validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
            config = json.loads(bundle_path.read_text(encoding="utf-8"))
            config["bundle_sha256"] = sha256(bundle_path)
            config["central_tokenizer_sha256"] = sha256(central_path)
            config["token_statistics_sha256"] = sha256(candidate_dir / "token_statistics.tsv")
            (candidate_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            candidate_dir.rename(final_dir)
        top_manifest["candidates"][kind] = {
            "path": str(final_dir),
            "input_vocab_size": config["vocab_size"],
            "n_anchor_count": len(n_anchors),
            "c_anchor_count": len(c_anchors),
            "bundle_sha256": config["bundle_sha256"],
            "central_tokenizer_sha256": config["central_tokenizer_sha256"],
            "encoding": reports,
        }
    (output_root / "preparation_manifest.json").write_text(json.dumps(top_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
