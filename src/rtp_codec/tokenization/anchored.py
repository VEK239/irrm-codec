"""Sequence-only terminal-anchor tokenizers with a WordPiece middle.

The serialized bundle keeps protected literal N/C anchors in one input token and
uses an independently trained WordPiece model only for the remaining middle.
No V/J annotation is accepted by the runtime encoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from rtp_codec.tokenization.character import BOS_ID, EOS_ID, PAD_ID, UNK_ID, VALID_AA


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
EXPECTED_SPECIAL_IDS = {
    "[PAD]": PAD_ID,
    "[BOS]": BOS_ID,
    "[EOS]": EOS_ID,
    "[UNK]": UNK_ID,
}
SUPPORTED_KINDS = {"edge_k", "data_anchor", "germline_anchor"}


@dataclass(frozen=True)
class AnchoredEncoding:
    ids: list[int]
    tokens: list[str]
    boundary_types: list[str]
    n_anchor: str
    middle: str
    c_anchor: str


def _normalize(sequence: str, max_len: int) -> str:
    sequence = "" if sequence is None else str(sequence).strip().upper()
    if not sequence:
        raise ValueError("Sequence must not be empty.")
    invalid = sorted(set(sequence).difference(VALID_AA))
    if invalid:
        raise ValueError(f"Sequence contains unsupported amino acids: {invalid}")
    if len(sequence) > max_len:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds max_len={max_len}."
        )
    return sequence


class AnchoredTokenizer:
    """Runtime encoder for an immutable anchored-tokenizer bundle."""

    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path)
        payload = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        kind = payload.get("tokenizer_type")
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"Unsupported anchored tokenizer type: {kind!r}.")
        self.kind = str(kind)
        self.k = int(payload.get("k") or 0)
        self.n_anchors = tuple(payload["n_anchors"])
        self.c_anchors = tuple(payload["c_anchors"])
        self.n_anchor_set = set(self.n_anchors)
        self.c_anchor_set = set(self.c_anchors)
        self.vocab = {str(token): int(identifier) for token, identifier in payload["vocab"].items()}
        self.id_to_token = {identifier: token for token, identifier in self.vocab.items()}
        if len(self.vocab) != len(self.id_to_token):
            raise ValueError("Anchored vocabulary IDs must be unique.")
        for token, expected in EXPECTED_SPECIAL_IDS.items():
            if self.vocab.get(token) != expected:
                raise ValueError(
                    f"Anchored tokenizer has {token}={self.vocab.get(token)}, expected {expected}."
                )
        expected_ids = list(range(len(self.vocab)))
        if sorted(self.id_to_token) != expected_ids:
            raise ValueError("Anchored vocabulary IDs must be contiguous from zero.")
        central_path = self.bundle_path.parent / payload["central_tokenizer_file"]
        self.central = Tokenizer.from_file(str(central_path))
        for token, expected in EXPECTED_SPECIAL_IDS.items():
            if self.central.token_to_id(token) != expected:
                raise ValueError(
                    f"Central WordPiece has {token}={self.central.token_to_id(token)}, expected {expected}."
                )
        self.central_to_input = {
            int(key): int(value) for key, value in payload["central_to_input"].items()
        }
        self.input_to_central = {value: key for key, value in self.central_to_input.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _select_anchors(self, sequence: str) -> tuple[str, str]:
        if self.kind == "edge_k":
            if len(sequence) < 2 * self.k:
                raise ValueError(
                    f"EDGE-{self.k} requires sequence length >= {2 * self.k}, got {len(sequence)}."
                )
            n_anchor = sequence[: self.k]
            c_anchor = sequence[-self.k :]
            if n_anchor not in self.n_anchor_set or c_anchor not in self.c_anchor_set:
                raise ValueError(
                    f"EDGE-{self.k} literal is absent from the exhaustive anchor inventory: {sequence!r}."
                )
            return n_anchor, c_anchor

        n_matches = [anchor for anchor in self.n_anchors if sequence.startswith(anchor)]
        c_matches = [anchor for anchor in self.c_anchors if sequence.endswith(anchor)]
        feasible = [
            (n_anchor, c_anchor)
            for n_anchor in n_matches
            for c_anchor in c_matches
            if len(n_anchor) + len(c_anchor) <= len(sequence)
        ]
        if not feasible:
            raise ValueError(f"No non-overlapping protected anchor pair matches {sequence!r}.")
        # Longest combined literal coverage, then longer N, then longer C, then lexical.
        return min(
            feasible,
            key=lambda pair: (
                -(len(pair[0]) + len(pair[1])),
                -len(pair[0]),
                -len(pair[1]),
                pair[0],
                pair[1],
            ),
        )

    def encode_with_boundaries(self, sequence: str, max_len: int) -> AnchoredEncoding:
        sequence = _normalize(sequence, max_len=max_len)
        n_anchor, c_anchor = self._select_anchors(sequence)
        middle = sequence[len(n_anchor) : len(sequence) - len(c_anchor)]
        central_encoding = self.central.encode(middle) if middle else None
        central_ids = [] if central_encoding is None else central_encoding.ids
        if UNK_ID in central_ids:
            raise ValueError(f"Central WordPiece emitted [UNK] for middle {middle!r}.")
        try:
            input_middle_ids = [self.central_to_input[identifier] for identifier in central_ids]
            n_id = self.vocab[f"[N:{n_anchor}]"]
            c_id = self.vocab[f"[C:{c_anchor}]"]
        except KeyError as error:
            raise ValueError(f"Anchored tokenizer bundle is missing token {error.args[0]!r}.") from error
        ids = [n_id, *input_middle_ids, c_id]
        if len(ids) > max_len:
            raise ValueError(
                f"Sequence {sequence!r} encodes to {len(ids)} tokens, exceeding max_len={max_len}."
            )
        tokens = [self.id_to_token[identifier] for identifier in ids]
        return AnchoredEncoding(
            ids=ids,
            tokens=tokens,
            boundary_types=["N_ANCHOR", *(["MIDDLE_WORDPIECE"] * len(input_middle_ids)), "C_ANCHOR"],
            n_anchor=n_anchor,
            middle=middle,
            c_anchor=c_anchor,
        )

    def encode(self, sequence: str, max_len: int) -> list[int]:
        return self.encode_with_boundaries(sequence, max_len).ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token[int(identifier)] for identifier in ids]
        if len(tokens) < 2 or not tokens[0].startswith("[N:") or not tokens[-1].startswith("[C:"):
            raise ValueError("Anchored encoding must start with N_ANCHOR and end with C_ANCHOR.")
        n_anchor = tokens[0][3:-1]
        c_anchor = tokens[-1][3:-1]
        central_ids = [self.input_to_central[int(identifier)] for identifier in ids[1:-1]]
        middle = self.central.decode(central_ids, skip_special_tokens=True) if central_ids else ""
        return f"{n_anchor}{middle}{c_anchor}"


class AnchoredUnpaddedEncodeFn:
    """Picklable callable used by multi-worker benchmark DataLoaders."""

    def __init__(self, tokenizer: AnchoredTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequence: str, max_len: int) -> list[int]:
        return self.tokenizer.encode(sequence, max_len)


def load_anchored_tokenizer(path: str | Path) -> AnchoredTokenizer:
    return AnchoredTokenizer(path)
