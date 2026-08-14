"""Validated WordPiece input tokenization for the joint IRRM-CODEC model."""

from pathlib import Path

from tokenizers import Tokenizer

from irrm_codec.tokenization import BOS_ID, EOS_ID, PAD_ID, UNK_ID, VALID_AA


EXPECTED_SPECIAL_IDS = {
    "[PAD]": PAD_ID,
    "[BOS]": BOS_ID,
    "[EOS]": EOS_ID,
    "[UNK]": UNK_ID,
}


def validate_wordpiece_tokenizer(tokenizer: Tokenizer, source: str | Path) -> None:
    for token, expected_id in EXPECTED_SPECIAL_IDS.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise ValueError(
                f"Tokenizer at {source} has {token}={actual_id}, expected {expected_id}."
            )


def load_wordpiece_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    tokenizer = Tokenizer.from_file(str(path))
    validate_wordpiece_tokenizer(tokenizer, path)
    return tokenizer


def encode_wordpiece_unpadded(
    sequence: str,
    tokenizer: Tokenizer,
    max_len: int,
) -> list[int]:
    sequence = "" if sequence is None else str(sequence).strip().upper()
    if not sequence:
        raise ValueError("Sequence must not be empty.")
    invalid = sorted(set(sequence).difference(VALID_AA))
    if invalid:
        raise ValueError(f"Sequence contains unsupported amino acids: {invalid}")
    token_ids = tokenizer.encode(sequence).ids
    if not token_ids:
        raise ValueError("WordPiece encoding must contain at least one token.")
    if UNK_ID in token_ids:
        raise ValueError(f"WordPiece encoding contains [UNK] for sequence {sequence!r}.")
    if len(token_ids) > max_len:
        raise ValueError(
            f"Sequence {sequence!r} encodes to {len(token_ids)} tokens, "
            f"exceeding max_len={max_len}."
        )
    return token_ids


class WordpieceUnpaddedEncodeFn:
    """Picklable callable used by multi-worker benchmark DataLoaders."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequence: str, max_len: int) -> list[int]:
        return encode_wordpiece_unpadded(sequence, self.tokenizer, max_len)


def wordpiece_vocab_size(tokenizer: Tokenizer) -> int:
    return tokenizer.get_vocab_size()
