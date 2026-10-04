"""Regression coverage for formats brought together by the publication merge."""
import tempfile
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece

from irrm_codec.wordpiece_tokenization import (
    decode_wordpiece, encode_wordpiece_unpadded, filter_vocab_by_max_token_length,
    load_wordpiece_tokenizer,
)


@pytest.mark.parametrize("specials", [
    ["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
    ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
])
def test_author_and_student_vocabularies_preserve_ids_and_decode(specials):
    vocab = {token: i for i, token in enumerate(specials + ["C", "##A", "CASS"])}
    tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.decoder = WordPieceDecoder(prefix="##")
    tokenizer.add_special_tokens(specials)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "tokenizer.json"
        tokenizer.save(str(path))
        restored = load_wordpiece_tokenizer(path)
        assert restored.get_vocab() == vocab
        ids = encode_wordpiece_unpadded("CA", restored, 40)
        assert ids == [vocab["C"], vocab["##A"]]
        assert decode_wordpiece(ids + [vocab["[EOS]"], vocab["C"]], restored) == "CA"
        filtered = filter_vocab_by_max_token_length(restored, 1)
        assert {s: filtered.token_to_id(s) for s in specials} == {s: vocab[s] for s in specials}
        assert decode_wordpiece(encode_wordpiece_unpadded("CA", filtered, 40), filtered) == "CA"
        with pytest.raises(ValueError, match="UNK"):
            encode_wordpiece_unpadded("CY", restored, 40)


def test_sequence_keyed_pgen_cache_survives_row_filtering(tmp_path):
    import pandas as pd
    from benchmark.prepare_splits import _pgen_lookup
    path = tmp_path / "pgen.tsv"
    pd.DataFrame({
        "junction_aa": ["CASSF", "CASRF", "CATGF"],
        "log10_pgen": [-5., -6., -7.],
        "log10_pgen_1mm": [-4., -5., -6.],
    }).to_csv(path, sep="\t", index=False)
    result = _pgen_lookup(path, pd.Series(["CATGF", "CASSF"]))
    assert result.log10_pgen.tolist() == [-7., -5.]
    assert _pgen_lookup(path, pd.Series(["CQQF"])) is None
