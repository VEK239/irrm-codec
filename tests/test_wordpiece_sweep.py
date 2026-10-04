import tempfile
import unittest
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from rtp_codec.data.multitask import resolve_encoder_tokenizer
from rtp_codec.tokenization.character import AA_VOCAB
from rtp_codec.tokenization.wordpiece import validate_wordpiece_tokenizer
from rtp_codec.experiments.tokenizers.prepare_wordpiece_sweep import SPECIAL_TOKENS, audit_tokenizer


class WordPieceSweepTest(unittest.TestCase):
    def _tokenizer(self, sequences, vocab_size=44):
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = WordPieceDecoder(prefix="##")
        tokenizer.train_from_iterator(
            sequences,
            WordPieceTrainer(
                vocab_size=vocab_size,
                min_frequency=1,
                special_tokens=SPECIAL_TOKENS,
                continuing_subword_prefix="##",
                show_progress=False,
            ),
        )
        return tokenizer

    def test_special_ids_roundtrip_and_separate_encoder_vocab(self):
        sequences = ["CASSLGQETQYF", "CASSPGQETQYF", "CASSIRSSYEQYF"]
        tokenizer = self._tokenizer(sequences)
        validate_wordpiece_tokenizer(tokenizer, "test")
        for sequence in sequences:
            encoding = tokenizer.encode(sequence)
            self.assertNotIn(3, encoding.ids)
            self.assertEqual(tokenizer.decode(encoding.ids), sequence)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokenizer.json"
            tokenizer.save(str(path))
            resolved = resolve_encoder_tokenizer("wordpiece", str(path))
            self.assertEqual(resolved.name, "wordpiece")
            self.assertEqual(resolved.vocab_size, tokenizer.get_vocab_size())
            self.assertNotEqual(resolved.vocab_size, len(AA_VOCAB))

    def test_audit_uses_train_only_for_pgen_association(self):
        sequences = {
            "train": ["CASSF", "CASRF", "CATGF", "CQQYF"],
            "val": ["CASSF"],
            "test": ["CASRF"],
        }
        tokenizer = self._tokenizer(sum(sequences.values(), []))
        summary, rows = audit_tokenizer(
            tokenizer,
            sequences,
            np.asarray([-8.0, -7.0, -6.0, -5.0], dtype=np.float64),
        )
        self.assertEqual(summary["pgen_association"]["split"], "train")
        self.assertEqual(summary["pgen_association"]["target"], "log10_pgen_1mm")
        self.assertTrue(rows)
        self.assertEqual(summary["encoding"]["val"]["unknown_sequences"], 0)
        self.assertEqual(summary["encoding"]["test"]["roundtrip_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
