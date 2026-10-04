import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece

from irrm_codec.anchored_tokenization import AnchoredTokenizer, SPECIAL_TOKENS
from irrm_codec.multitask_data import resolve_encoder_tokenizer


class AnchoredTokenizerTest(unittest.TestCase):
    def make_bundle(self, root: Path, kind: str, n_anchors: list[str], c_anchors: list[str]):
        central_vocab = {token: identifier for identifier, token in enumerate(SPECIAL_TOKENS)}
        for amino_acid in "ACDEFGHIKLMNPQRSTVWY":
            central_vocab[amino_acid] = len(central_vocab)
        for amino_acid in "ACDEFGHIKLMNPQRSTVWY":
            central_vocab[f"##{amino_acid}"] = len(central_vocab)
        central = Tokenizer(WordPiece(vocab=central_vocab, unk_token="[UNK]"))
        central.decoder = WordPieceDecoder(prefix="##")
        central.save(str(root / "central_tokenizer.json"))

        vocab = {token: identifier for identifier, token in enumerate(SPECIAL_TOKENS)}
        for anchor in sorted(n_anchors):
            vocab[f"[N:{anchor}]"] = len(vocab)
        for anchor in sorted(c_anchors):
            vocab[f"[C:{anchor}]"] = len(vocab)
        central_to_input = {}
        by_id = {identifier: token for token, identifier in central_vocab.items()}
        for identifier in range(4, len(central_vocab)):
            central_to_input[str(identifier)] = len(vocab)
            vocab[f"[M:{by_id[identifier]}]"] = len(vocab)
        payload = {
            "tokenizer_type": kind,
            "k": 3 if kind == "edge_k" else None,
            "central_tokenizer_file": "central_tokenizer.json",
            "n_anchors": n_anchors,
            "c_anchors": c_anchors,
            "central_to_input": central_to_input,
            "vocab": vocab,
        }
        path = root / "anchored_tokenizer.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_edge_roundtrip_and_boundary_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_bundle(Path(temporary), "edge_k", ["CAS"], ["QYF"])
            tokenizer = AnchoredTokenizer(path)
            encoding = tokenizer.encode_with_boundaries("CASSLGQETQYF", 40)
            self.assertEqual(tokenizer.decode(encoding.ids), "CASSLGQETQYF")
            self.assertEqual(encoding.n_anchor, "CAS")
            self.assertEqual(encoding.middle, "SLGQET")
            self.assertEqual(encoding.c_anchor, "QYF")
            self.assertEqual(encoding.boundary_types[0], "N_ANCHOR")
            self.assertEqual(encoding.boundary_types[-1], "C_ANCHOR")
            self.assertNotIn(3, encoding.ids)

    def test_variable_longest_nonoverlapping_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_bundle(
                Path(temporary), "data_anchor", ["CASSL", "CASS", "C"], ["ETQYF", "TQYF", "F"]
            )
            tokenizer = AnchoredTokenizer(path)
            encoding = tokenizer.encode_with_boundaries("CASSLGQETQYF", 40)
            self.assertEqual((encoding.n_anchor, encoding.c_anchor), ("CASSL", "ETQYF"))
            self.assertEqual(encoding.middle, "GQ")
            self.assertEqual(tokenizer.decode(encoding.ids), "CASSLGQETQYF")

    def test_fail_closed_unknown_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_bundle(Path(temporary), "edge_k", ["CAS"], ["QYF"])
            tokenizer = AnchoredTokenizer(path)
            with self.assertRaisesRegex(ValueError, "absent from the exhaustive"):
                tokenizer.encode("CARSLGQETQYF", 40)

    def test_resolver_keeps_sequence_only_interface(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_bundle(Path(temporary), "germline_anchor", ["CASS", "C"], ["TQYF", "F"])
            resolved = resolve_encoder_tokenizer("germline_anchor", str(path))
            self.assertEqual(resolved.name, "germline_anchor")
            self.assertEqual(resolved.path, str(path))
            self.assertGreater(resolved.vocab_size, 4)
            self.assertTrue(resolved.encode("CASSLGQETQYF", 40))


if __name__ == "__main__":
    unittest.main()
