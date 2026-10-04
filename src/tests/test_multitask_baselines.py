import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from rtp_codec.data.multitask import TargetStandardizer, resolve_encoder_tokenizer
from rtp_codec.training.objectives import RTPCodecMultiTaskLoss, MultiTaskLossWeights
from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer
from rtp_codec.tokenization.character import AA_VOCAB, EOS_ID, encode_raw, encode_reconstruction_pair


class CharacterTokenizerTest(unittest.TestCase):
    def test_character_reconstruction_pair(self):
        encoded = encode_raw("CASSLGQETQYF", max_len=40)
        decoder_input, target = encode_reconstruction_pair("CASSLGQETQYF", max_len=40)
        self.assertEqual(decoder_input[1:], encoded)
        self.assertEqual(target[:-1], encoded)
        self.assertEqual(target[-1], EOS_ID)
        self.assertEqual(resolve_encoder_tokenizer("char").vocab_size, len(AA_VOCAB))

    def test_character_tokenizer_rejects_overflow(self):
        with self.assertRaises(ValueError):
            encode_raw("A" * 41, max_len=40)


class LockedStandardizerTest(unittest.TestCase):
    def test_loader_accepts_benchmark_normalizer_with_extra_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target_standardizer.npz"
            np.savez(
                path,
                tcremp_mean=np.zeros(12, dtype=np.float32),
                tcremp_std=np.ones(12, dtype=np.float32),
                pgen_mean=np.float32(-8.3),
                pgen_std=np.float32(2.5),
                pgen_target=np.asarray("log10_pgen_1mm"),
                train_rows=np.int64(79_544),
                log10_pgen_mean=np.float32(-10.4),
            )
            loaded = TargetStandardizer.load(path)
            self.assertEqual(loaded.pgen_target, "log10_pgen_1mm")
            self.assertEqual(loaded.train_rows, 79_544)
            self.assertEqual(loaded.tcremp_mean.shape, (12,))


class MatchedBaselineLossTest(unittest.TestCase):
    def test_tiny_locked_benchmark_trains_and_writes_terminal_artifacts(self):
        from rtp_codec.training.multitask import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            manifests = data_dir / "manifests"
            output_dir = root / "output"
            manifests.mkdir(parents=True)
            table = pd.DataFrame(
                {
                    "row_index": range(6),
                    "junction_aa": ["CASSF", "CASRF", "CAT", "CQQ", "CAR", "CAS"],
                    "split": ["train", "train", "train", "val", "test", "test"],
                    "log10_pgen": [-5.0, -6.0, -7.0, -6.0, -7.0, -8.0],
                    "log10_pgen_1mm": [-4.0, -5.0, -6.0, -5.0, -6.0, -7.0],
                }
            )
            table.to_parquet(data_dir / "dataset.parquet", index=False)
            embeddings = np.arange(72, dtype=np.float32).reshape(6, 12)
            np.save(data_dir / "embeddings.npy", embeddings)
            for name, rows in (("train", [0, 1, 2]), ("val", [3]), ("test", [4, 5])):
                pd.DataFrame({"row_index": rows}).to_csv(
                    manifests / f"{name}.tsv", sep="\t", index=False
                )
            np.savez(
                data_dir / "target_standardizer.npz",
                tcremp_mean=embeddings[:3].mean(axis=0),
                tcremp_std=embeddings[:3].std(axis=0),
                pgen_mean=np.float32(-5.0),
                pgen_std=np.float32(np.std([-4.0, -5.0, -6.0])),
                pgen_target=np.asarray("log10_pgen_1mm"),
                train_rows=np.int64(3),
            )
            argv = [
                "train_multitask",
                "--data-dir", str(data_dir),
                "--output-dir", str(output_dir),
                "--standardizer-path", str(data_dir / "target_standardizer.npz"),
                "--d-model", "32",
                "--latent-dim", "24",
                "--nhead", "4",
                "--encoder-layers", "1",
                "--decoder-layers", "1",
                "--ff-dim", "64",
                "--tcremp-head-dim", "24",
                "--pgen-head-dim", "12",
                "--decoder-memory-tokens", "2",
                "--batch-size", "2",
                "--epochs", "1",
                "--early-stopping-patience", "0",
                "--num-workers", "0",
                "--device", "cpu",
                "--no-amp",
                "--no-progress",
                "--skip-generation-metrics",
                "--no-save-test-predictions",
            ]
            with patch("sys.argv", argv):
                main()
            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertTrue((output_dir / "last.pt").is_file())
            metrics = json.loads((output_dir / "test_metrics.json").read_text())
            self.assertEqual(metrics["samples"], 2)
            self.assertEqual(metrics["tokenizer_type"], "char")
            logging.shutdown()

    def test_phase1_configs_are_matched_except_condition_and_weights(self):
        root = Path(__file__).resolve().parents[2]
        r = json.loads(
            (root / "research/configs/trb/rq1-char-r-seed42/experiment_config.json").read_text()
        )
        rtp = json.loads(
            (root / "research/configs/trb/rq1-char-rtp-seed42/experiment_config.json").read_text()
        )
        for key in (
            "base_git_revision",
            "saved_pr_revision",
            "data_dir",
            "tokenizer",
            "seed",
            "train_subset",
            "pgen_target",
            "model",
            "training",
        ):
            self.assertEqual(r[key], rtp[key], key)
        self.assertEqual(r["loss"]["reconstruction"], 1.0)
        self.assertEqual(rtp["loss"]["reconstruction"], 1.0)
        self.assertEqual((r["loss"]["tcremp"], r["loss"]["pgen"]), (0.0, 0.0))
        self.assertEqual((rtp["loss"]["tcremp"], rtp["loss"]["pgen"]), (1.0, 1.0))

    def test_reconstruction_and_multitask_configs_share_architecture(self):
        torch.manual_seed(42)
        config = RTPCodecConfig(
            input_vocab_size=len(AA_VOCAB),
            output_vocab_size=len(AA_VOCAB),
            max_sequence_len=8,
            d_model=32,
            latent_dim=24,
            nhead=4,
            encoder_layers=1,
            decoder_layers=1,
            ff_dim=64,
            dropout=0.0,
            tcremp_dim=12,
            tcremp_head_dim=24,
            pgen_head_dim=12,
            decoder_memory_tokens=2,
        )
        model = RTPCodecTransformer(config)
        tokens = torch.randint(5, len(AA_VOCAB), (3, 6))
        decoder_input = torch.randint(5, len(AA_VOCAB), (3, 7))
        target = torch.randint(5, len(AA_VOCAB), (3, 7))
        outputs = model(tokens, tokens.ne(0), decoder_input)
        common = dict(
            outputs=outputs,
            tcremp_target=torch.randn(3, 12),
            pgen_target=torch.randn(3),
            reconstruction_target=target,
            tcremp_mean=torch.zeros(12),
            tcremp_std=torch.ones(12),
            pgen_mean=torch.tensor(0.0),
            pgen_std=torch.tensor(1.0),
        )
        r_losses = RTPCodecMultiTaskLoss(
            MultiTaskLossWeights(tcremp=0.0, pgen=0.0, reconstruction=1.0)
        )(**common)
        rtp_losses = RTPCodecMultiTaskLoss(
            MultiTaskLossWeights(tcremp=1.0, pgen=1.0, reconstruction=1.0)
        )(**common)
        self.assertTrue(torch.isfinite(r_losses["loss"]))
        self.assertTrue(torch.isfinite(rtp_losses["loss"]))
        self.assertTrue(torch.allclose(r_losses["loss"], r_losses["reconstruction_loss"]))
        self.assertGreater(float(rtp_losses["loss"]), float(r_losses["loss"]))


if __name__ == "__main__":
    unittest.main()
