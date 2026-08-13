import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from irrm_codec.tokenization import AA_VOCAB, EOS_ID, encode_raw, encode_reconstruction_pair


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DATA_RUNTIME = HAS_TORCH and all(
    importlib.util.find_spec(package) is not None
    for package in ("numpy", "pandas", "pyarrow")
)


class TokenizationTest(unittest.TestCase):
    def test_raw_character_reconstruction_pair(self):
        sequence = "CASSLGQETQYF"
        encoded = encode_raw(sequence, max_len=40)
        decoder_input, target = encode_reconstruction_pair(sequence, max_len=40)
        self.assertEqual(decoder_input[1:], encoded)
        self.assertEqual(target[:-1], encoded)
        self.assertEqual(target[-1], EOS_ID)

    def test_raw_character_encoder_rejects_overflow(self):
        with self.assertRaises(ValueError):
            encode_raw("A" * 41, max_len=40)


class ComparisonRunnerTest(unittest.TestCase):
    def test_existing_metrics_require_matching_request(self):
        from scripts.run_multitask_tokenizer_comparison import run_one

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "char" / "seed_42"
            run_dir.mkdir(parents=True)
            (run_dir / "test_metrics.json").write_text('{"loss": 1.0}', encoding="utf-8")
            (run_dir / "comparison_request.json").write_text(
                '{"arguments": ["stale"]}',
                encoding="utf-8",
            )
            args = SimpleNamespace(
                output_root=str(root),
                data_dir="data/benchmark/trb",
                train_subset="1k",
                epochs=3,
                batch_size=2,
                force=False,
                extra_args=[],
            )
            with self.assertRaisesRegex(ValueError, "different configuration"):
                run_one(args, None, 42)


@unittest.skipUnless(HAS_TORCH, "PyTorch runtime is not installed")
class MultiTaskModelRuntimeTest(unittest.TestCase):
    def test_char_and_separate_vocab_forward_backward(self):
        import torch

        from irrm_codec.multitask_losses import IRRMCodecMultiTaskLoss
        from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer

        for input_vocab_size, share in ((len(AA_VOCAB), True), (64, False)):
            config = IRRMCodecConfig(
                input_vocab_size=input_vocab_size,
                output_vocab_size=len(AA_VOCAB),
                share_input_output_embeddings=share,
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
            model = IRRMCodecTransformer(config)
            encoder_tokens = torch.randint(4, input_vocab_size, (3, 6))
            encoder_mask = torch.ones_like(encoder_tokens, dtype=torch.bool)
            decoder_input = torch.randint(2, len(AA_VOCAB), (3, 7))
            target = torch.randint(4, len(AA_VOCAB), (3, 7))
            outputs = model(encoder_tokens, encoder_mask, decoder_input)
            self.assertEqual(outputs["latent"].shape, (3, 24))
            self.assertEqual(outputs["tcremp_standardized"].shape, (3, 12))
            self.assertEqual(outputs["pgen_standardized"].shape, (3,))
            self.assertEqual(outputs["reconstruction_logits"].shape, (3, 7, len(AA_VOCAB)))

            criterion = IRRMCodecMultiTaskLoss()
            losses = criterion(
                outputs,
                tcremp_target=torch.randn(3, 12),
                pgen_target=torch.randn(3),
                reconstruction_target=target,
                tcremp_mean=torch.zeros(12),
                tcremp_std=torch.ones(12),
                pgen_mean=torch.tensor(0.0),
                pgen_std=torch.tensor(1.0),
            )
            losses["loss"].backward()
            self.assertTrue(torch.isfinite(losses["loss"]))

    def test_partial_gradient_accumulation_group_uses_actual_size(self):
        from irrm_codec.train_multitask import _gradient_accumulation_group_size

        self.assertEqual(_gradient_accumulation_group_size(1, 3, 2), 2)
        self.assertEqual(_gradient_accumulation_group_size(2, 3, 2), 2)
        self.assertEqual(_gradient_accumulation_group_size(3, 3, 2), 1)



@unittest.skipUnless(HAS_DATA_RUNTIME, "PyTorch/pandas/pyarrow runtime is not installed")
class MultiTaskDataRuntimeTest(unittest.TestCase):
    def test_streamed_embedding_materialization_and_filter(self):
        import numpy as np
        import pandas as pd

        from benchmark.prepare_splits import filter_embedding_rows, load_embeddings
        from irrm_codec.utils import setup_logging

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = np.arange(40, dtype=np.float32).reshape(8, 5)
            parquet_path = root / "embeddings.parquet"
            pd.DataFrame(source).to_parquet(parquet_path, index=False)
            output_path = root / "embeddings.npy"
            selected = load_embeddings(
                parquet_path,
                np.array([0, 2, 3, 7]),
                len(source),
                output_path,
                setup_logging(),
                batch_size=3,
            )
            np.testing.assert_array_equal(selected, source[[0, 2, 3, 7]])
            filtered = filter_embedding_rows(
                selected,
                np.array([True, False, True, True]),
                output_path,
                chunk_size=2,
            )
            np.testing.assert_array_equal(filtered, source[[0, 3, 7]])
            if isinstance(filtered, np.memmap):
                filtered._mmap.close()

    def test_streamed_embeddings_align_by_clone_id(self):
        import numpy as np
        import pandas as pd

        from benchmark.prepare_splits import load_embeddings
        from irrm_codec.utils import setup_logging

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parquet_path = root / "embeddings.parquet"
            pd.DataFrame(
                {
                    "clone_id": ["b", "a", "c"],
                    "x": [20.0, 10.0, 30.0],
                    "y": [21.0, 11.0, 31.0],
                }
            ).to_parquet(parquet_path, index=False)
            output_path = root / "embeddings.npy"
            selected = load_embeddings(
                parquet_path,
                np.array([0, 1, 2]),
                3,
                output_path,
                setup_logging(),
                batch_size=2,
                expected_clone_ids=["a", "b", "c"],
            )
            np.testing.assert_array_equal(
                selected,
                np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32),
            )
            if isinstance(selected, np.memmap):
                selected._mmap.close()

    def test_resume_falls_back_to_checkpoint_when_no_best_exists(self):
        from irrm_codec.train_multitask import _resolve_best_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "new_output"
            output_dir.mkdir()
            resume_path = root / "last.pt"
            resume_path.write_bytes(b"checkpoint")
            self.assertEqual(
                _resolve_best_checkpoint(output_dir, str(resume_path)),
                resume_path,
            )

    def test_prepared_data_and_standardizer(self):
        import numpy as np
        import pandas as pd

        from irrm_codec.multitask_data import (
            MultiTaskBenchmarkDataset,
            build_multitask_dataloader,
            compute_target_standardizer,
            load_prepared_benchmark,
            resolve_encoder_tokenizer,
            select_split_indices,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            table = pd.DataFrame(
                {
                    "row_index": list(range(6)),
                    "junction_aa": ["CASSF", "CASRF", "CAT", "CQQ", "CAR", "CAS"],
                    "split": ["train", "train", "train", "val", "test", "test"],
                    "log10_pgen": [-5.0, -6.0, -7.0, -6.0, -7.0, -8.0],
                    "log10_pgen_1mm": [-4.0, -5.0, -6.0, -5.0, -6.0, -7.0],
                }
            )
            table.to_parquet(root / "dataset.parquet", index=False)
            embeddings = np.arange(30, dtype=np.float32).reshape(6, 5)
            np.save(root / "embeddings.npy", embeddings)
            for name, rows in (("train", [0, 1, 2]), ("val", [3]), ("test", [4, 5])):
                pd.DataFrame({"row_index": rows}).to_csv(
                    manifest_dir / f"{name}.tsv",
                    sep="\t",
                    index=False,
                )

            loaded_table, loaded_embeddings = load_prepared_benchmark(root)
            train_indices = select_split_indices(loaded_table, root, "train")
            standardizer = compute_target_standardizer(
                loaded_table,
                loaded_embeddings,
                train_indices,
                "log10_pgen_1mm",
                chunk_size=2,
            )
            np.testing.assert_allclose(
                standardizer.tcremp_mean,
                embeddings[:3].mean(axis=0),
            )
            dataset = MultiTaskBenchmarkDataset(
                loaded_table,
                loaded_embeddings,
                train_indices,
                resolve_encoder_tokenizer("char"),
                "log10_pgen_1mm",
                max_sequence_len=8,
            )
            batch = next(iter(build_multitask_dataloader(dataset, 2, shuffle=False)))
            self.assertEqual(tuple(batch["tcremp_target"].shape), (2, 5))
            self.assertEqual(tuple(batch["pgen_target"].shape), (2,))
            self.assertEqual(batch["sequence"], ["CASSF", "CASRF"])
            if isinstance(loaded_embeddings, np.memmap):
                loaded_embeddings._mmap.close()


if __name__ == "__main__":
    unittest.main()
