import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from rtp_codec.cli import training_arguments
from rtp_codec.inference import load_encoder, main
from rtp_codec.models.codec import RTPCodecConfig, RTPCodecTransformer


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(42)
    config = RTPCodecConfig(d_model=16, latent_dim=8, nhead=2, encoder_layers=1,
                            decoder_layers=1, ff_dim=32, dropout=0, tcremp_dim=12,
                            tcremp_head_dim=16, pgen_head_dim=8)
    model = RTPCodecTransformer(config).eval()
    path = tmp_path / "best.pt"
    torch.save({"model_config": asdict(config), "model_state": model.state_dict(),
                "tokenizer": {"type": "char", "path": None}}, path)
    return path


def test_saved_checkpoint_preserves_order_and_batching(checkpoint):
    encoder = load_encoder(checkpoint)
    sequences = ["CASSLGQETQYF", "CASSIRSSYEQYF", "CASSLGQETQYF"]
    batched = encoder.encode_sequences(sequences, batch_size=2)
    singles = encoder.encode_sequences(sequences, batch_size=1)
    assert batched.shape == (3, 8)
    assert batched.dtype == np.float32
    np.testing.assert_allclose(batched, singles, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(batched[0], batched[2], atol=1e-6)
    assert not np.allclose(batched[0], batched[1])
    assert encoder.encode_sequences([]).shape == (0, 8)


def test_encoder_rejects_invalid_input(checkpoint):
    encoder = load_encoder(checkpoint)
    for sequence in ("", "CASSX", "A" * 41, None):
        with pytest.raises(ValueError):
            encoder.encode_sequences([sequence])
    with pytest.raises(ValueError, match="batch_size"):
        encoder.encode_sequences(["CASSF"], batch_size=0)


def test_tsv_cli_writes_matrix_and_metadata(checkpoint, tmp_path):
    source = tmp_path / "input.tsv"
    pd.DataFrame({"junction_aa": ["CASSF", "CASRF"]}).to_csv(source, sep="\t", index=False)
    output = tmp_path / "encoded.npy"
    argv = ["rtp-codec-encode", "--checkpoint", str(checkpoint), "--input", str(source),
            "--output", str(output), "--batch-size", "1"]
    with patch("sys.argv", argv):
        main()
    assert np.load(output).shape == (2, 8)
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["ordering_preserved"] is True
    assert metadata["shape"] == [2, 8]
    with patch("sys.argv", argv), pytest.raises(FileExistsError):
        main()


def test_paper_preset_accepts_explicit_overrides():
    root = Path(__file__).resolve().parents[2]
    path = root / "research/configs/paper/rtp.json"
    arguments = training_arguments(["--config", str(path), "--epochs", "1", "--device", "cpu"])
    from rtp_codec.training.multitask import parse_args
    with patch("sys.argv", ["train", *arguments]):
        resolved = parse_args()
    assert resolved.epochs == 1
    assert resolved.latent_dim == 128
    assert resolved.early_stopping_patience == 0
    assert resolved.tokenizer_type == "data_anchor"
    assert resolved.device == "cpu"


def test_anchored_checkpoint_resolves_relative_and_relocated_bundle(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_anchored_tokenization import AnchoredTokenizerTest
    from rtp_codec.data.multitask import resolve_encoder_tokenizer
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundle = AnchoredTokenizerTest().make_bundle(bundle_root, "data_anchor", ["CASS", "C"], ["QYF", "F"])
    tokenizer = resolve_encoder_tokenizer("data_anchor", str(bundle))
    config = RTPCodecConfig(d_model=16, latent_dim=8, nhead=2, encoder_layers=1,
                            decoder_layers=1, ff_dim=32, dropout=0, tcremp_dim=12,
                            tcremp_head_dim=16, pgen_head_dim=8,
                            input_vocab_size=tokenizer.vocab_size,
                            share_input_output_embeddings=False)
    model = RTPCodecTransformer(config)
    run = tmp_path / "run"
    run.mkdir()
    path = run / "best.pt"
    payload = {"model_config": asdict(config), "model_state": model.state_dict(),
               "tokenizer": {"type": "data_anchor", "path": "bundle/anchored_tokenizer.json"}}
    torch.save(payload, path)
    monkeypatch.chdir(tmp_path)
    relative = load_encoder(path).encode_sequences(["CASSLGQETQYF"])
    payload["tokenizer"]["path"] = "/missing/historical/tokenizer.json"
    torch.save(payload, path)
    relocated = load_encoder(path, tokenizer_path=bundle).encode_sequences(["CASSLGQETQYF"])
    np.testing.assert_allclose(relative, relocated)
