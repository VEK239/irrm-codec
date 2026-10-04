import numpy as np
import pytest
import torch

from rtp_codec.benchmarks.representations.external_residue_models import (
    FrozenResidueDecoder,
    FrozenResiduePgenHead,
    ResidueArrayDataset,
)
from rtp_codec.benchmarks.reconstruction.train_external_residue_decoder import levenshtein_distance
from rtp_codec.benchmarks.representations.fit_external_residue_pca256 import fit_randomized_pca


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("", "", 0), ("CASS", "CASS", 0), ("CASS", "CAS", 1), ("CASS", "CATS", 1), ("CA", "CASS", 2)],
)
def test_exact_levenshtein_distance(left, right, expected):
    assert levenshtein_distance(left, right) == expected
    assert levenshtein_distance(right, left) == expected


def test_train_only_flattened_pca_is_finite_and_256_style():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(30, 4, 5)).astype(np.float32)
    train_rows = np.arange(20)
    mean, scale, components, explained = fit_randomized_pca(
        matrix, train_rows, n_components=6, oversample=2, niter=2, seed=42,
        device=torch.device("cpu"),
    )
    scores = ((matrix.reshape(30, -1) - mean) / scale) @ components
    assert mean.shape == scale.shape == (20,)
    assert components.shape == (20, 6)
    assert scores.shape == (30, 6)
    assert np.isfinite(scores).all()
    assert 0 < explained <= 1


def test_residue_dataset_requires_matching_shapes(tmp_path):
    features = np.zeros((4, 5, 3), dtype=np.float16)
    mask = np.zeros((4, 4), dtype=bool)
    np.save(tmp_path / "features.npy", features)
    np.save(tmp_path / "mask.npy", mask)
    with pytest.raises(ValueError, match="shapes"):
        ResidueArrayDataset(tmp_path / "features.npy", tmp_path / "mask.npy", [0], [1])


def test_position_aware_decoder_is_finite_and_order_sensitive():
    torch.manual_seed(7)
    model = FrozenResidueDecoder(input_dim=8, max_len=6, hidden_dim=32, num_layers=1, nhead=4).eval()
    features = torch.randn(2, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    with torch.no_grad():
        original = model(features, mask)
        reversed_states = model(features.flip(1), mask.flip(1))
    assert original.shape == (2, 6, 25)
    assert torch.isfinite(original).all()
    assert not torch.allclose(original, reversed_states)


def test_position_aware_pgen_head_is_finite():
    torch.manual_seed(9)
    model = FrozenResiduePgenHead(input_dim=12, max_len=7, hidden_dim=32, num_layers=1, nhead=4).eval()
    features = torch.randn(3, 7, 12)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        prediction = model(features, mask)
    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()


def test_models_reject_mask_shape_mismatch():
    model = FrozenResidueDecoder(input_dim=4, max_len=5, hidden_dim=16, num_layers=1, nhead=4)
    with pytest.raises(ValueError, match="mask"):
        model(torch.randn(2, 5, 4), torch.ones(2, 4, dtype=torch.bool))
