import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip("torch")

from benchmark.cache_external_model_embeddings import ENCODERS, _mean_pool, encode_sceptr


class _Log:
    def info(self, *args, **kwargs):
        pass


class _FakeModel:
    def __init__(self, seen):
        self.seen = seen

    def calc_vector_representations(self, frame):
        self.seen.append(tuple(frame.columns))
        return np.ones((len(frame), 64), dtype=np.float32)


def test_author_global_encoder_catalog_contains_preregistered_conditions():
    assert "esm2_35m" in ENCODERS
    assert "sceptr" in ENCODERS
    assert "sceptr_cdr3" in ENCODERS


def test_mean_pool_uses_only_real_residue_positions():
    states = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [99.0, 99.0]]])
    keep = torch.tensor([[True, True, False]])
    assert torch.allclose(_mean_pool(states, keep), torch.tensor([[2.0, 4.0]]))


def test_sceptr_default_and_cdr3_only_use_native_vector_api(monkeypatch):
    default_seen, cdr3_seen = [], []
    fake = SimpleNamespace(
        enable_hardware_acceleration=lambda: None,
        calc_vector_representations=_FakeModel(default_seen).calc_vector_representations,
        variant=SimpleNamespace(cdr3_only=lambda: _FakeModel(cdr3_seen)),
    )
    monkeypatch.setitem(sys.modules, "sceptr", fake)
    frame = pd.DataFrame(
        {
            "junction_aa": ["CASSF", "CASRF"],
            "v_call": ["TRBV1", "TRBV2"],
            "j_call": ["TRBJ1", "TRBJ2"],
        }
    )
    args = SimpleNamespace(device="cpu", batch_size=2)
    default = encode_sceptr(frame["junction_aa"].tolist(), args, _Log(), frame=frame)
    cdr3 = encode_sceptr(
        frame["junction_aa"].tolist(), args, _Log(), frame=frame, cdr3_only=True
    )
    assert default.shape == cdr3.shape == (2, 64)
    assert default_seen == [("TRBV", "CDR3B", "TRBJ")]
    assert cdr3_seen == [("CDR3B",)]
