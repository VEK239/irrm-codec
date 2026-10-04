import numpy as np
import pandas as pd

from benchmark.evaluate_latent_sufficiency import (
    EOS_CLASS,
    aggregate_losses,
    composition_features,
    fit_probe,
    one_hot_flat,
    sequence_metrics,
    sequence_targets,
)


def test_sequence_target_roundtrip_and_linear_probe_selection():
    sequences = ["CASS", "CASR", "CATS", "CATR"] * 3
    targets = sequence_targets(sequences, 4)
    assert np.all(targets[:, 4] == EOS_CLASS)
    one_hot = one_hot_flat(targets)
    metrics, rows = sequence_metrics(one_hot, targets)
    assert metrics["token_accuracy"] == 1.0
    assert metrics["exact_accuracy"] == 1.0
    assert np.all(rows == 0)

    x = np.arange(len(sequences), dtype=np.float32)[:, None]
    train = np.arange(0, 8)
    val = np.arange(8, 10)
    test = np.arange(10, 12)
    alpha, val_pred, test_pred, trace = fit_probe(
        x, train, val, test, np.arange(len(sequences), dtype=np.float32)[:, None],
        [0.01, 1.0], lambda pred, index: np.mean((pred[:, 0] - index) ** 2),
    )
    assert alpha in {0.01, 1.0}
    assert val_pred.shape == (2, 1)
    assert test_pred.shape == (2, 1)
    assert set(trace) == {"0.01", "1.0"}


def test_controls_and_aggregate_orientation():
    table = pd.DataFrame({
        "junction_aa": ["CASS", "CATS"],
        "v_call": ["TRBV1", "TRBV2"],
        "j_call": ["TRBJ1", "TRBJ1"],
    })
    sequence_only, meta = composition_features(table, include_vj=False)
    annotated, annotated_meta = composition_features(table, include_vj=True, fit_indices=np.array([0]))
    assert sequence_only.shape == (2, 21)
    assert annotated.shape[1] > sequence_only.shape[1]
    assert meta["sequence_only"] is True
    assert annotated_meta["sequence_only"] is False

    aggregate = aggregate_losses(
        {"sequence": 0.2, "tcremp": 0.5, "pgen": 0.4},
        {"sequence": 0.2, "tcremp": 0.4, "pgen": 0.3},
        {"sequence": 0.8, "tcremp": 1.0, "pgen": 1.0},
    )
    assert aggregate["normalized_regret_by_task"]["sequence"] == 0
    assert aggregate["worst_task_normalized_regret"] > 0
    assert 0 <= aggregate["fixed_reference_hypervolume"] <= 1
