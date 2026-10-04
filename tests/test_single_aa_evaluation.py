import numpy as np
import pandas as pd

from benchmark.evaluate_single_aa_challenge import (
    bootstrap_or_disabled,
    evaluate_representation,
    identity_features,
    ridge_fit,
    ridge_predict,
)


def test_zero_bootstrap_is_explicitly_disabled_without_resampling():
    frame = pd.DataFrame({"parent_id": ["p1"], "probe_split": ["test"]})
    disabled = bootstrap_or_disabled(
        frame,
        np.array([0.0]),
        np.zeros((1, 2)),
        {},
        np.array([True]),
        repeats=0,
        seed=42,
    )
    assert disabled == {
        "status": "disabled_by_user",
        "repeats": 0,
        "uncertainty_available": False,
        "significance_claim_permitted": False,
    }


def test_ridge_and_delta_evaluator_are_finite():
    rng = np.random.default_rng(42)
    parents = np.repeat([f"p{i}" for i in range(15)], 3)
    split = np.repeat(["train"] * 9 + ["val"] * 3 + ["test"] * 3, 3)
    n = len(parents)
    frame = pd.DataFrame({
        "parent_id": parents,
        "probe_split": split,
        "mutation_position_zero_based": np.tile([1, 2, 3], 15),
        "from_aa": np.tile(["A", "C", "D"], 15),
        "to_aa": np.tile(["G", "H", "I"], 15),
        "length": 8,
    })
    x = rng.normal(size=(n, 8))
    pgen = x[:, 0] - 0.5 * x[:, 1]
    tcremp = x @ rng.normal(size=(8, 12))
    masks = {name: split == name for name in ("train", "val", "test")}
    metrics, predictions = evaluate_representation("toy", x, frame, pgen, tcremp, masks)
    assert metrics["pgen"]["r2"] > 0.9
    assert metrics["tcremp"]["mean_delta_cosine"] > 0.9
    assert np.isfinite(predictions["tcremp"]).all()
    assert identity_features(frame, False).shape == (n, 20)
    assert identity_features(frame, True).shape == (n, 100)


def test_ridge_preserves_multioutput_shape():
    x = np.arange(40, dtype=float).reshape(10, 4)
    y = np.column_stack((x[:, 0], x[:, 1] - x[:, 2]))
    pred = ridge_predict(ridge_fit(x, y, 1.0), x)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
