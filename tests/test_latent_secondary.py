import numpy as np
import pandas as pd

from benchmark.evaluate_latent_secondary import (
    deterministic_substitutions,
    joint_relevance,
    make_candidate_gallery,
    nested_subsets,
    project_tcremp,
    preregistration,
    retrieval_metrics,
    ridge_coefficients,
    ridge_predict,
)


def test_preregistration_covers_tracks_and_does_not_assume_synthetic_source() -> None:
    report = preregistration()
    assert set(report["models"]) == {"r", "t", "p", "rt", "rp", "tp", "rtp"}
    assert set(report["tracks"]) == {
        "low_label", "strata", "joint_retrieval", "robustness", "synthetic_discrimination"
    }
    assert report["tracks"]["synthetic_discrimination"]["status"].startswith("disabled")
    assert "No track" in report["no_selection_rule"]


def test_nested_subsets_are_deterministic_and_strictly_nested() -> None:
    indices = np.arange(1000)
    first = nested_subsets(indices, (0.1, 0.25, 1.0), 42)
    second = nested_subsets(indices, (0.1, 0.25, 1.0), 42)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert set(first[0.1]).issubset(first[0.25])
    assert set(first[0.25]).issubset(first[1.0])


def test_ridge_fit_uses_only_selected_rows() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 3))
    y = x @ np.array([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    train = np.arange(30)
    fit = ridge_coefficients(x, y, train, 0.1)
    prediction = ridge_predict(fit, x, np.arange(30, 40))
    assert np.mean((prediction - y[30:]) ** 2) < 0.01


def test_tcremp_projection_is_deterministic_and_finite() -> None:
    class Standardizer:
        tcremp_mean = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        tcremp_std = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)

    values = np.arange(18, dtype=np.float32).reshape(6, 3)
    first, report = project_tcremp(values, Standardizer(), dimension=2, seed=42, chunk_size=2)
    second, _ = project_tcremp(values, Standardizer(), dimension=2, seed=42, chunk_size=3)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert report["source_dimension"] == 3


def test_joint_gallery_excludes_query_and_retrieval_is_perfect_when_ranked() -> None:
    gallery = make_candidate_gallery(20, 5, 42)
    assert gallery.shape == (20, 5)
    assert all(query not in gallery[query] for query in range(20))
    relevant = np.zeros_like(gallery, dtype=bool)
    relevant[:, 0] = True
    distance = np.ones_like(gallery, dtype=float)
    distance[:, 0] = 0
    metrics = retrieval_metrics(distance, relevant)
    assert metrics["precision_at_1"] == 1.0
    assert metrics["map"] == 1.0


def test_joint_relevance_and_substitutions_are_valid_and_no_indels() -> None:
    sequences = pd.Series(["CASSLGQETQYF", "CASSLGQETQFF", "CASSPGQETQYF", "CARDRGNEQFF"])
    tcremp = np.asarray([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]], dtype=np.float32)
    pgen = np.asarray([0.0, 0.1, 0.2, 2.0])
    gallery = np.asarray([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int32)
    relevant = joint_relevance(sequences, tcremp, pgen, gallery, 0.5)
    assert relevant.shape == gallery.shape
    conservative, nonconservative = deterministic_substitutions(sequences.iloc[0], 42)
    assert len(conservative) == len(nonconservative) == len(sequences.iloc[0])
    assert sum(a != b for a, b in zip(conservative, sequences.iloc[0])) == 1
    assert sum(a != b for a, b in zip(nonconservative, sequences.iloc[0])) == 1
