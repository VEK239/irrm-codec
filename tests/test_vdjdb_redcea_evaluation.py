import numpy as np
import pandas as pd

from benchmark.evaluate_vdjdb_redcea_clonotypes import (
    build_candidate_gallery,
    candidate_distances,
    summarize_candidates,
)


def test_perfect_epitope_geometry_scores_perfect_retrieval() -> None:
    rows = []
    features = []
    for label_id, label in enumerate(("E1", "E2", "E3")):
        for index in range(4):
            rows.append({
                "label": label, "species": "HomoSapiens", "length": 10,
                "v_call": "TRBV1", "j_call": "TRBJ1", "cdr3": f"C{label}{index}F",
            })
            vector = np.zeros(3, dtype=np.float32)
            vector[label_id] = 10.0
            vector += index * 1e-3
            features.append(vector)
    cohort = pd.DataFrame(rows)
    candidates, relevant, _, report = build_candidate_gallery(cohort, 2, 1, 42)
    distances, _ = candidate_distances(np.asarray(features), candidates, "cosine")
    names = report["label_names"]
    label_ids = np.array([names.index(value) for value in cohort["label"]], dtype=np.int32)
    summary, per_query, _ = summarize_candidates(
        distances, candidates, relevant, label_ids, names
    )
    assert summary["macro_epitope"]["precision_at_1"] == 1.0
    assert summary["macro_epitope"]["precision_at_5"] > 0.0
    assert summary["macro_epitope"]["within_between_ratio"] < 0.01
    assert summary["knn_classification"]["k_1"]["macro_f1"] == 1.0
    assert per_query["nearest_same_is_closer"].all()
