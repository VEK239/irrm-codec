import numpy as np
import pandas as pd

from rtp_codec.benchmarks.downstream.evaluate_vdjdb_epitope_prediction import (
    audit_dataset,
    make_splits,
    redcea_component_groups,
    score_predictions,
)


def test_dataset_audit_detects_duplicates_and_label_conflicts():
    frame = pd.DataFrame({"cdr3": ["CASSF", "CASSF", "CASRF"], "label": ["A", "B", "B"]})
    report, sizes = audit_dataset(frame)
    assert report["sequences"] == 3
    assert report["epitopes"] == 2
    assert report["duplicate_cdr3_rows"] == 2
    assert report["identical_cdr3_with_different_labels"] == 1
    assert sizes.set_index("label").loc["B", "sequences"] == 2


def test_redcea_components_merge_rows_sharing_any_source_cluster():
    frame = pd.DataFrame({
        "source_cids": ["a|b", "b|c", "d", "e"],
        "cid": ["a", "b", "d", "e"],
    })
    groups = redcea_component_groups(frame)
    assert groups[0] == groups[1]
    assert len(np.unique(groups)) == 3


def test_grouped_splits_have_no_group_leakage_and_shared_exact_alias():
    y = np.repeat(np.arange(2), 12)
    exact = np.arange(len(y))
    similarity = np.repeat(np.arange(8), 3)
    protocols, report = make_splits(y, exact, similarity, [42], 3, True)
    assert protocols["sequence_stratified"] == protocols["exact_cdr3_grouped"]
    assert report["exact_cdr3_grouped"]["alias_of"] == "sequence_stratified"
    for _, _, train, test in protocols["redcea_similarity_grouped"]:
        assert not set(similarity[train]).intersection(similarity[test])


def test_multiclass_metrics_are_perfect_for_perfect_probabilities():
    y = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[y]
    result = score_predictions(y, probabilities, np.arange(3))
    assert all(value == 1.0 for value in result.values())
