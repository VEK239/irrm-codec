import tempfile
from pathlib import Path

import pandas as pd

from benchmark.evaluate_vdjdb_viral_motif_top10 import (
    EXPECTED_WEIGHTS,
    MODEL_NAMES,
    summarize_distribution,
)
from benchmark.prepare_vdjdb_viral_motif_top10 import (
    choose_top10,
    rank_epitopes,
    read_species_list,
)


def test_species_allowlist_is_exact_and_rejects_duplicates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "viral.txt"
        path.write_text("# pinned exact values\nInfluenzaA\nSARS-CoV-2\n", encoding="utf-8")
        assert read_species_list(path) == ["InfluenzaA", "SARS-CoV-2"]
        path.write_text("InfluenzaA\nInfluenzaA\n", encoding="utf-8")
        try:
            read_species_list(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate species values must be rejected.")


def test_top10_is_ranked_by_support_and_relaxation_is_explicit() -> None:
    records = []
    for index, count in enumerate(range(50, 38, -1)):
        for row in range(count):
            records.append({
                "cdr3": f"C{index:02d}{row:03d}",
                "epitope": f"PEP{index:02d}",
                "antigen_species": "VirusExact",
            })
    records.append({"cdr3": "OTHER", "epitope": "PEPX", "antigen_species": "Tumor"})
    ranking, inventory = rank_epitopes(records, {"VirusExact"})
    selected, gate = choose_top10(ranking, requested_threshold=45)
    assert [row["epitope"] for row in selected] == [f"PEP{i:02d}" for i in range(10)]
    assert gate["threshold_transparently_lowered"] is True
    assert gate["effective_minimum_unique_cdr3"] == 41
    assert next(row for row in inventory if row["antigen_species"] == "Tumor") \
        ["classified_viral_by_exact_allowlist"] == 0


def test_epitope_count_shortfall_excludes_singleton_targets() -> None:
    ranking = [
        {"epitope": f"E{index}", "unique_cdr3": count}
        for index, count in enumerate((50, 46, 40, 5, 4, 2, 1, 1))
    ]
    selected, gate = choose_top10(ranking, requested_threshold=40)
    assert len(selected) == 6
    assert gate["selected_epitopes"] == 6
    assert gate["evaluable_epitope_shortfall"] == 4
    assert gate["effective_minimum_unique_cdr3"] == 2
    assert gate["viral_epitopes_below_distinct_pair_minimum"] == 2


def test_full_factorial_model_labels_and_distribution_summary() -> None:
    assert tuple(EXPECTED_WEIGHTS) == MODEL_NAMES
    rows = []
    for epitope_index in range(10):
        for model_index, model in enumerate(MODEL_NAMES):
            rows.append({
                "epitope": f"E{epitope_index}",
                "distance": "cosine",
                "metric": "same_label_auroc",
                "model": model,
                "value": 0.5 + model_index / 100,
            })
    summary, wins, win_summary = summarize_distribution(pd.DataFrame(rows))
    assert len(summary) == 7
    assert set(wins["model"]) == {"rtp"}
    assert win_summary.loc[win_summary["model"] == "rtp", "win_count"].item() == 10
