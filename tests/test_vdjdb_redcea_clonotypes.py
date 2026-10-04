import pandas as pd

from benchmark.prepare_vdjdb_redcea_clonotypes import prepare_cohort


def member(sequence: str, label: str, cid: str, species: str = "HomoSapiens") -> dict:
    return {
        "species": species, "antigen.epitope": label, "antigen.gene": "g",
        "antigen.species": "virus", "mhc.a": "", "mhc.b": "", "mhc.class": "",
        "gene": "TRB", "cdr3aa": sequence, "cid": cid, "csz": "5",
        "v.segm": "TRBV1", "j.segm": "TRBJ1",
    }


def test_preparation_deduplicates_removes_ambiguity_overlap_and_applies_support() -> None:
    rows = [
        member("CASSAAAFF", "E1", "E1.1"),
        member("CASSAAAFF", "E1", "E1.2"),
        member("CASSQQQFF", "E1", "E1.1"),
        member("CASSCCCFF", "E2", "E2.1"),
        member("CASSCCCFF", "E3", "E3.1"),
        member("CASSDDDFF", "E2", "E2.1"),
        member("CASSEEEXF", "E2", "E2.1"),  # invalid AA X
        member("CASSFFFYF", "E2", "E2.1"),
    ]
    benchmark = {"train": {"CASSDDDFF"}, "val": set(), "test": set()}
    cohort, summary, report = prepare_cohort(pd.DataFrame(rows), benchmark, 2, 40)
    assert cohort[["label", "cdr3"]].to_dict("records") == [
        {"label": "E1", "cdr3": "CASSAAAFF"},
        {"label": "E1", "cdr3": "CASSQQQFF"},
    ]
    assert summary.loc[0, "unique_cdr3"] == 2
    assert cohort.loc[0, "source_rows"] == 2
    assert cohort.loc[0, "source_cids"] == "E1.1|E1.2"
    assert report["filter_counts"]["ambiguous_multilabel_cdr3"] == 1
    assert report["filter_counts"]["benchmark_overlap_cdr3"] == 1
    assert report["filter_counts"]["invalid_sequence_rows"] == 1
