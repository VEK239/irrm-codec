import numpy as np
import pandas as pd

from benchmark.prepare_single_aa_challenge import choose_parents, generate_mutants, parent_folds
from benchmark.validate_single_aa_targets import align_targets


def test_mutants_are_unique_one_substitution_and_parent_split_is_disjoint():
    parents = pd.DataFrame({
        "row_index": [10, 20, 30, 40, 50],
        "junction_aa": ["CASSF", "CASRF", "CATQF", "CARGF", "CSAWF"],
        "v_call": ["TRBV1"] * 5,
        "j_call": ["TRBJ1-1"] * 5,
    })
    benchmark = set(parents["junction_aa"])
    mutants = generate_mutants(parents, benchmark, variants_per_parent=3, seed=42)
    assert len(mutants) == 15
    assert mutants["junction_aa"].nunique() == len(mutants)
    assert not (set(mutants["junction_aa"]) & benchmark)
    for row in mutants.itertuples(index=False):
        assert len(row.parent_sequence) == len(row.junction_aa)
        assert sum(a != b for a, b in zip(row.parent_sequence, row.junction_aa)) == 1
        assert row.mutation_position_zero_based not in (0, len(row.junction_aa) - 1)
    folds = parent_folds(mutants["parent_id"].unique(), seed=42)
    mutants["fold"] = mutants["parent_id"].map(folds)
    by_fold = {fold: set(group["parent_id"]) for fold, group in mutants.groupby("fold")}
    assert by_fold["train"].isdisjoint(by_fold["val"])
    assert by_fold["train"].isdisjoint(by_fold["test"])
    assert by_fold["val"].isdisjoint(by_fold["test"])


def test_length_stratified_parent_limit_is_exact_even_with_more_strata_than_limit():
    table = pd.DataFrame({
        "row_index": np.arange(20),
        "junction_aa": ["C" + "A" * length + "F" for length in range(2, 22)],
    })
    chosen = choose_parents(table, np.arange(20), limit=5, seed=42)
    assert len(chosen) == 5
    assert chosen["row_index"].nunique() == 5


def test_validator_accepts_exact_zero_but_requires_finite_1mm_target(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    mutants = pd.DataFrame({
        "clone_id": ["a", "b"], "junction_aa": ["CASSF", "CASRF"],
        "parent_id": ["p1", "p2"], "parent_sequence": ["CASAF", "CASSF"],
    })
    mutants.to_csv(prepared / "mutants.tsv", sep="\t", index=False)
    pgen = mutants[["clone_id", "junction_aa"]].copy()
    pgen["log10_pgen"] = [-np.inf, -5.0]
    pgen["log10_pgen_1mm"] = [-4.0, -4.5]
    pgen_path = tmp_path / "pgen.tsv"
    pgen.to_csv(pgen_path, sep="\t", index=False)
    representations = pd.DataFrame({
        "clone_id": ["a", "b"], "cdr3aa_TRB": ["CASSF", "CASRF"],
    })
    reps_path = tmp_path / "representations.tsv"
    representations.to_csv(reps_path, sep="\t", index=False)
    tcremp = pd.DataFrame(np.zeros((2, 9000), dtype=np.float32))
    tcremp.insert(0, "clone_id", ["a", "b"])
    tcremp_path = tmp_path / "tcremp.parquet"
    tcremp.to_parquet(tcremp_path, index=False)
    report = align_targets(prepared, pgen_path, reps_path, tcremp_path, tmp_path / "targets")
    assert report["finite_pgen_target"] is True
    assert report["nonfinite_exact_log10_pgen_diagnostic_rows"] == 1
    accepted = pd.read_csv(tmp_path / "targets" / "targets.tsv", sep="\t")
    assert "log10_pgen" not in accepted
    assert np.isfinite(accepted["log10_pgen_1mm"]).all()
