import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.prepare_trb_joint import (
    add_identity_columns,
    make_splits,
    prepare_identity_table,
    write_standardizer,
)


class JointTRBPreparationTest(unittest.TestCase):
    def sources(self):
        airr = pd.DataFrame(
            {
                "junction_aa": ["CASSF", "CASRF", "CAT", "CAA"],
                "v_call": ["TRBV1", "TRBV2", "TRBV3", "TRBV4"],
                "j_call": ["TRBJ1", "TRBJ2", "TRBJ1", "TRBJ2"],
                "locus": ["beta"] * 4,
            }
        )
        pgen = airr.copy()
        pgen["log10_pgen"] = [-5.0, -6.0, -7.0, -8.0]
        pgen["log10_pgen_1mm"] = [-4.0, -5.0, -6.0, -7.0]
        representations = pd.DataFrame(
            {
                "clone_id": [12, 10, 13, 11],
                "cdr3aa_TRB": ["CAT", "CASSF", "CAA", "CASRF"],
                "v_TRB": ["TRBV3*01", "TRBV1*01", "TRBV4*01", "TRBV2*01"],
                "j_TRB": ["TRBJ1*01", "TRBJ1*01", "TRBJ2*01", "TRBJ2*01"],
            }
        )
        return airr, pgen, representations

    def test_identity_join_does_not_depend_on_row_order(self):
        airr, pgen, representations = self.sources()
        table, report = prepare_identity_table(
            airr,
            pgen.sample(frac=1.0, random_state=2),
            representations,
            min_len=1,
            max_len=40,
        )
        self.assertEqual(report["rows_kept"], 4)
        observed = dict(zip(table["junction_aa"], table["clone_id"]))
        self.assertEqual(observed, {"CASSF": 10, "CASRF": 11, "CAT": 12, "CAA": 13})
        self.assertTrue(table["record_id"].is_unique)

    def test_non_finite_targets_are_removed_before_fixed_split(self):
        airr, pgen, representations = self.sources()
        pgen.loc[1, "log10_pgen_1mm"] = -np.inf
        table, report = prepare_identity_table(
            airr, pgen, representations, min_len=1, max_len=40
        )
        self.assertEqual(report["dropped_non_finite_pgen"], 1)
        splits, overlaps = make_splits(
            table, train_fraction=0.5, val_fraction=0.25, seed=42
        )
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 1, "val": 0, "test": 2})
        self.assertEqual(overlaps, {"train_val": 0, "train_test": 0, "val_test": 0})

    def test_standardizer_uses_only_train_rows(self):
        airr, pgen, representations = self.sources()
        table, _ = prepare_identity_table(
            airr, pgen, representations, min_len=1, max_len=40
        )
        embeddings = np.arange(20, dtype=np.float32).reshape(4, 5)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "target_standardizer.npz"
            summary = write_standardizer(
                output,
                table,
                embeddings,
                np.array([0, 2]),
                chunk_size=1,
            )
            with np.load(output, allow_pickle=False) as standardizer:
                np.testing.assert_allclose(
                    standardizer["tcremp_mean"], embeddings[[0, 2]].mean(axis=0)
                )
                self.assertEqual(int(standardizer["train_rows"]), 2)
                self.assertEqual(str(standardizer["pgen_target"]), "log10_pgen_1mm")
            self.assertEqual(summary["fit_split"], "train")

    def test_identity_normalizes_gene_alleles(self):
        table = add_identity_columns(
            pd.DataFrame(
                {"cdr3": [" cassf "], "v": ["TRBV7-6*01"], "j": ["TRBJ2-7*01"]}
            ),
            cdr3_col="cdr3",
            v_col="v",
            j_col="j",
        )
        self.assertEqual(table.loc[0, "junction_aa"], "CASSF")
        self.assertEqual(table.loc[0, "v_identity"], "TRBV7-6")
        self.assertEqual(table.loc[0, "j_identity"], "TRBJ2-7")


if __name__ == "__main__":
    unittest.main()
