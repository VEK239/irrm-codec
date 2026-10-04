import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.prepare_vdjdb_ylq_motif_pooled import (
    build_pairs,
    read_joined_records,
    read_official_members,
    select_compact_negatives,
)


class OfficialMotifPooledTests(unittest.TestCase):
    def test_official_members_require_exact_context_and_chain(self) -> None:
        fields = [
            "species", "antigen.epitope", "mhc.a", "mhc.b", "mhc.class", "gene",
            "cdr3aa", "cid", "csz", "v.segm", "j.segm",
        ]
        rows = [
            ["HomoSapiens", "YLQPRTFLL", "HLA-A*02:01", "B2M", "MHCI", "TRB",
             "CASST", "H.B.YLQPRTFLL.1", "5", "TRBV1", "TRBJ1"],
            ["HomoSapiens", "YLQPRTFLL", "HLA-A*02", "B2M", "MHCI", "TRB",
             "CASSA", "wrong_mhc", "5", "TRBV1", "TRBJ1"],
            ["HomoSapiens", "YLQPRTFLL", "HLA-A*02:01", "B2M", "MHCI", "TRA",
             "CAVAA", "wrong_chain", "5", "TRAV1", "TRAJ1"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster_members.txt"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(fields)
                writer.writerows(rows)
            membership, report = read_official_members(path)
        self.assertEqual(membership, {("CASST", "YLQPRTFLL"): {"H.B.YLQPRTFLL.1"}})
        self.assertEqual(report["counts"]["exact_context_trb_rows"], 1)

    def test_join_pools_donors_and_removes_ambiguous_cdr3(self) -> None:
        fields = [
            "gene", "cdr3", "v.segm", "j.segm", "species", "mhc.a", "mhc.b",
            "mhc.class", "antigen.epitope", "reference.id", "vdjdb.score", "meta", "cdr3fix",
        ]

        def row(cdr3: str, epitope: str, donor: str) -> list[str]:
            return [
                "TRB", cdr3, "TRBV1", "TRBJ1", "HomoSapiens", "HLA-A*02:01", "B2M",
                "MHCI", epitope, "PMID:1", "1",
                json.dumps({"study.id": "s", "subject.id": donor}),
                json.dumps({"good": True, "cdr3": cdr3}),
            ]

        rows = [
            row("CASST", "YLQPRTFLL", "d1"),
            row("CASST", "YLQPRTFLL", "d2"),
            row("CASSA", "YLQPRTFLL", "d3"),
            row("CASSA", "OTHER", "d3"),
            row("CASSG", "OTHER", "d4"),
        ]
        membership = {
            ("CASST", "YLQPRTFLL"): {"ylq"},
            ("CASSA", "YLQPRTFLL"): {"ylq"},
            ("CASSA", "OTHER"): {"other"},
            ("CASSG", "OTHER"): {"other"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vdjdb.txt"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(fields)
                writer.writerows(rows)
            records, report = read_joined_records(
                path, membership, {"train": set(), "val": set(), "test": set()}, 1.0
            )
        self.assertEqual({record["cdr3"] for record in records}, {"CASST", "CASSG"})
        pooled = next(record for record in records if record["cdr3"] == "CASST")
        self.assertEqual(pooled["source_rows"], 2)
        self.assertEqual(pooled["donor_count_descriptive_only"], 2)
        self.assertEqual(report["filter_counts"]["ambiguous_multilabel_cdr3"], 1)

    def test_pair_builder_has_no_donor_dependency_and_exact_length_negative(self) -> None:
        def record(cdr3: str) -> dict:
            return {"cdr3": cdr3, "length": len(cdr3), "v_call": "TRBV1", "j_call": "TRBJ1"}

        positives = [record("CASS"), record("CASR"), record("CAST")]
        negatives = [record("CAVG"), record("CATG")]
        pairs, report = build_pairs(positives, negatives, seed=42)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(report["donor_constraint"], "none_pooled_by_user_protocol")
        for pair in pairs:
            self.assertEqual(pair["positive_length"], pair["negative_length"])
            self.assertEqual(len({pair["query_cdr3"], pair["positive_cdr3"], pair["negative_cdr3"]}), 3)

    def test_compact_controls_are_unique_capped_and_same_length(self) -> None:
        def record(cdr3: str, v: str = "TRBV1", j: str = "TRBJ1") -> dict:
            return {"cdr3": cdr3, "length": len(cdr3), "v_call": v, "j_call": j}

        positives = [record("CASS"), record("CASR"), record("CASST")]
        negatives = [
            record("CAVG"), record("CATG"), record("CAGG"), record("CAAG"),
            record("CGGG"), record("CTTG"), record("CAVGT"),
        ]
        selected, assignments, strata, report = select_compact_negatives(
            positives, negatives, per_positive=2, seed=42
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(len({row["cdr3"] for row in selected}), 5)
        self.assertLessEqual(len(selected), 2 * len(positives))
        self.assertTrue(all(len(row["positive_cdr3"]) == len(row["control_cdr3"])
                            for row in assignments))
        length_five = next(row for row in strata if row["length"] == 5)
        self.assertEqual(length_five["shortfall"], 1)
        self.assertEqual(report["gallery_control_reuse"], 0)

    def test_pairs_orient_away_from_unsupported_control_length(self) -> None:
        def record(cdr3: str) -> dict:
            return {"cdr3": cdr3, "length": len(cdr3), "v_call": "TRBV1", "j_call": "TRBJ1"}

        positives = [record("CASS"), record("CASRT"), record("CASST")]
        negatives = [record("CAVG")]
        pairs, report = build_pairs(positives, negatives, seed=42)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["positive_length"] == 4 for pair in pairs))
        self.assertEqual(report["excluded_pairs_both_endpoints_lack_exact_length_controls"], 1)
        self.assertEqual(report["forced_orientation_to_supported_comparator"], 2)


if __name__ == "__main__":
    unittest.main()
