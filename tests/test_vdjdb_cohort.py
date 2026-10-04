import json
import unittest

from benchmark.prepare_vdjdb_epitope_cohort import normalize_vdjdb_row


def row(**changes):
    value = {
        "gene": "TRB",
        "cdr3": "CASSLGQETQYF",
        "species": "HomoSapiens",
        "mhc.a": "HLA-A*02:01",
        "mhc.b": "B2M",
        "mhc.class": "MHCI",
        "antigen.epitope": "GILGFVFTL",
        "reference.id": "PMID:1",
        "vdjdb.score": "2",
        "meta": json.dumps({"subject.id": "donor1", "study.id": "study1"}),
        "cdr3fix": json.dumps({"good": True, "cdr3": "CASSLGQETQYF"}),
    }
    value.update(changes)
    return value


class NormalizeVDJdbRowTests(unittest.TestCase):
    def test_accepts_unmodified_human_trb_and_builds_scoped_donor(self):
        normalized, reason = normalize_vdjdb_row(row(), min_score=1)
        self.assertEqual(reason, "accepted_row")
        self.assertEqual(normalized["donor_key"], "study1|donor1")
        self.assertEqual(
            normalized["label"], "GILGFVFTL|HLA-A*02:01|B2M|MHCI"
        )

    def test_rejects_sequence_changing_fix(self):
        normalized, reason = normalize_vdjdb_row(
            row(cdr3fix=json.dumps({"good": True, "cdr3": "CASSIRSSYEQYF"})),
            min_score=1,
        )
        self.assertIsNone(normalized)
        self.assertEqual(reason, "cdr3fix_changes_sequence")

    def test_rejects_missing_donor_and_low_score(self):
        normalized, reason = normalize_vdjdb_row(
            row(meta="{}"), min_score=1
        )
        self.assertIsNone(normalized)
        self.assertEqual(reason, "missing_donor")
        normalized, reason = normalize_vdjdb_row(row(**{"vdjdb.score": "0"}), min_score=1)
        self.assertIsNone(normalized)
        self.assertEqual(reason, "low_or_invalid_score")


if __name__ == "__main__":
    unittest.main()
