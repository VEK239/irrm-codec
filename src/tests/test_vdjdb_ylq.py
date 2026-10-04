import unittest

from rtp_codec.benchmarks.datasets.prepare_vdjdb_ylq import build_pairs


def record(cdr3: str, donor: str, v_call: str = "TRBV1", j_call: str = "TRBJ1") -> dict:
    return {
        "cdr3": cdr3,
        "donor_key": donor,
        "length": len(cdr3),
        "v_call": v_call,
        "j_call": j_call,
    }


class YLQPairConstructionTests(unittest.TestCase):
    def test_pairs_are_exact_length_and_fully_donor_disjoint(self) -> None:
        positives = [
            record("CASS", "study|p1"),
            record("CASR", "study|p2"),
            record("CAST", "study|p3"),
        ]
        negatives = [
            record("CATG", "study|n1"),
            record("CAVG", "study|n2", v_call="TRBV2"),
        ]
        pairs, report = build_pairs(positives, negatives, seed=42)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(report["exact_length_fraction"], 1.0)
        for pair in pairs:
            self.assertEqual(pair["positive_length"], pair["negative_length"])
            self.assertEqual(
                len({pair["query_donor"], pair["positive_donor"], pair["negative_donor"]}),
                3,
            )

    def test_missing_exact_length_negative_is_blocking(self) -> None:
        positives = [record("CASS", "p1"), record("CASR", "p2")]
        negatives = [record("CASSL", "n1")]
        with self.assertRaisesRegex(ValueError, "No exact-length donor-disjoint negative"):
            build_pairs(positives, negatives, seed=42)


if __name__ == "__main__":
    unittest.main()
