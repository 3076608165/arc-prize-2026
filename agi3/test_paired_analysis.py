"""Pure-stdlib tests: incomplete or contaminated rounds must not pass the gate."""
import copy
import unittest

from paired_experiment import analyze


def fixtures():
    protocol = {"seeds": list(range(12)), "arms": {"a": {"sha256": "a"}, "b": {"sha256": "b"}},
                "control": "a", "candidate": "b", "games": ["g"], "action_budget": 5000,
                "bootstrap_draws": 1000, "analysis_seed": 1, "minimum_absolute_gain": .02,
                "minimum_relative_gain": .15, "minimum_positive_pairs": 8, "alpha": .05,
                "limitations": ["synthetic test only"]}
    records = [{"seed": seed, "arm": arm, "source_sha256": arm,
                "result": {"exact_budget": True, "games": {"g": {"actions": 5000}},
                           "rows": [{"id": "g-v1", "score": value}]}}
               for seed in range(12) for arm, value in (("a", 1), ("b", 2))]
    return protocol, records


class AnalysisTests(unittest.TestCase):
    def test_complete_positive_round_is_eligible_only_for_experiment(self):
        protocol, records = fixtures()
        self.assertEqual(analyze(protocol, records)["status"], "eligible_for_online_experiment")

    def test_missing_pair_never_advances(self):
        protocol, records = fixtures()
        self.assertEqual(analyze(protocol, records[:-1])["status"], "collecting")

    def test_contamination_fails_even_before_round_is_complete(self):
        for field in ("hash", "budget", "game", "score"):
            protocol, records = fixtures()
            row = copy.deepcopy(records[0])
            if field == "hash":
                row["source_sha256"] = "changed"
            elif field == "budget":
                row["result"]["games"]["g"]["actions"] = 5001
            elif field == "game":
                row["result"]["rows"][0]["id"] = "other-v1"
            else:
                row["result"]["rows"][0]["score"] = None
            with self.assertRaises(ValueError):
                analyze(protocol, [row])

    def test_game_version_change_fails(self):
        protocol, records = fixtures()
        records[-1]["result"]["rows"][0]["id"] = "g-v2"
        with self.assertRaises(ValueError):
            analyze(protocol, records)

    def test_small_consistent_gain_below_practical_bar_is_inconclusive(self):
        protocol, records = fixtures()
        for row in records:
            if row["arm"] == "b":
                row["result"]["rows"][0]["score"] = 1.01
        self.assertEqual(analyze(protocol, records)["status"], "inconclusive_do_not_promote")


if __name__ == "__main__":
    unittest.main()
