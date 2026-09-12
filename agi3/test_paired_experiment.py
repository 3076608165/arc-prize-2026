import unittest
from unittest.mock import patch
from types import SimpleNamespace

from test_experiments import agent
import coverage_agent
import stable_agent
import paired_experiment as experiment
import run


class ExperimentTests(unittest.TestCase):
    def test_both_arms_start_with_identical_random_streams(self):
        a = agent(stable_agent, "example-version")
        b = agent(coverage_agent, "example-version")
        self.assertEqual([a.rng.random() for _ in range(50)], [b.rng.random() for _ in range(50)])

    def test_full_pixel_coverage_with_worst_case_repeated_object_probes(self):
        policy = agent(coverage_agent)
        frame = SimpleNamespace(frame=[[[0]*64 for _ in range(64)]])
        with patch.object(coverage_agent, "_components", return_value=[(1, 1)]):
            clicks = [policy._next_click(frame) for _ in range(4552)]
        self.assertEqual(set(clicks), {(x, y) for x in range(64) for y in range(64)})
        self.assertEqual(policy.sweep_pos, 4096)

    def test_missing_duplicate_and_nonfinite_results_fail(self):
        for rows in ([{"id": "a-v1", "score": 1}],
                     [{"id": "a-v1", "score": 1}, {"id": "a-v1", "score": 1}],
                     [{"id": "a-v1", "score": float("nan")}, {"id": "b-v1", "score": 1}]):
            with self.assertRaises(ValueError):
                run.validate_results({"a": {}, "b": {}}, rows, ["a", "b"])

    def test_paired_statistics_handle_constant_effect_and_null(self):
        positive = experiment.paired_statistics([1]*12, [2]*12, draws=1000)
        self.assertEqual(positive["paired_bootstrap_ci95"], [1, 1])
        self.assertEqual(positive["sign_flip_p_two_sided"], 2/4096)
        self.assertEqual(positive["positive_pairs"], 12)
        null = experiment.paired_statistics([1]*12, [1]*12, draws=1000)
        self.assertEqual(null["paired_bootstrap_ci95"], [0, 0])
        self.assertEqual(null["sign_flip_p_two_sided"], 1)


if __name__ == "__main__":
    unittest.main()
