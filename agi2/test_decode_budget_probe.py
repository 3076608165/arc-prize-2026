"""Check information isolation and conservative stopping behavior."""
import unittest

from decode_budget_probe import ROOT, decide, load_selector


def eight_views(second=None):
    views = {}
    for rotation in range(4):
        for permutation in range(2):
            name = "t_0." + "rot90." * rotation + f"permute{permutation}.ex01"
            samples = [{"solution": [[1]], "beam_score": .1, "score_aug": [.1] * 8}]
            if second is not None:
                samples.append({"solution": [[second]], "beam_score": 2.0, "score_aug": [2.] * 8})
            views[name] = samples
    return views


class DecodeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = staticmethod(load_selector(ROOT / "agi2/out/v7/arc_decoder.py"))

    def test_unanimous_complete_views_stop(self):
        self.assertTrue(decide(eight_views(), self.selector)["stop"])
        self.assertTrue(decide(eight_views(second=2), self.selector)["stop"])

    def test_missing_or_empty_view_never_stops(self):
        views = eight_views()
        views.pop(next(iter(views)))
        self.assertFalse(decide(views, self.selector)["stop"])
        views = eight_views()
        views[next(iter(views))] = []
        self.assertFalse(decide(views, self.selector)["stop"])

    def test_one_dissenting_view_never_stops(self):
        views = eight_views()
        views[next(iter(views))][0]["solution"] = [[2]]
        self.assertEqual(decide(views, self.selector)["reason"], "view_winner_disagreement")

    def test_second_candidate_instability_never_stops(self):
        views = eight_views(second=2)
        for name, samples in views.items():
            if name.count("rot90") % 2:
                samples[1]["solution"] = [[3]]
        self.assertEqual(decide(views, self.selector)["reason"], "batch_top_two_disagreement")

    def test_future_views_are_rejected(self):
        views = eight_views()
        views["t_0.transpose.permute0.ex01"] = views[next(iter(views))]
        with self.assertRaisesRegex(ValueError, "Future"):
            decide(views, self.selector)


if __name__ == "__main__":
    unittest.main()
