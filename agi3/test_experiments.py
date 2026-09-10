"""Regression checks for experimental policy feedback and reproducible seeds."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "data/ARC-AGI-3-Kaggle-Starter/vendor/ARC-AGI-3-Agents"))
from arcengine import GameAction
import object_agent
import state_agent


def agent(module, game_id="example"):
    return module.MyAgent(card_id="test", game_id=game_id, agent_name="test",
                          ROOT_URL="http://localhost", record=False, arc_env=None)


class PolicyTests(unittest.TestCase):
    def test_counter_mask_change_is_not_click_feedback(self):
        policy = agent(state_agent)
        before = [[0] * 64 for _ in range(64)]
        after = [row[:] for row in before]
        before[0][0], after[0][0] = 1, 2
        policy.prev_grid = before
        policy.prev_sig = policy._signature(before)
        policy.prev_action = GameAction.ACTION6
        policy.pending_click = (5, 5)
        policy.steps_compared = 23
        policy.row_changes[0] = 23
        policy._observe(SimpleNamespace(frame=[after], levels_completed=0))
        self.assertEqual(policy.noise_rows, frozenset({0}))
        self.assertNotIn((5, 5), policy.responsive)
        self.assertEqual(policy.effect[GameAction.ACTION6.value], [0, 1])

    def test_short_and_versioned_game_ids_have_same_seed(self):
        short = agent(object_agent, "example")
        versioned = agent(object_agent, "example-01234567")
        self.assertEqual([short.rng.getrandbits(32) for _ in range(20)],
                         [versioned.rng.getrandbits(32) for _ in range(20)])

    def test_object_can_be_clicked_again_after_no_effect(self):
        policy = agent(object_agent)
        grid = [[0] * 64 for _ in range(64)]
        grid[7][9] = 2
        policy.dead.add((9, 7))
        policy.rng = SimpleNamespace(random=lambda: 0.0, choice=lambda items: items[0])
        frame = SimpleNamespace(frame=[grid])
        self.assertEqual(policy._next_click(frame), (9, 7))
        self.assertEqual(policy._next_click(frame), (9, 7))


if __name__ == "__main__":
    unittest.main()
