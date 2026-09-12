"""Runtime policy checks for the isolated context-memory candidate pair."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "data/ARC-AGI-3-Kaggle-Starter/vendor/ARC-AGI-3-Agents"))
from arcengine import GameAction, GameState
import context_agent
import context_control_agent


def agent(module):
    return module.MyAgent(card_id="test", game_id="example", agent_name="test",
                          ROOT_URL="http://localhost", record=False, arc_env=None)


class AgentTests(unittest.TestCase):
    def test_both_arms_preserve_full_scan_despite_repeating_context_probes(self):
        frame = SimpleNamespace(frame=[[[0] * 64 for _ in range(64)]])
        for module in (context_agent, context_control_agent):
            policy = agent(module)
            policy.context_memory.observe(0, (1, 1), 1)
            with patch.object(module, "_components", return_value=[(1, 1)]):
                clicks = [policy._next_click(frame) for _ in range(4552)]
            self.assertEqual(set(clicks), {(x, y) for y in range(64) for x in range(64)})
            self.assertEqual(policy.sweep_pos, 4096)
            self.assertEqual(policy.context_probes, 456)

    def test_counter_mask_switch_cannot_manufacture_action_effect(self):
        for module in (context_agent, context_control_agent):
            policy = agent(module)
            before = [[0] * 64 for _ in range(64)]
            after = [row[:] for row in before]
            before[0][0], after[0][0] = 1, 2
            policy.prev_grid = before
            policy.prev_sig = policy._signature(before)
            policy.prev_action = GameAction.ACTION6
            policy.pending_click = (5, 5)
            policy.steps_compared = 23
            policy.row_changes[0] = 23
            policy.context_memory.observe("stale", (6, 6), "other")
            policy._observe(SimpleNamespace(frame=[after], levels_completed=0))
            self.assertEqual(policy.noise_rows, frozenset({0}))
            self.assertEqual(policy.effect[GameAction.ACTION6.value], [0, 1])
            self.assertEqual(len(policy.context_memory.states), 0)
            self.assertEqual(len(policy.context_memory.controls), 0)

    def test_action_effect_belongs_to_source_context(self):
        policy = agent(context_agent)
        before = [[0] * 64 for _ in range(64)]
        after = [row[:] for row in before]
        after[3][4] = 2
        policy.prev_grid = before
        policy.prev_action = GameAction.ACTION6
        policy.pending_click = (5, 5)
        policy.steps_compared = 25
        policy._observe(SimpleNamespace(frame=[after], levels_completed=0))
        self.assertEqual(policy.context_memory.weight(policy._signature(before), (5, 5)), 2 / 3)
        self.assertEqual(policy.context_memory.weight(policy._signature(after), (5, 5)), 1 / 2)

    def test_new_level_discards_old_context_and_control_transfer(self):
        policy = agent(context_agent)
        policy.context_memory.observe(1, (5, 5), 2)
        policy._observe(SimpleNamespace(frame=[[[0] * 64 for _ in range(64)]], levels_completed=1))
        self.assertEqual(len(policy.context_memory.states), 0)
        self.assertEqual(len(policy.context_memory.controls), 0)

    def test_reset_clears_memory(self):
        policy = agent(context_agent)
        policy.context_memory.observe(1, (5, 5), 2)
        action = policy.choose_action([], SimpleNamespace(state=GameState.GAME_OVER))
        self.assertIs(action, GameAction.RESET)
        self.assertEqual(len(policy.context_memory.states), 0)
        self.assertEqual(len(policy.context_memory.controls), 0)


if __name__ == "__main__":
    unittest.main()
