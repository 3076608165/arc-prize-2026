"""Stdlib mechanism checks; these do not establish a score improvement."""
import ast
import random
import unittest
from types import SimpleNamespace

from context_memory import TransitionMemory
from prepare_context_agents import make_source


class MemoryTests(unittest.TestCase):
    def test_noop_evidence_is_context_local_and_never_forbids(self):
        memory = TransitionMemory()
        first, second = (1, 1), (2, 2)
        for _ in range(10):
            memory.observe("before", first, "before")
        fixed = SimpleNamespace(random=lambda: 0.4)
        self.assertEqual(memory.choose("before", [first, second], fixed, True), second)
        self.assertEqual(memory.choose("changed", [first, second], fixed, True), first)
        self.assertGreater(memory.weight("before", first), 0)

    def test_previously_effective_control_can_be_recalled_without_a_centroid(self):
        memory = TransitionMemory()
        visible, control = (1, 1), (17, 39)
        memory.observe("before", control, "after")
        fixed = SimpleNamespace(random=lambda: 0.99)
        self.assertEqual(memory.choose("new", [visible], fixed, True), control)
        self.assertEqual(memory.choose("new", [visible], fixed, False), visible)

    def test_empty_observation_cannot_create_a_probe_slot_in_either_arm(self):
        memory = TransitionMemory()
        memory.observe(0, (17, 39), 1)
        for enabled in (False, True):
            self.assertIsNone(memory.choose(1, [], random.Random(0), enabled))

    def test_all_memory_dimensions_are_bounded(self):
        memory = TransitionMemory(max_states=2, max_actions=2, max_controls=2)
        for state in range(8):
            for action in range(7):
                memory.observe(state, (action, state), state + 1)
        self.assertEqual(len(memory.states), 2)
        self.assertEqual(len(memory.controls), 2)
        self.assertTrue(all(len(actions) <= 2 for actions in memory.states.values()))

    def test_uninformed_pair_uses_the_same_random_stream(self):
        a, b = random.Random(17), random.Random(17)
        memory = TransitionMemory()
        visible = [(2, 3), (5, 7), (13, 19)]
        self.assertEqual([memory.choose(0, visible, a, False) for _ in range(40)],
                         [memory.choose(0, visible, b, True) for _ in range(40)])
        self.assertEqual(a.random(), b.random())

    def test_generated_pair_differs_in_one_boolean_only(self):
        candidate = make_source(True)
        control = make_source(False)
        self.assertEqual(candidate.replace("USE_CONTEXT_MEMORY = True", "USE_CONTEXT_MEMORY = False"), control)
        ast.parse(candidate)
        self.assertNotIn("Measured over 5 seeds", candidate)


if __name__ == "__main__":
    unittest.main()
