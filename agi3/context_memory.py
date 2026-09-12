"""Bounded state-conditioned click feedback, independent of the game runtime.

An observed frame change is evidence of an effect, not evidence of progress.
No-op feedback changes sampling weights only; it never forbids an action.
"""
from collections import OrderedDict


class TransitionMemory:
    def __init__(self, max_states=128, max_actions=256, max_controls=64):
        self.max_states = max_states
        self.max_actions = max_actions
        self.max_controls = max_controls
        self.states = OrderedDict()
        self.controls = OrderedDict()
        self.observations = 0
        self.effects = 0
        self.context_reuses = 0

    def clear(self):
        self.states.clear()
        self.controls.clear()

    def observe(self, before, action, after):
        """Record only the source context: a no-op elsewhere says nothing here."""
        if before in self.states:
            self.context_reuses += 1
            self.states.move_to_end(before)
        else:
            self.states[before] = OrderedDict()
        while len(self.states) > self.max_states:
            self.states.popitem(last=False)
        actions = self.states[before]
        counts = actions.setdefault(action, [0, 0])
        actions.move_to_end(action)
        changed = before != after
        counts[0] += int(changed)
        counts[1] += 1
        while len(actions) > self.max_actions:
            actions.popitem(last=False)
        if changed:
            self.controls[action] = None
            self.controls.move_to_end(action)
            while len(self.controls) > self.max_controls:
                self.controls.popitem(last=False)
        self.observations += 1
        self.effects += int(changed)

    def weight(self, state, action):
        changed, attempts = self.states.get(state, {}).get(action, (0, 0))
        return (changed + 1.0) / (attempts + 2.0)

    def choose(self, state, visible, rng, enabled):
        """One RNG draw in either arm; the caller fixes the probe budget.

        Both arms require at least one visible target. With memory enabled,
        append coordinates previously observed to have an effect, even if they
        are not current centroids. All targets retain a positive weight.
        """
        if not visible:
            return None
        pool = list(dict.fromkeys(visible))
        if enabled:
            pool = list(dict.fromkeys([*pool, *self.controls]))
        weights = [self.weight(state, target) if enabled else 1.0 for target in pool]
        cursor = rng.random() * sum(weights)
        for target, weight in zip(pool, weights):
            cursor -= weight
            if cursor < 0:
                return target
        return pool[-1]

    def summary(self):
        return {"observations": self.observations, "effects": self.effects,
                "context_reuses": self.context_reuses, "states": len(self.states),
                "controls": len(self.controls)}
