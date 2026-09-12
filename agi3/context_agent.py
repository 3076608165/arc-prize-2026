"""Generated context-memory prototype; unproven. Rebuild with prepare_context_agents.py. The pair differs only in USE_CONTEXT_MEMORY."""
from __future__ import annotations
import hashlib
import os
import random
from collections import defaultdict
from typing import Any
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent
BOARD = 64
MAX_COMPONENT_CELLS = 220
MAX_CLICK_TARGETS = 128
NOISE_WARMUP = 24
NOISE_ROW_FRACTION = 0.85
EXPLORE_RATE = 1.0
BY_VALUE = {a.value: a for a in GameAction}

def _layer(frame: list[list[list[int]]]) -> list[list[int]]:
    """The visible layer. Every game seen so far ships exactly one."""
    return frame[-1] if frame else []

def _components(grid: list[list[int]]) -> list[tuple[int, int]]:
    """(x, y) centroids of connected same-color regions, rare colors first.

    Background (the most common color) is skipped, as are regions too big to be
    a game piece. 4-connectivity, iterative flood fill.
    """
    if not grid:
        return []
    h, w = (len(grid), len(grid[0]))
    counts: dict[int, int] = defaultdict(int)
    for row in grid:
        for v in row:
            counts[v] += 1
    bg = max(counts, key=lambda c: counts[c])
    seen = [[False] * w for _ in range(h)]
    found: list[tuple[int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or grid[r0][c0] == bg:
                continue
            color = grid[r0][c0]
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = (r + dr, c + dc)
                    if 0 <= nr < h and 0 <= nc < w and (not seen[nr][nc]) and (grid[nr][nc] == color):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(cells) > MAX_COMPONENT_CELLS:
                continue
            cy = sum((r for r, _ in cells)) // len(cells)
            cx = sum((c for _, c in cells)) // len(cells)
            if (cy, cx) not in set(cells):
                cy, cx = min(cells, key=lambda rc: (rc[0] - cy) ** 2 + (rc[1] - cx) ** 2)
            found.append((cx, cy, len(cells), color))
    freq: dict[int, int] = defaultdict(int)
    for _, _, _, col in found:
        freq[col] += 1
    found.sort(key=lambda t: (freq[t[3]], t[2]))
    return [(x, y) for x, y, _, _ in found[:MAX_CLICK_TARGETS]]

def _sweep_order(board: int=BOARD) -> list[tuple[int, int]]:
    """Coarse-to-fine cover of the board: stride 8, then 4, then 2, then all.

    Coarse first so that a responsive region is found early even if the run is
    cut short, rather than after scanning the top-left quadrant pixel by pixel.
    """
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for step in (8, 4, 2, 1):
        for y in range(0, board, step):
            for x in range(0, board, step):
                if (x, y) not in seen:
                    seen.add((x, y))
                    out.append((x, y))
    return out
SWEEP = _sweep_order()

class CoverageBase(Agent):
    """Systematic, counter-aware exploration over the legal action set."""
    MAX_ACTIONS = 5000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed_base = int(os.environ.get('ARC_SEED', '0'))
        game_key = self.game_id.split('-')[0]
        game_seed = int.from_bytes(hashlib.sha256(game_key.encode()).digest()[:4], 'big')
        self.rng = random.Random(2620 ^ game_seed ^ seed_base * 40503)
        self.row_changes: list[int] = [0] * BOARD
        self.steps_compared = 0
        self.noise_rows: frozenset[int] = frozenset()
        self.prev_grid: list[list[int]] | None = None
        self.tried_simple: dict[int, set[int]] = defaultdict(set)
        self.sweep_pos = 0
        self.click_calls = 0
        self.responsive: list[tuple[int, int]] = []
        self.dead: set[tuple[int, int]] = set()
        self.pending_click: tuple[int, int] | None = None
        self.exploit_pos = -1
        self.effect: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        self.prev_sig: int | None = None
        self.prev_action: GameAction | None = None
        self.best_levels = 0
        self.seen_sigs: set[int] = set()

    @property
    def name(self) -> str:
        return f'{super().name}.sweep'

    def _learn_noise(self, grid: list[list[int]]) -> None:
        if self.prev_grid is not None and len(self.prev_grid) == len(grid):
            self.steps_compared += 1
            for r, (a, b) in enumerate(zip(self.prev_grid, grid)):
                if a != b and r < BOARD:
                    self.row_changes[r] += 1
            if self.steps_compared >= NOISE_WARMUP:
                cutoff = self.steps_compared * NOISE_ROW_FRACTION
                self.noise_rows = frozenset((r for r, c in enumerate(self.row_changes) if c >= cutoff))
        self.prev_grid = grid

    def _signature(self, grid: list[list[int]]) -> int:
        """Hash of the board with the detected counter rows masked out."""
        if not self.noise_rows:
            return hash(tuple((tuple(row) for row in grid)))
        return hash(tuple((tuple(row) for r, row in enumerate(grid) if r not in self.noise_rows)))

    def _observe(self, latest_frame: FrameData) -> int:
        grid = _layer(latest_frame.frame)
        previous_grid = self.prev_grid
        previous_mask = self.noise_rows
        self._learn_noise(grid)
        sig = self._signature(grid)
        if previous_grid is not None:
            self.prev_sig = self._signature(previous_grid)
        if previous_mask != self.noise_rows:
            self.tried_simple.clear()
            self.effect.clear()
            self.responsive.clear()
            self.dead.clear()
        if self.prev_action is not None and self.prev_sig is not None:
            changed = sig != self.prev_sig
            st = self.effect[self.prev_action.value]
            st[1] += 1
            st[0] += int(changed)
            if self.pending_click is not None:
                if self.noise_rows or self.steps_compared >= NOISE_WARMUP:
                    if changed:
                        if self.pending_click not in self.responsive:
                            self.responsive.append(self.pending_click)
                    else:
                        self.dead.add(self.pending_click)
                self.pending_click = None
        if latest_frame.levels_completed > self.best_levels:
            self.best_levels = latest_frame.levels_completed
            self.tried_simple.clear()
            self.dead.clear()
            self.sweep_pos = 0
        self.seen_sigs.add(sig)
        return sig

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_sig = None
            self.prev_action = None
            self.pending_click = None
            self.tried_simple.clear()
            return GameAction.RESET
        sig = self._observe(latest_frame)
        legal = [BY_VALUE[v] for v in latest_frame.available_actions or list(BY_VALUE) if v in BY_VALUE and v != GameAction.RESET.value]
        if not legal:
            return GameAction.RESET
        simple = [a for a in legal if not a.is_complex()]
        complex_ = [a for a in legal if a.is_complex()]
        action = self._pick(sig, simple, complex_, latest_frame)
        self.prev_sig = sig
        self.prev_action = action
        return action

    def _pick(self, sig: int, simple: list[GameAction], complex_: list[GameAction], latest_frame: FrameData) -> GameAction:
        untried = [a for a in simple if a.value not in self.tried_simple[sig]]
        if untried:
            action = self.rng.choice(untried)
            self.tried_simple[sig].add(action.value)
            action.reasoning = 'untried simple action in this masked state'
            return action
        if complex_:
            target = self._next_click(latest_frame)
            if target is not None:
                x, y = target
                action = complex_[0]
                action.set_data({'x': int(x), 'y': int(y)})
                action.reasoning = {'why': 'sweep' if not self.responsive else 'known-responsive cell', 'x': int(x), 'y': int(y)}
                self.pending_click = (x, y)
                return action
        pool = simple or complex_
        action = self._sample_by_effect(pool)
        if action.is_complex():
            x, y = (self.rng.randrange(BOARD), self.rng.randrange(BOARD))
            action.set_data({'x': x, 'y': y})
            action.reasoning = {'why': 'exhausted, random probe', 'x': x, 'y': y}
            self.pending_click = None
        else:
            action.reasoning = 'exhausted, sampled by effect rate'
        return action

    def _sample_by_effect(self, pool: list[GameAction]) -> GameAction:
        weights = [self._effect_rate(a) for a in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def _next_click(self, latest_frame: FrameData) -> tuple[int, int] | None:
        """One object probe per ten clicks; all other clicks advance the sweep.

        No location is permanently excluded. In a stable level with only clicks,
        all 4096 pixels are visited within 4552 clicks, even when object probes
        keep choosing the same location. Level transitions restart the sweep.
        """
        self.click_calls += 1
        if self.click_calls % 10 == 1:
            targets = _components(_layer(latest_frame.frame))
            if targets:
                return self.rng.choice(targets)
        target = SWEEP[self.sweep_pos % len(SWEEP)]
        self.sweep_pos += 1
        return target

    def _effect_rate(self, action: GameAction) -> float:
        hits, tries = self.effect[action.value]
        return (hits + 1.0) / (tries + 1.0)

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


USE_CONTEXT_MEMORY = True


class MyAgent(CoverageBase):
    """Experimental context memory within the fixed 10% object-probe budget."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.context_memory = TransitionMemory()
        self.context = None
        self.context_probes = 0

    def _observe(self, latest_frame: FrameData) -> int:
        before_grid = self.prev_grid
        before_mask = self.noise_rows
        before_click = self.pending_click
        before_action = self.prev_action
        before_level = self.best_levels
        sig = super()._observe(latest_frame)
        mask_changed = before_mask != self.noise_rows
        level_changed = before_level != self.best_levels
        if mask_changed or level_changed:
            self.context_memory.clear()
        if (before_grid is not None and before_click is not None
                and before_action is not None and not mask_changed
                and not level_changed and self.steps_compared >= NOISE_WARMUP):
            self.context_memory.observe(self._signature(before_grid), before_click, sig)
        self.context = sig
        return sig

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.context_memory.clear()
            self.context = None
        return super().choose_action(frames, latest_frame)

    def _next_click(self, latest_frame: FrameData) -> tuple[int, int] | None:
        self.click_calls += 1
        if self.click_calls % 10 == 1:
            visible = _components(_layer(latest_frame.frame))
            target = self.context_memory.choose(
                self.context, visible, self.rng, enabled=USE_CONTEXT_MEMORY)
            if target is not None:
                self.context_probes += 1
                return target
        target = SWEEP[self.sweep_pos % len(SWEEP)]
        self.sweep_pos += 1
        return target
