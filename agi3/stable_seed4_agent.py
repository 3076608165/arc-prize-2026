"""ARC-AGI-3 agent: legal actions only, counter-masked state, systematic click sweep.

Source of truth lives here; `agi3/run.py` copies it into the starter kit.

Four measured facts about the environment drive every choice below. Each was
established with `agi3/probe.py` against the real engine, and the numbers are
reproducible.

1. **`available_actions` varies per game and the starter ignores it.** Of the 24
   games, 6 accept ACTION6 (click) and nothing else; ls20 accepts only
   ACTION1-4. Sampling uniformly from all seven throws away up to 86% of the
   budget. Filtering to the reported legal set is the single cheapest win.

2. **Click games have ~2-11 responsive cells out of 4096.** The engine's own
   internal `_get_valid_actions()` -- which is deliberately never exposed to
   agents -- reports 2 clickable cells for vc33, 2 for r11l, 2 for lp85, 11 for
   tn36. A uniform random click lands on one with probability ~0.05%. Random
   clicking cannot work here; only systematic coverage can.

3. **Part of the frame is a step counter, and it moves every single step.**
   On vc33 it is row 0; on ls20 it is rows 61-62. Hashing the raw frame
   therefore makes almost every state look novel: over 250 random steps vc33
   visits 51 "distinct" full-frame states but exactly **1** once row 0 is
   masked out. Any novelty- or frame-delta-driven policy that skips this
   masking is chasing a counter. We detect the noisy rows online rather than
   hardcoding them, because their position differs per game.

4. **Single-step frame deltas do not separate valid from invalid clicks.**
   Clicking a known-valid cell and a known-invalid cell on vc33 produce
   byte-identical deltas (one cell, in the counter row). So "did the frame
   change" is only a usable reward signal *after* masking, not before.

The resulting policy: mask the counter, only ever emit legal actions, sweep the
board systematically on click games while recording which cells actually move
the masked state, and fall back to novelty-seeking over the masked signature
for games driven by simple actions.
"""
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

# A row is treated as a counter once we have enough evidence and it has changed
# on at least this fraction of steps.
NOISE_WARMUP = 24
NOISE_ROW_FRACTION = 0.85

# Fraction of clicks still spent sweeping once some controls are known.
#
# 1.0 means "always keep sweeping" -- the exploit phase is OFF. It looked like
# it should help: scoring squares per-level action count, so sweeping after the
# controls are found is paid for twice. Measured over 5 seeds it did the
# opposite, scoring 0.1356 against 0.1812 for plain sweeping while clearing
# *more* levels (7.6 vs 6.0). Clearing more and scoring less is the whole
# tension in this benchmark.
#
# At 2 seeds it looked 34% better. That was noise. See agi3/exploit_agent.py
# for the variant and what to try next.
EXPLORE_RATE = 1.0

# `GameAction.value` is 1-7 but the by-value constructor rejects those ints,
# so build the reverse map by iterating the members.
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
    h, w = len(grid), len(grid[0])
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
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(cells) > MAX_COMPONENT_CELLS:
                continue
            cy = sum(r for r, _ in cells) // len(cells)
            cx = sum(c for _, c in cells) // len(cells)
            if (cy, cx) not in set(cells):
                cy, cx = min(cells, key=lambda rc: (rc[0] - cy) ** 2 + (rc[1] - cx) ** 2)
            found.append((cx, cy, len(cells), color))

    freq: dict[int, int] = defaultdict(int)
    for _, _, _, col in found:
        freq[col] += 1
    found.sort(key=lambda t: (freq[t[3]], t[2]))
    return [(x, y) for x, y, _, _ in found[:MAX_CLICK_TARGETS]]


def _sweep_order(board: int = BOARD) -> list[tuple[int, int]]:
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


class MyAgent(Agent):
    """Systematic, counter-aware exploration over the legal action set."""

    MAX_ACTIONS = 5000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ARC_SEED lets `run.py --seeds` replay the whole sweep under different
        # randomisation. Without it a single A/B is one sample of a noisy
        # measurement: the random baseline alone has scored 4 levels and 2
        # levels on identical settings.
        seed_base = int(os.environ.get("ARC_SEED", "4"))
        game_key = self.game_id.split("-")[0]
        game_seed = int.from_bytes(hashlib.sha256(game_key.encode()).digest()[:4], "big")
        self.rng = random.Random(0xA3C ^ game_seed ^ (seed_base * 0x9E37))

        # counter detection
        self.row_changes: list[int] = [0] * BOARD
        self.steps_compared = 0
        self.noise_rows: frozenset[int] = frozenset()
        self.prev_grid: list[list[int]] | None = None

        # exploration memory
        self.tried_simple: dict[int, set[int]] = defaultdict(set)
        self.sweep_pos = 0
        self.responsive: list[tuple[int, int]] = []   # cells that moved the masked state
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
        return f"{super().name}.sweep"

    # -- state signature --------------------------------------------------

    def _learn_noise(self, grid: list[list[int]]) -> None:
        if self.prev_grid is not None and len(self.prev_grid) == len(grid):
            self.steps_compared += 1
            for r, (a, b) in enumerate(zip(self.prev_grid, grid)):
                if a != b and r < BOARD:
                    self.row_changes[r] += 1
            if self.steps_compared >= NOISE_WARMUP:
                cutoff = self.steps_compared * NOISE_ROW_FRACTION
                self.noise_rows = frozenset(
                    r for r, c in enumerate(self.row_changes) if c >= cutoff
                )
        self.prev_grid = grid

    def _signature(self, grid: list[list[int]]) -> int:
        """Hash of the board with the detected counter rows masked out."""
        if not self.noise_rows:
            return hash(tuple(tuple(row) for row in grid))
        return hash(
            tuple(tuple(row) for r, row in enumerate(grid) if r not in self.noise_rows)
        )

    # -- feedback ---------------------------------------------------------

    def _observe(self, latest_frame: FrameData) -> int:
        grid = _layer(latest_frame.frame)
        self._learn_noise(grid)
        sig = self._signature(grid)

        if self.prev_action is not None and self.prev_sig is not None:
            changed = sig != self.prev_sig
            st = self.effect[self.prev_action.value]
            st[1] += 1
            st[0] += int(changed)
            if self.pending_click is not None:
                # Only trust this verdict once the counter rows are known;
                # before that a "change" may just be the counter ticking.
                if self.noise_rows or self.steps_compared >= NOISE_WARMUP:
                    if changed:
                        if self.pending_click not in self.responsive:
                            self.responsive.append(self.pending_click)
                    else:
                        self.dead.add(self.pending_click)
                self.pending_click = None

        if latest_frame.levels_completed > self.best_levels:
            # New level: board semantics changed, so per-state memory is stale.
            # Keep `responsive` -- the control layout usually survives -- but
            # re-open the sweep in case new things became clickable.
            self.best_levels = latest_frame.levels_completed
            self.tried_simple.clear()
            self.dead.clear()
            self.sweep_pos = 0

        self.seen_sigs.add(sig)
        return sig

    # -- policy -----------------------------------------------------------

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_sig = None
            self.prev_action = None
            self.pending_click = None
            self.tried_simple.clear()
            return GameAction.RESET

        sig = self._observe(latest_frame)

        legal = [
            BY_VALUE[v]
            for v in (latest_frame.available_actions or list(BY_VALUE))
            if v in BY_VALUE and v != GameAction.RESET.value
        ]
        if not legal:
            return GameAction.RESET

        simple = [a for a in legal if not a.is_complex()]
        complex_ = [a for a in legal if a.is_complex()]
        action = self._pick(sig, simple, complex_, latest_frame)

        self.prev_sig = sig
        self.prev_action = action
        return action

    def _pick(
        self,
        sig: int,
        simple: list[GameAction],
        complex_: list[GameAction],
        latest_frame: FrameData,
    ) -> GameAction:
        untried = [a for a in simple if a.value not in self.tried_simple[sig]]
        if untried:
            # Uniform, deliberately -- not argmax by effect rate.
            #
            # The size of the action space decides whether coverage or sampling
            # wins, and these games span both regimes. Over 4096 click cells,
            # of which 2 do anything, chance never finds them and only a sweep
            # works. Over four to seven buttons, chance tries everything within
            # a few steps, and imposing an order on top only removes diversity:
            # ranking by a statistic gathered in *other* states commits early
            # to whatever happened to move the frame first.
            #
            # Measured: the argmax version cleared 3 games to the random
            # agent's 4, losing exactly the four that expose 5+ actions.
            action = self.rng.choice(untried)
            self.tried_simple[sig].add(action.value)
            action.reasoning = "untried simple action in this masked state"
            return action

        if complex_:
            target = self._next_click(latest_frame)
            if target is not None:
                x, y = target
                action = complex_[0]
                action.set_data({"x": int(x), "y": int(y)})
                action.reasoning = {
                    "why": "sweep" if not self.responsive else "known-responsive cell",
                    "x": int(x),
                    "y": int(y),
                }
                self.pending_click = (x, y)
                return action

        # Everything here has been tried. Sample in proportion to how often each
        # action has moved the masked state, rather than replaying the single
        # best one -- an argmax here is an absorbing state, and a game whose
        # progress needs a button that happens to rank second never sees it
        # again.
        pool = simple or complex_
        action = self._sample_by_effect(pool)
        if action.is_complex():
            x, y = self.rng.randrange(BOARD), self.rng.randrange(BOARD)
            action.set_data({"x": x, "y": y})
            action.reasoning = {"why": "exhausted, random probe", "x": x, "y": y}
            self.pending_click = None
        else:
            action.reasoning = "exhausted, sampled by effect rate"
        return action

    def _sample_by_effect(self, pool: list[GameAction]) -> GameAction:
        weights = [self._effect_rate(a) for a in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def _next_click(self, latest_frame: FrameData) -> tuple[int, int] | None:
        """Exploit known controls first; sweep for new ones only in the gaps.

        The scoring formula makes this the difference between a level being
        worth something and worth nothing. A level scores
        ``(human_baseline_actions / actions_on_that_level) ** 2``, so on a game
        whose first level a human clears in 7 actions, taking 3000 returns
        essentially zero -- while taking 17 collects the entire structural cap.

        The previous version returned every untried cell before it would reuse
        one it had already seen work, so it finished discovering the controls
        and then kept sweeping the remaining few thousand cells anyway. It was
        paying full exploration price for a level it had already learned how to
        play.

        Once a cell is known to move the masked state, replay it. Keep a small
        exploration rate so a level that needs a *second* control can still
        find it.
        """
        if self.responsive and self.rng.random() > EXPLORE_RATE:
            # Cycle rather than sample, so several known controls all get used
            # instead of the RNG favouring one.
            self.exploit_pos = (self.exploit_pos + 1) % len(self.responsive)
            return self.responsive[self.exploit_pos]

        for pt in _components(_layer(latest_frame.frame)):
            if pt not in self.dead and pt not in self.responsive:
                return pt

        while self.sweep_pos < len(SWEEP):
            pt = SWEEP[self.sweep_pos]
            self.sweep_pos += 1
            if pt not in self.dead and pt not in self.responsive:
                return pt

        # Board fully swept. Re-use whatever actually did something.
        if self.responsive:
            self.exploit_pos = (self.exploit_pos + 1) % len(self.responsive)
            return self.responsive[self.exploit_pos]
        return None

    def _effect_rate(self, action: GameAction) -> float:
        hits, tries = self.effect[action.value]
        return (hits + 1.0) / (tries + 1.0)   # optimistic for untried
