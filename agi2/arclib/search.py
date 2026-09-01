"""Breadth-first composition search over the DSL, plus two same-shape learners.

Three things run here, in increasing specificity:

``compose_search``
    BFS over compositions of `dsl.PRIMITIVES`. Nodes are deduplicated by what
    they *compute* on the demonstration inputs, not by their spelling, so
    ``fliplr . fliplr`` collapses into the identity node instead of doubling the
    frontier. That is what keeps depth 3 affordable in pure Python.

``learn_color_map_after``
    The search finds the right *shape* far more often than the right *colors*.
    Whenever a program lands on grids of the correct shape, we try to close the
    gap with a single global recolor learned from the demos.

``learn_neighborhood_rule``
    81 of the 120 evaluation tasks preserve grid shape. For that class a
    cell-local rule -- output color as a function of the 3x3 input
    neighbourhood -- is a genuinely different hypothesis from any global
    transform, and it is learnable in closed form: read the mapping off the
    demos and check it is a function.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

from .dsl import PRIMITIVES
from .grids import shape
from .loader import Grid, Pair

Program = tuple[str, ...]
MAX_NODES_PER_DEPTH = 600


def _key(grids: list[Grid]) -> int:
    return hash(tuple(tuple(tuple(r) for r in g) for g in grids))


def _apply(name_fn, grids: list[Grid]) -> list[Grid] | None:
    _, fn = name_fn
    out = []
    for g in grids:
        try:
            r = fn(g)
        except Exception:
            return None
        if not r or not r[0] or len(r) > 30 or len(r[0]) > 30:
            return None
        out.append(r)
    return out


def compose_search(
    demos: list[Pair], tests: list[Grid], max_depth: int = 3
) -> list[tuple[Program, list[Grid]]]:
    """Return every program (up to ``max_depth``) that reproduces all demos.

    The result is ordered shortest-program-first, which is the prior we want:
    among hypotheses that explain the data equally well, take the simplest.
    """
    ins = [p.input for p in demos]
    outs = [p.output for p in demos]
    n_demo = len(ins)

    # A node carries the program and its value on (demo inputs + test inputs),
    # so a hit can read the test predictions straight off without re-running.
    start = ins + tests
    frontier: list[tuple[Program, list[Grid]]] = [((), start)]
    seen: set[int] = {_key(start)}
    hits: list[tuple[Program, list[Grid]]] = []

    if start[:n_demo] == outs:
        hits.append(((), start[n_demo:]))

    for _depth in range(max_depth):
        nxt: list[tuple[Program, list[Grid]]] = []
        for prog, grids in frontier:
            for prim in PRIMITIVES:
                res = _apply(prim, grids)
                if res is None:
                    continue
                k = _key(res)
                if k in seen:
                    continue
                seen.add(k)
                p2 = prog + (prim[0],)
                if res[:n_demo] == outs:
                    hits.append((p2, res[n_demo:]))
                    continue          # no point extending a solved program
                nxt.append((p2, res))
                if len(nxt) >= MAX_NODES_PER_DEPTH:
                    break
            if len(nxt) >= MAX_NODES_PER_DEPTH:
                break
        frontier = nxt
        if not frontier:
            break
    return hits


def learn_color_map_after(
    demos: list[Pair], tests: list[Grid], max_depth: int = 2
) -> list[tuple[Program, list[Grid]]]:
    """Programs that reach the right shape, finished off by a global recolor."""
    ins = [p.input for p in demos]
    outs = [p.output for p in demos]
    n = len(ins)

    start = ins + tests
    frontier: list[tuple[Program, list[Grid]]] = [((), start)]
    seen: set[int] = {_key(start)}
    hits: list[tuple[Program, list[Grid]]] = []

    def try_recolor(prog: Program, grids: list[Grid]) -> None:
        mapping: dict[int, int] = {}
        for got, want in zip(grids[:n], outs):
            if shape(got) != shape(want):
                return
            for ra, rb in zip(got, want):
                for a, b in zip(ra, rb):
                    if mapping.setdefault(a, b) != b:
                        return
        if not mapping or all(k == v for k, v in mapping.items()):
            return          # the plain program already covers this
        preds = [[[mapping.get(v, v) for v in row] for row in g] for g in grids[n:]]
        hits.append((prog + ("recolor",), preds))

    try_recolor((), start)
    for _depth in range(max_depth):
        nxt: list[tuple[Program, list[Grid]]] = []
        for prog, grids in frontier:
            for prim in PRIMITIVES:
                res = _apply(prim, grids)
                if res is None:
                    continue
                k = _key(res)
                if k in seen:
                    continue
                seen.add(k)
                p2 = prog + (prim[0],)
                try_recolor(p2, res)
                nxt.append((p2, res))
                if len(nxt) >= MAX_NODES_PER_DEPTH:
                    break
            if len(nxt) >= MAX_NODES_PER_DEPTH:
                break
        frontier = nxt
        if not frontier:
            break
    return hits


def _neighborhood(g: Grid, r: int, c: int, oob: int = -1) -> tuple[int, ...]:
    h, w = shape(g)
    return tuple(
        g[r + dr][c + dc] if 0 <= r + dr < h and 0 <= c + dc < w else oob
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    )


def learn_neighborhood_rule(
    demos: list[Pair], tests: list[Grid]
) -> list[tuple[Program, list[Grid]]]:
    """Output color as a deterministic function of the 3x3 input neighbourhood.

    Only defined when every demo preserves grid shape. Fails closed: if any
    neighbourhood in a test input was never seen in a demo, we do not guess --
    a partially-right grid scores exactly the same as no answer at all, so an
    unfounded prediction is pure noise in the ranking.
    """
    if not demos or any(shape(p.input) != shape(p.output) for p in demos):
        return []

    table: dict[tuple[int, ...], int] = {}
    for p in demos:
        h, w = shape(p.input)
        for r in range(h):
            for c in range(w):
                nb = _neighborhood(p.input, r, c)
                v = p.output[r][c]
                if table.setdefault(nb, v) != v:
                    return []          # not a function of the neighbourhood

    # A rule that only ever echoes the centre cell is the identity in disguise.
    if all(nb[4] == v for nb, v in table.items()):
        return []

    preds: list[Grid] = []
    for t in tests:
        h, w = shape(t)
        out = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                nb = _neighborhood(t, r, c)
                if nb not in table:
                    return []
                out[r][c] = table[nb]
        preds.append(out)
    return [(("neighborhood_rule",), preds)]


def learn_symmetry_repair(
    demos: list[Pair], tests: list[Grid]
) -> list[tuple[Program, list[Grid]]]:
    """Fill cells of one 'hole' color by copying from a symmetric position.

    A recurring ARC shape: a symmetric picture with a rectangle punched out.
    The demos tell us which color marks the hole; the symmetry that maps the
    intact part onto itself tells us what belongs there.
    """
    if not demos:
        return []

    # Identify the hole color: present in every input, absent from every output.
    in_colors = [set(v for row in p.input for v in row) for p in demos]
    out_colors = [set(v for row in p.output for v in row) for p in demos]
    cands = set.intersection(*in_colors) - set.union(*out_colors)
    same_shape = all(shape(p.input) == shape(p.output) for p in demos)
    if len(cands) != 1 or not same_shape:
        return []
    hole = cands.pop()

    def repair(g: Grid) -> Grid | None:
        h, w = shape(g)
        out = [list(row) for row in g]
        maps: list[Callable[[int, int], tuple[int, int]]] = [
            lambda r, c: (r, w - 1 - c),
            lambda r, c: (h - 1 - r, c),
            lambda r, c: (h - 1 - r, w - 1 - c),
        ]
        if h == w:
            maps += [lambda r, c: (c, r), lambda r, c: (w - 1 - c, h - 1 - r)]
        # Keep only the symmetries the intact part actually satisfies.
        good = []
        for m in maps:
            ok = True
            for r in range(h):
                for c in range(w):
                    rr, cc = m(r, c)
                    if g[r][c] != hole and g[rr][cc] != hole and g[r][c] != g[rr][cc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                good.append(m)
        if not good:
            return None
        for r in range(h):
            for c in range(w):
                if out[r][c] != hole:
                    continue
                for m in good:
                    rr, cc = m(r, c)
                    if g[rr][cc] != hole:
                        out[r][c] = g[rr][cc]
                        break
                else:
                    return None
        return out

    for p in demos:
        if repair(p.input) != p.output:
            return []
    preds = []
    for t in tests:
        r = repair(t)
        if r is None:
            return []
        preds.append(r)
    return [(("symmetry_repair",), preds)]
