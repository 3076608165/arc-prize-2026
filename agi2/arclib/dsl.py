"""Parameterised grid primitives for compositional search.

Each entry in ``PRIMITIVES`` is ``(name, fn)`` where ``fn: Grid -> Grid`` may
raise to signal "not applicable here" -- the search treats a raising primitive
as a dead branch rather than a crash.

Primitives are deliberately total-ish and cheap. Anything that needs to look at
the demonstration pairs to choose its parameters (which color maps to which,
what the output size is) belongs in `solvers.py`, not here: the search composes
these blind, and only the final verification step sees the outputs.
"""
from __future__ import annotations

from collections import Counter

from .grids import (
    D4,
    background,
    bbox,
    crop,
    shape,
)
from .loader import Grid


def _nonempty(g: Grid) -> Grid:
    if not g or not g[0]:
        raise ValueError("empty grid")
    return g


# -- structure ------------------------------------------------------------


def crop_content_zero(g: Grid) -> Grid:
    b = bbox(g, 0)
    if b is None:
        raise ValueError("all background")
    return _nonempty(crop(g, *b))


def crop_content_modal(g: Grid) -> Grid:
    b = bbox(g, background(g))
    if b is None:
        raise ValueError("all background")
    return _nonempty(crop(g, *b))


def trim_border(g: Grid) -> Grid:
    h, w = shape(g)
    if h < 3 or w < 3:
        raise ValueError("too small to trim")
    return [row[1:-1] for row in g[1:-1]]


def _halves(g: Grid) -> dict[str, Grid]:
    h, w = shape(g)
    return {
        "top": g[: h // 2],
        "bottom": g[(h + 1) // 2 :],
        "left": [row[: w // 2] for row in g],
        "right": [row[(w + 1) // 2 :] for row in g],
    }


def half_top(g: Grid) -> Grid:
    return _nonempty(_halves(g)["top"])


def half_bottom(g: Grid) -> Grid:
    return _nonempty(_halves(g)["bottom"])


def half_left(g: Grid) -> Grid:
    return _nonempty(_halves(g)["left"])


def half_right(g: Grid) -> Grid:
    return _nonempty(_halves(g)["right"])


def dedup_rows(g: Grid) -> Grid:
    """Collapse runs of identical adjacent rows to one."""
    out = [list(g[0])]
    for row in g[1:]:
        if row != out[-1]:
            out.append(list(row))
    if len(out) == len(g):
        raise ValueError("nothing to collapse")
    return out


def dedup_cols(g: Grid) -> Grid:
    t = [list(c) for c in zip(*g)]
    out = [t[0]]
    for col in t[1:]:
        if col != out[-1]:
            out.append(list(col))
    if len(out) == len(t):
        raise ValueError("nothing to collapse")
    return [list(r) for r in zip(*out)]


# -- movement -------------------------------------------------------------


def _gravity(g: Grid, bg: int, axis: str) -> Grid:
    if axis in ("down", "up"):
        cols = []
        for c in zip(*g):
            vals = [v for v in c if v != bg]
            pad = [bg] * (len(c) - len(vals))
            cols.append(pad + vals if axis == "down" else vals + pad)
        return [list(r) for r in zip(*cols)]
    rows = []
    for row in g:
        vals = [v for v in row if v != bg]
        pad = [bg] * (len(row) - len(vals))
        rows.append(pad + vals if axis == "right" else vals + pad)
    return rows


def gravity_down(g: Grid) -> Grid:
    return _gravity(g, background(g), "down")


def gravity_up(g: Grid) -> Grid:
    return _gravity(g, background(g), "up")


def gravity_left(g: Grid) -> Grid:
    return _gravity(g, background(g), "left")


def gravity_right(g: Grid) -> Grid:
    return _gravity(g, background(g), "right")


# -- objects --------------------------------------------------------------


def _labels(g: Grid, bg: int, diagonal: bool = False) -> list[list[tuple[int, int]]]:
    h, w = shape(g)
    seen = [[False] * w for _ in range(h)]
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or g[r0][c0] == bg:
                continue
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in steps:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] != bg:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _extract(g: Grid, cells: list[tuple[int, int]], bg: int) -> Grid:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    out = [[bg] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
    for r, c in cells:
        out[r - r0][c - c0] = g[r][c]
    return out


def largest_object(g: Grid) -> Grid:
    bg = background(g)
    comps = _labels(g, bg, diagonal=True)
    if not comps:
        raise ValueError("no objects")
    return _extract(g, max(comps, key=len), bg)


def smallest_object(g: Grid) -> Grid:
    bg = background(g)
    comps = _labels(g, bg, diagonal=True)
    if not comps:
        raise ValueError("no objects")
    return _extract(g, min(comps, key=len), bg)


def unique_object(g: Grid) -> Grid:
    """The object whose shape appears exactly once among the objects."""
    bg = background(g)
    comps = _labels(g, bg, diagonal=True)
    if len(comps) < 2:
        raise ValueError("need several objects")
    grids = [_extract(g, c, bg) for c in comps]
    keys = [repr(x) for x in grids]
    counts = Counter(keys)
    singles = [x for x, k in zip(grids, keys) if counts[k] == 1]
    if len(singles) != 1:
        raise ValueError("no single odd-one-out")
    return singles[0]


# -- color ----------------------------------------------------------------


def to_binary(g: Grid) -> Grid:
    """Background to 0, everything else to 1 -- strips color identity."""
    bg = background(g)
    return [[0 if v == bg else 1 for v in row] for row in g]


def most_common_color_only(g: Grid) -> Grid:
    """Keep only the most common non-background color."""
    bg = background(g)
    c = Counter(v for row in g for v in row if v != bg)
    if not c:
        raise ValueError("no foreground")
    keep = c.most_common(1)[0][0]
    return [[v if v == keep else bg for v in row] for row in g]


def least_common_color_only(g: Grid) -> Grid:
    bg = background(g)
    c = Counter(v for row in g for v in row if v != bg)
    if not c:
        raise ValueError("no foreground")
    keep = c.most_common()[-1][0]
    return [[v if v == keep else bg for v in row] for row in g]


# -- registry -------------------------------------------------------------

PRIMITIVES: list[tuple[str, object]] = [
    *[(f"d4.{n}", f) for n, f in D4.items() if n != "identity"],
    ("crop_content_zero", crop_content_zero),
    ("crop_content_modal", crop_content_modal),
    ("trim_border", trim_border),
    ("half_top", half_top),
    ("half_bottom", half_bottom),
    ("half_left", half_left),
    ("half_right", half_right),
    ("dedup_rows", dedup_rows),
    ("dedup_cols", dedup_cols),
    ("gravity_down", gravity_down),
    ("gravity_up", gravity_up),
    ("gravity_left", gravity_left),
    ("gravity_right", gravity_right),
    ("largest_object", largest_object),
    ("smallest_object", smallest_object),
    ("unique_object", unique_object),
    ("to_binary", to_binary),
    ("most_common_color_only", most_common_color_only),
    ("least_common_color_only", least_common_color_only),
]
