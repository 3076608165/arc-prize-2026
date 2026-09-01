"""Small pure-Python grid primitives shared by the solvers."""
from __future__ import annotations

from collections import Counter

from .loader import Grid


def shape(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0]) if g else 0


def transpose(g: Grid) -> Grid:
    return [list(r) for r in zip(*g)]


def fliplr(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def flipud(g: Grid) -> Grid:
    return [list(row) for row in g[::-1]]


def rot90(g: Grid) -> Grid:          # counter-clockwise
    return [list(r) for r in zip(*g)][::-1]


def rot180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]


def rot270(g: Grid) -> Grid:
    return [list(r) for r in zip(*g[::-1])]


def anti_transpose(g: Grid) -> Grid:
    return rot180(transpose(g))


def identity(g: Grid) -> Grid:
    return [list(row) for row in g]


#: The dihedral group of the square, as named transforms.
D4 = {
    "identity": identity,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "fliplr": fliplr,
    "flipud": flipud,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
}


def colors(g: Grid) -> Counter:
    return Counter(v for row in g for v in row)


def background(g: Grid) -> int:
    """Most frequent color -- the usual stand-in for 'empty'."""
    c = colors(g)
    return c.most_common(1)[0][0] if c else 0


def bbox(g: Grid, bg: int) -> tuple[int, int, int, int] | None:
    """(r0, c0, r1, c1) inclusive bounds of the non-``bg`` cells."""
    rs = [r for r, row in enumerate(g) for v in row if v != bg]
    cs = [c for row in g for c, v in enumerate(row) if v != bg]
    if not rs:
        return None
    return min(rs), min(cs), max(rs), max(cs)


def crop(g: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    return [row[c0 : c1 + 1] for row in g[r0 : r1 + 1]]


def upscale(g: Grid, kh: int, kw: int) -> Grid:
    return [[v for v in row for _ in range(kw)] for row in g for _ in range(kh)]


def tile(g: Grid, m: int, n: int) -> Grid:
    return [row * n for _ in range(m) for row in g]


def replace(g: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(v, v) for v in row] for row in g]


def equal(a: Grid, b: Grid) -> bool:
    return a == b
