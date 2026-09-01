"""A baseline portfolio of induction rules for ARC-AGI-2.

Each solver takes the demonstration pairs and, if it can explain *every* one of
them, returns a callable that maps a test input to a predicted output. Solvers
that cannot explain the demos return ``None``. ``solve_task`` runs the whole
portfolio, keeps the predictions of the rules that fit, and returns them ranked
by the portfolio order (cheapest / most specific hypotheses first).

This is deliberately a floor, not a contender: it is the pipeline check that
tells you the harness, the submission format and the scorer all work. Real
progress on ARC-AGI-2 comes from what you put next to these.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable

from .grids import (
    D4,
    background,
    bbox,
    crop,
    replace,
    shape,
    tile,
    upscale,
)
from .loader import Grid, Pair

Transform = Callable[[Grid], Grid]
Solver = Callable[[list[Pair]], Transform | None]

_REGISTRY: list[tuple[str, Solver]] = []


def solver(name: str):
    def deco(fn: Solver) -> Solver:
        _REGISTRY.append((name, fn))
        return fn

    return deco


def _fits(fn: Transform, demos: list[Pair]) -> bool:
    try:
        return all(fn(p.input) == p.output for p in demos)
    except Exception:
        return False


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------


@solver("constant")
def constant(demos: list[Pair]) -> Transform | None:
    """Every demo has the same output, regardless of input."""
    outs = [p.output for p in demos]
    if len(demos) >= 2 and all(o == outs[0] for o in outs):
        out = outs[0]
        return lambda g: [list(r) for r in out]
    return None


@solver("d4")
def d4(demos: list[Pair]) -> Transform | None:
    """A single rotation/reflection of the square explains every demo."""
    for fn in D4.values():
        if _fits(fn, demos):
            return fn
    return None


@solver("color_map")
def color_map(demos: list[Pair]) -> Transform | None:
    """Same shape, and one global color substitution explains every demo."""
    mapping: dict[int, int] = {}
    for p in demos:
        if p.output is None or shape(p.input) != shape(p.output):
            return None
        for ri, ro in zip(p.input, p.output):
            for a, b in zip(ri, ro):
                if mapping.setdefault(a, b) != b:
                    return None
    return lambda g: replace(g, mapping)


@solver("d4_then_color_map")
def d4_then_color_map(demos: list[Pair]) -> Transform | None:
    """A reorientation followed by a global recolor."""
    for geo in D4.values():
        mapping: dict[int, int] = {}
        ok = True
        for p in demos:
            gi = geo(p.input)
            if shape(gi) != shape(p.output):
                ok = False
                break
            for ri, ro in zip(gi, p.output):
                for a, b in zip(ri, ro):
                    if mapping.setdefault(a, b) != b:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok and mapping:
            return lambda g, geo=geo, m=dict(mapping): replace(geo(g), m)
    return None


@solver("upscale")
def upscale_rule(demos: list[Pair]) -> Transform | None:
    """Output is the input with every cell blown up to a kh x kw block."""
    factors = set()
    for p in demos:
        (ih, iw), (oh, ow) = shape(p.input), shape(p.output)
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return None
        factors.add((oh // ih, ow // iw))
    if len(factors) != 1:
        return None
    kh, kw = factors.pop()
    if (kh, kw) == (1, 1):
        return None
    fn = lambda g: upscale(g, kh, kw)
    return fn if _fits(fn, demos) else None


@solver("tile")
def tile_rule(demos: list[Pair]) -> Transform | None:
    """Output is the input repeated m x n times, optionally mirroring cells."""
    factors = set()
    for p in demos:
        (ih, iw), (oh, ow) = shape(p.input), shape(p.output)
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return None
        factors.add((oh // ih, ow // iw))
    if len(factors) != 1:
        return None
    m, n = factors.pop()
    if (m, n) == (1, 1):
        return None

    plain = lambda g: tile(g, m, n)
    if _fits(plain, demos):
        return plain

    # Mirrored tilings. Which axis alternates -- and whether the flip starts on
    # the even or the odd block -- are four separate hypotheses, and tasks in
    # the wild use all of them.
    def mirrored(flip_rows: bool, flip_cols: bool, offset: int):
        def fn(g: Grid) -> Grid:
            h, w = shape(g)
            out = [[0] * (w * n) for _ in range(h * m)]
            for i in range(m):
                for j in range(n):
                    blk = [list(row) for row in g]
                    if flip_cols and (j + offset) % 2:
                        blk = [row[::-1] for row in blk]
                    if flip_rows and (i + offset) % 2:
                        blk = blk[::-1]
                    for r in range(h):
                        for c in range(w):
                            out[i * h + r][j * w + c] = blk[r][c]
            return out

        return fn

    for flip_rows, flip_cols in ((False, True), (True, False), (True, True)):
        for offset in (0, 1):
            fn = mirrored(flip_rows, flip_cols, offset)
            if _fits(fn, demos):
                return fn
    return None


@solver("crop_to_content")
def crop_to_content(demos: list[Pair]) -> Transform | None:
    """Output is the input cropped to the bounding box of the foreground."""
    for bg_mode in ("zero", "modal"):

        def fn(g: Grid, bg_mode=bg_mode) -> Grid:
            bg = 0 if bg_mode == "zero" else background(g)
            b = bbox(g, bg)
            if b is None:
                raise ValueError("empty grid")
            return crop(g, *b)

        if _fits(fn, demos):
            return fn
    return None


def _split_panels(g: Grid) -> list[Grid] | None:
    """Split on full-width / full-height separator lines of a single color."""
    h, w = shape(g)
    counts = Counter(v for row in g for v in row)
    for sep, _ in counts.most_common(4):
        rows = [r for r in range(h) if all(v == sep for v in g[r])]
        cols = [c for c in range(w) if all(g[r][c] == sep for r in range(h))]
        if not rows and not cols:
            continue

        def spans(cuts: list[int], n: int) -> list[tuple[int, int]]:
            out, start = [], 0
            for x in sorted(cuts) + [n]:
                if x > start:
                    out.append((start, x - 1))
                start = x + 1
            return out

        rspans, cspans = spans(rows, h), spans(cols, w)
        panels = [crop(g, r0, c0, r1, c1) for r0, r1 in rspans for c0, c1 in cspans]
        panels = [p for p in panels if p and p[0]]
        if len(panels) >= 2:
            return panels
    return None


@solver("select_panel")
def select_panel(demos: list[Pair]) -> Transform | None:
    """The grid splits into panels; pick the odd one out, or the modal one."""

    def picker(mode: str):
        def fn(g: Grid) -> Grid:
            panels = _split_panels(g)
            if not panels:
                raise ValueError("no panels")
            keys = [repr(p) for p in panels]
            cnt = Counter(keys)
            if mode == "unique":
                for p, k in zip(panels, keys):
                    if cnt[k] == 1:
                        return p
                raise ValueError("no unique panel")
            if mode == "modal":
                top = cnt.most_common(1)[0][0]
                return panels[keys.index(top)]
            if mode == "most_colors":
                return max(panels, key=lambda p: len({v for r in p for v in r}))
            raise ValueError(mode)

        return fn

    for mode in ("unique", "modal", "most_colors"):
        fn = picker(mode)
        if _fits(fn, demos):
            return fn
    return None


@solver("downscale")
def downscale(demos: list[Pair]) -> Transform | None:
    """Output is the input shrunk by an integer block factor."""
    factors = set()
    for p in demos:
        (ih, iw), (oh, ow) = shape(p.input), shape(p.output)
        if oh == 0 or ow == 0 or ih % oh or iw % ow:
            return None
        factors.add((ih // oh, iw // ow))
    if len(factors) != 1:
        return None
    kh, kw = factors.pop()
    if (kh, kw) == (1, 1):
        return None

    def fn(g: Grid) -> Grid:
        h, w = shape(g)
        return [
            [
                Counter(
                    g[r][c] for r in range(i, i + kh) for c in range(j, j + kw)
                ).most_common(1)[0][0]
                for j in range(0, w, kw)
            ]
            for i in range(0, h, kh)
        ]

    return fn if _fits(fn, demos) else None


# --------------------------------------------------------------------------
# portfolio
# --------------------------------------------------------------------------


def solve_task(
    demos: list[Pair], tests: Iterable[Grid]
) -> tuple[list[list[Grid]], list[str]]:
    """Return (per-test ranked predictions, names of the rules that fit)."""
    fitted: list[tuple[str, Transform]] = []
    for name, s in _REGISTRY:
        try:
            fn = s(demos)
        except Exception:
            fn = None
        if fn is not None and _fits(fn, demos):
            fitted.append((name, fn))

    per_test: list[list[Grid]] = []
    for t in list(tests):
        preds: list[Grid] = []
        for _, fn in fitted:
            try:
                g = fn(t)
            except Exception:
                continue
            if g and g[0] and g not in preds:
                preds.append(g)
        per_test.append(preds)
    return per_test, [n for n, _ in fitted]


def rule_names() -> list[str]:
    return [n for n, _ in _REGISTRY]


# --------------------------------------------------------------------------
# full pipeline: rule portfolio + compositional search
# --------------------------------------------------------------------------


def solve_task_full(
    demos: list[Pair],
    tests: list[Grid],
    max_depth: int = 3,
    use_search: bool = True,
) -> tuple[list[list[Grid]], list[str]]:
    """Rank candidate outputs from every stage we have.

    Stage order encodes a prior over hypothesis classes, strongest first:
    a hand-written rule that fits is more trustworthy than a searched
    composition of the same length, and a verified closed-form learner
    (symmetry repair, neighbourhood rule) is more trustworthy than either.
    Only the first two survive into the submission, so order matters as much
    as coverage.
    """
    from .search import (
        compose_search,
        learn_color_map_after,
        learn_neighborhood_rule,
        learn_symmetry_repair,
    )

    tests = list(tests)
    ranked: list[tuple[str, list[Grid]]] = []

    for name, fn in _REGISTRY:
        try:
            t = fn(demos)
        except Exception:
            t = None
        if t is None or not _fits(t, demos):
            continue
        try:
            preds = [t(x) for x in tests]
        except Exception:
            continue
        if all(p and p[0] for p in preds):
            ranked.append((name, preds))

    if use_search:
        for stage in (learn_symmetry_repair, learn_neighborhood_rule):
            try:
                for prog, preds in stage(demos, tests):
                    ranked.append(("+".join(prog), preds))
            except Exception:
                pass
        try:
            for prog, preds in compose_search(demos, tests, max_depth=max_depth):
                ranked.append(("+".join(prog) or "identity", preds))
        except Exception:
            pass
        try:
            for prog, preds in learn_color_map_after(demos, tests, max_depth=2):
                ranked.append(("+".join(prog), preds))
        except Exception:
            pass

    per_test: list[list[Grid]] = [[] for _ in tests]
    for _, preds in ranked:
        for i, g in enumerate(preds):
            if g and g[0] and g not in per_test[i]:
                per_test[i].append(g)
    return per_test, [n for n, _ in ranked]
