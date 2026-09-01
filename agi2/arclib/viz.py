"""Render ARC grids -- to the terminal, or to a standalone HTML page.

The HTML renderer is what you want for the Paper Track: a writeup needs a cover
image and a media gallery, and screenshots of real tasks next to your model's
predictions are the cheapest way to make a method legible to a judge.
"""
from __future__ import annotations

import html
from pathlib import Path

from .loader import Grid, Task

#: The canonical ARC ten-color palette.
PALETTE = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 fuchsia
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 teal
    "#870C25",  # 9 brown
]

#: 256-color terminal codes that approximate the palette above.
_ANSI = [16, 33, 203, 41, 220, 145, 200, 208, 117, 88]


def to_terminal(g: Grid) -> str:
    """Two-space colored blocks, one per cell."""
    lines = []
    for row in g:
        lines.append("".join(f"\033[48;5;{_ANSI[v]}m  \033[0m" for v in row))
    return "\n".join(lines)


def _svg(g: Grid, cell: int = 14) -> str:
    h, w = len(g), len(g[0]) if g else 0
    rects = [
        f'<rect x="{c * cell}" y="{r * cell}" width="{cell}" height="{cell}" '
        f'fill="{PALETTE[v % 10]}"/>'
        for r, row in enumerate(g)
        for c, v in enumerate(row)
    ]
    grid_lines = [
        f'<path d="M0 {r * cell}H{w * cell}" stroke="#222" stroke-width="0.5"/>'
        for r in range(h + 1)
    ] + [
        f'<path d="M{c * cell} 0V{h * cell}" stroke="#222" stroke-width="0.5"/>'
        for c in range(w + 1)
    ]
    return (
        f'<svg viewBox="0 0 {w * cell} {h * cell}" width="{w * cell}" '
        f'height="{h * cell}" shape-rendering="crispEdges">'
        + "".join(rects)
        + "".join(grid_lines)
        + "</svg>"
    )


_CSS = """
body{background:#111;color:#ddd;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
     margin:0;padding:24px}
h1{font-size:16px;font-weight:600;margin:0 0 4px}
h2{font-size:13px;font-weight:600;color:#888;margin:24px 0 8px;text-transform:uppercase;
   letter-spacing:.08em}
.row{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;
     overflow-x:auto;padding-bottom:8px}
.pair{display:flex;gap:10px;align-items:center;background:#1a1a1a;padding:10px;
      border-radius:6px;border:1px solid #2a2a2a}
.cell{display:flex;flex-direction:column;gap:4px;align-items:center}
.cap{color:#777;font-size:11px}
.arrow{color:#555;font-size:18px}
.ok{border-color:#2ECC40}
.bad{border-color:#FF4136}
"""


def _panel(g: Grid, caption: str, cell: int) -> str:
    return (
        f'<div class="cell">{_svg(g, cell)}'
        f'<div class="cap">{html.escape(caption)} {len(g)}x{len(g[0]) if g else 0}</div></div>'
    )


def task_to_html(
    task: Task,
    predictions: list[list[Grid]] | None = None,
    cell: int = 14,
    title: str | None = None,
) -> str:
    """Render demos, test inputs, and (optionally) ranked predictions."""
    parts = [
        "<style>",
        _CSS,
        "</style>",
        f"<h1>{html.escape(title or task.task_id)}</h1>",
        "<h2>demonstrations</h2><div class='row'>",
    ]
    for i, p in enumerate(task.train):
        parts.append(
            f"<div class='pair'>{_panel(p.input, f'train{i} in', cell)}"
            f"<div class='arrow'>&rarr;</div>"
            f"{_panel(p.output, f'train{i} out', cell)}</div>"
        )
    parts.append("</div><h2>test</h2><div class='row'>")
    for i, p in enumerate(task.test):
        block = [_panel(p.input, f"test{i} in", cell)]
        if p.output is not None:
            block += ["<div class='arrow'>&rarr;</div>", _panel(p.output, "truth", cell)]
        cls = "pair"
        if predictions and i < len(predictions):
            for j, pred in enumerate(predictions[i][:2]):
                block += [
                    "<div class='arrow'>&rarr;</div>",
                    _panel(pred, f"attempt_{j + 1}", cell),
                ]
            if p.output is not None:
                cls += " ok" if p.output in predictions[i][:2] else " bad"
        parts.append(f"<div class='{cls}'>" + "".join(block) + "</div>")
    parts.append("</div>")
    return "".join(parts)


def write_html(body: str, path: str | Path, title: str = "ARC") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>{body}",
        encoding="utf-8",
    )
    return path
