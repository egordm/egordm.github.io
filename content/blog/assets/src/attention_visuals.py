#!/usr/bin/env python3
"""Build only attention-v SVGs, using system Python and its standard library.

Run: /usr/bin/python3 content/blog/assets/src/attention_visuals.py
All canvases are 420 px wide. Geometry is deterministic. Attention line widths
are proportional to the supplied weights; both vector plots use identical,
isotropic scales. Token/vector glyphs and the abbreviated layer stack are
schematic. The example is illustrative, not a model measurement.
"""

import sys

sys.dont_write_bytecode = True

from html import escape
from math import atan2, cos, sin, pi
from pathlib import Path

from luna_v_visuals import PALETTES


ASSETS = Path(__file__).resolve().parent.parent
WEIGHTS = {"A": (0.10, 0.05, 0.70, 0.05, 0.10),
           "B": (0.40, 0.10, 0.30, 0.10, 0.10)}
WRITES = {"A": (0.1, 1.0), "B": (3.0, 0.2)}


class Figure:
    def __init__(self, theme, name, height, description):
        self.theme, self.name, self.height = theme, name, height
        self.c = PALETTES[theme]
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="420" height="{height}" '
            f'viewBox="0 0 420 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(name.replace("-", " "))}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
        ]
        self.box(0, 0, 420, height, "bg", radius=0)

    def col(self, c):
        return self.c.get(c, c)

    def box(self, x, y, w, h, fill="panel", stroke=None, radius=7):
        border = f' stroke="{self.col(stroke)}"' if stroke else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{radius}" fill="{self.col(fill)}"{border}/>')

    def text(self, x, y, value, size=14, color="ink", anchor="start", bold=False):
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                          f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
                          f'font-weight="{700 if bold else 400}" fill="{self.col(color)}">'
                          f'{escape(value)}</text>')

    def math(self, x, y, body, size=20, color="ink", anchor="middle"):
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                          f'font-family="Georgia, Times New Roman, serif" font-size="{size}" '
                          f'fill="{self.col(color)}">{body}</text>')

    def path(self, d, color="muted", width=1.5, dashed=False, opacity=1):
        dash = ' stroke-dasharray="4 5"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{self.col(color)}" '
                          f'stroke-width="{width}" stroke-linecap="round" '
                          f'stroke-linejoin="round" opacity="{opacity}"{dash}/>')

    def line(self, x1, y1, x2, y2, **kw):
        self.path(f'M{x1} {y1} L{x2} {y2}', **kw)

    def arrow(self, x1, y1, x2, y2, color="muted", width=1.5, opacity=1):
        self.line(x1, y1, x2, y2, color=color, width=width, opacity=opacity)
        a = atan2(y2 - y1, x2 - x1)
        length = 6 if width < 3 else 8
        pts = [(x2 + length * cos(a + pi + sign * 0.5),
                y2 + length * sin(a + pi + sign * 0.5)) for sign in (-1, 1)]
        self.path(f'M{pts[0][0]} {pts[0][1]} L{x2} {y2} L{pts[1][0]} {pts[1][1]}',
                  color=color, width=width, opacity=opacity)

    def vector(self, x, y, color="blue", phase=0, w=22, h=44):
        self.box(x, y, w, h, color + "_bg", color, radius=3)
        for i in range(5):
            self.line(x + 5, y + 7 + i * 7, x + w - 5, y + 7 + i * 7,
                      color=color, width=3, opacity=(0.25, 0.55, 0.9)[(i + phase) % 3])

    def node(self, x, y, value, color="ink"):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="14" '
                          f'fill="{self.col("bg")}" stroke="{self.col(color)}" stroke-width="1.5"/>')
        self.text(x, y + 6, value, 21, color, "middle")

    def save(self):
        (ASSETS / f"attention-v-{self.name}-{self.theme}.svg").write_text(
            "\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


def sym(letter, sub=None):
    result = f'<tspan font-style="italic">{escape(letter)}</tspan>'
    if sub is not None:
        result += f'<tspan baseline-shift="sub" font-size="70%">{escape(sub)}</tspan>'
    return result


def stack(theme):
    f = Figure(theme, "stack", 420,
               "Selected tokens become vectors and pass upward through an abbreviated layer stack.")
    centers = (64, 154, 250, 354)
    for y in (106, 204):
        f.box(26, y, 368, 42, "panel", "line")
        f.text(210, y + 26, "layer", 14, "muted", "middle")
    f.text(210, 182, "⋮", 24, "muted", "middle")
    for i, (x, label, w) in enumerate(zip(centers, ("half", "a", "minute", "?"), (70, 52, 86, 52))):
        color = "teal" if i < 3 else "blue"
        f.box(x - w / 2, 349, w, 36, color + "_bg", color)
        f.text(x, 372, label, 16, color, "middle")
        f.arrow(x, 340, x, 318, color)
        f.vector(x - 11, 266, color, i)
        f.arrow(x, 258, x, 249, color)
        f.arrow(x, 196, x, 156, color)
        f.arrow(x, 97, x, 78, color)
        f.vector(x - 11, 25, color, i + 1)
    f.text(210, 407, "Selected positions; layer stack abbreviated.", 12, "muted", "middle")
    f.save()


def attention(theme):
    f = Figure(theme, "attention", 494,
               "A source's key is matched to the receiver's query. Softmax over permitted sources "
               "gives a weight that scales the separately transformed value.")
    f.text(117, 24, "source", 13, "muted", "middle")
    f.text(339, 24, "receiver", 13, "muted", "middle")
    for x, sub, color in ((117, "j", "teal"), (339, "t", "blue")):
        f.box(x - 34, 38, 68, 42, color + "_bg", color)
        f.math(x, 64, sym("x", sub), color=color)
    f.path("M117 80 L117 100 L58 100 L58 132", "teal")
    f.path("M117 100 L173 100 L173 132", "teal")
    f.arrow(339, 80, 339, 132, "blue")
    f.math(46, 120, sym("W", "V"), 16, "teal", "end")
    f.math(189, 120, sym("W", "K"), 16, "teal", "start")
    f.math(355, 115, sym("W", "Q"), 16, "blue", "start")
    for x, symbol, sub, color in ((58, "v", "j", "teal"),
                                  (173, "k", "j", "teal"), (339, "q", "t", "blue")):
        f.box(x - 26, 133, 52, 38, color + "_bg")
        f.math(x, 158, sym(symbol, sub), color=color)
    f.arrow(173, 177, 228, 207, "teal")
    f.arrow(339, 177, 284, 207, "blue")
    f.box(201, 212, 110, 36, "panel", "line")
    f.text(256, 235, "dot product", 14, anchor="middle")
    f.arrow(256, 250, 256, 286)
    f.text(272, 274, "scale", 12, "muted")
    f.box(197, 291, 118, 36, "panel", "line")
    f.text(256, 314, "softmax", 14, anchor="middle")
    f.text(256, 348, "all allowed sources", 12, "muted", "middle")
    f.arrow(256, 355, 256, 370)
    f.math(256, 394, sym("α", "j"), color="blue")
    f.arrow(58, 179, 58, 273, "teal")
    f.box(32, 280, 52, 40, "teal_bg")
    f.math(58, 306, sym("W", "O"), color="teal")
    f.arrow(58, 327, 58, 369, "teal")
    f.math(58, 394, sym("W", "O") + sym("v", "j"), color="teal")
    f.path("M58 407 L58 433 L145 433", "teal")
    f.path("M256 407 L256 433 L173 433", "blue")
    f.node(159, 433, "×")
    f.arrow(159, 448, 159, 462)
    f.text(176, 468, "source contribution", 13)
    f.save()


def heads(theme):
    f = Figure(theme, "heads", 454,
               "Two heads in one layer attend differently at the same answer position. "
               "Line width is proportional to span attention, on the same scale for both heads.")
    xs = (45, 126, 210, 294, 375)
    labels = ("start", "S1", "S2: fact", "S3", "Q")
    for head, dy, color in (("A", 0, "teal"), ("B", 218, "blue")):
        for x, label, weight in zip(xs, labels, WEIGHTS[head]):
            f.line(x, 101 + dy, 210, 174 + dy, color=color, width=weight * 22)
            w = 72 if label == "S2: fact" else 54
            f.box(x - w / 2, 32 + dy, w, 38,
                  "amber_bg" if label == "S2: fact" else "panel",
                  "amber" if label == "S2: fact" else "line")
            f.text(x, 56 + dy, label, 13, "amber" if label == "S2: fact" else "ink", "middle")
            f.text(x, 91 + dy, f"{weight:.2f}", 13, color, "middle")
        f.box(153, 165 + dy, 114, 40, color + "_bg", color)
        f.text(210, 191 + dy, f"Head {head}", 16, color, "middle", True)
    f.text(210, 445, "Attention summed over tokens in each part", 12, "muted", "middle")
    f.save()


def residual(theme):
    f = Figure(theme, "residual", 446,
               "At one position, the residual stream continues upward while heads and then an MLP "
               "read it and add updates. The branches do not replace the main stream.")
    f.box(115, 89, 278, 289, "panel", "line", radius=12)
    f.text(375, 365, "one layer", 12, "muted", "end")
    f.vector(63, 384, "blue")
    f.vector(63, 19, "blue", 1)
    f.arrow(74, 379, 74, 68, "blue", 4)
    f.text(99, 410, "embedding / earlier layers", 13, "muted")
    f.text(99, 45, "to later layers", 13, "muted")
    # Heads read the same pre-attention stream and return parallel updates.
    f.path("M74 345 L153 345 L153 321", "muted")
    f.path("M153 345 L248 345 L248 321", "muted")
    f.path("M248 345 L343 345 L343 321", "muted")
    f.line(91, 245, 343, 245)
    for x, label, color in ((153, "A", "teal"), (248, "B", "blue"), (343, "…", "muted")):
        f.box(x - 30, 271, 60, 50, "panel", color)
        f.text(x, 301, label, 18, color, "middle")
        f.arrow(x, 271, x, 245, color)
    f.text(248, 232, "heads", 12, "muted", "middle")
    f.node(74, 245, "+", "blue")
    # The MLP reads the stream after the attention update.
    f.path("M74 206 L248 206 L248 185", "muted")
    f.box(206, 139, 84, 46, "panel", "blue")
    f.text(248, 168, "MLP", 16, "blue", "middle")
    f.path("M248 139 L248 112 L91 112", "blue")
    f.node(74, 112, "+", "blue")
    f.save()


def shadow(theme):
    f = Figure(theme, "shadow", 550,
               "Identical coordinate scales show how each fact write is weighted then projected "
               "onto the unit answer direction. A: 0.70 times (0.1, 1.0), shadow 0.07. "
               "B: 0.30 times (3.0, 0.2), shadow 0.90.")
    origin_x, scale = 58, 88
    for head, oy, color in (("A", 186, "teal"), ("B", 438, "blue")):
        wx, wy = WRITES[head]
        alpha = WEIGHTS[head][2]
        raw_x, raw_y = origin_x + wx * scale, oy - wy * scale
        tip_x, tip_y = origin_x + wx * alpha * scale, oy - wy * alpha * scale
        f.text(204, oy - 128, f"Head {head}", 17, color, bold=True)
        f.text(204, oy - 106, f"attention {alpha:.2f}", 14, "muted")
        f.arrow(origin_x, oy, origin_x, oy - 120, "line")
        f.arrow(origin_x, oy, 364, oy, "muted")
        f.text(27, oy - 130, "other", 12, "muted")
        f.math(384, oy + 5, sym("u", "30"), 19, "muted")
        f.arrow(origin_x, oy, raw_x, raw_y, color, 2, opacity=0.35)
        f.arrow(origin_x, oy, tip_x, tip_y, color, 3)
        f.line(tip_x, tip_y, tip_x, oy, color=color, dashed=True)
        f.line(origin_x, oy, tip_x, oy, color=color, width=7)
        if head == "A":
            f.text(raw_x + 17, raw_y - 10, "(0.1, 1.0)", 13, "muted")
            f.path(f"M{tip_x} {oy + 8} L{tip_x} {oy + 26} L98 {oy + 26}", color)
            f.text(106, oy + 31, "0.07", 17, color, bold=True)
        else:
            f.text(raw_x - 5, raw_y - 14, "(3.0, 0.2)", 13, "muted", "end")
            f.text((origin_x + tip_x) / 2, oy + 31, "0.90", 17, color, "middle", True)
        f.text(363, oy + 31, 'toward “30”', 13, "muted", "end")
    f.line(26, 260, 394, 260, color="line")
    f.text(210, 527, "Faint: write · solid: weighted · axis: shadow", 12, "muted", "middle")
    f.save()


def main():
    for theme in PALETTES:
        for draw in (stack, attention, heads, residual, shadow):
            draw(theme)


if __name__ == "__main__":
    main()
