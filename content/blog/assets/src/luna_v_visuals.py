# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Rebuild the three editorial diagrams, offline, with system Python.

Run: /usr/bin/python3 content/blog/assets/src/luna_v_visuals.py
Writes only luna-v-{name}-{light,dark}.svg alongside this src directory.
All data comes from the author's supplied facts; the Rijksmuseum example is
invented. Bars are linear; the experiment's document lengths are schematic.
SVG text stays selectable. Each canvas is designed at 420 CSS pixels wide.
No plotting dependency is required. MPLCONFIGDIR is kept outside content/
even if this generator is later extended to use matplotlib.
"""

import os
import tempfile
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent
PALETTES = {
    "light": dict(bg="#ffffff", panel="#ffffff", ink="#252725",
                  muted="#595e5b", line="#d8dcd7", teal="#00796b",
                  teal_bg="#e4f3ee", blue="#4557ba", blue_bg="#e9edfc",
                  amber="#a8460c", amber_bg="#faebdf", faint="#eeeFEB"),
    "dark": dict(bg="#1e1e1e", panel="#292b2a", ink="#efefeb",
                 muted="#bdbdb8", line="#4c504d", teal="#7bd7c8",
                 teal_bg="#253e38", blue="#a6b8ff", blue_bg="#30374f",
                 amber="#ffa96e", amber_bg="#443226", faint="#323532"),
}


class SVG:
    def __init__(self, theme, height, title, description):
        self.c = PALETTES[theme]
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="420" height="{height}" '
            f'viewBox="0 0 420 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            '<defs><pattern id="untested" width="8" height="8" '
            'patternUnits="userSpaceOnUse"><path d="M-2 2L2-2M0 8L8 0M6 10L10 6" '
            f'fill="none" stroke="{self.c["line"]}" stroke-width="1"/></pattern></defs>',
        ]
        self.box(0, 0, 420, height, "bg", radius=0)

    def color(self, name):
        return self.c.get(name, name)

    def box(self, x, y, w, h, fill="panel", stroke=None, radius=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{radius}" fill="{self.color(fill)}"'
                          + (f' stroke="{self.color(stroke)}"' if stroke else '') + '/>')

    def text(self, x, y, value, size=15, color="ink", weight=400, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" fill="{self.color(color)}" '
                          f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
                          f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')

    def line(self, x1, y1, x2, y2, color="line", width=1.5, dashed=False):
        self.parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" '
                          f'stroke="{self.color(color)}" stroke-width="{width}" '
                          f'stroke-linecap="round"' + (' stroke-dasharray="4 4"' if dashed else '') + '/>')

    def arrow(self, x1, y1, x2, y2, color="muted"):
        self.line(x1, y1, x2, y2, color)
        if y1 == y2:
            self.line(x2 - 5, y2 - 4, x2, y2, color)
            self.line(x2 - 5, y2 + 4, x2, y2, color)
        else:
            self.line(x2 - 4, y2 - 5, x2, y2, color)
            self.line(x2 + 4, y2 - 5, x2, y2, color)

    def header(self, number, kicker, title, subtitle):
        self.text(24, 30, f"{number} / {kicker}", 12, "muted", 700)
        self.text(24, 65, title, 25, weight=700)
        self.text(24, 91, subtitle, 14, "muted")

    def source(self, y, label="Method: NoLiMa · arXiv 2502.05167"):
        self.parts.append('<a href="https://arxiv.org/abs/2502.05167">')
        self.text(24, y, label, 13, "muted")
        self.parts.append('</a>')

    def save(self, name, theme):
        path = ASSETS / f"luna-v-{name}-{theme}.svg"
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


def curve(theme):
    """Draw every JSON observation and its pointwise 95% Wilson interval."""
    import json
    import math

    data = json.loads(Path(__file__).with_name("luna_context_curve.json").read_text())
    series = [("literal_reasoning_off", "teal", "Shared words · reasoning off", "square"),
              ("latent_reasoning_off", "amber", "Meaning · reasoning off", "circle"),
              ("latent_reasoning_high", "blue", "Meaning · reasoning high", "diamond")]
    s = SVG(theme, 460, "Luna recall as context grows",
            "All observations from luna_context_curve.json, connected without extrapolation. "
            "Context length uses a logarithmic scale; accuracy is correct answers divided by calls. "
            "Shading shows pointwise 95% Wilson binomial intervals, not variation across books or runs. "
            "Control and reasoning off end at 224K; reasoning high ends at 512K. No data at 1M.")
    minimum = min(p[0] for name, _, _, _ in series for p in data[name]["points"])
    maximum = data["markers"]["advertised_window"]

    def x(tokens):
        return 48 + 348 * math.log(tokens / minimum) / math.log(maximum / minimum)

    def y(accuracy):
        return 368 - 250 * accuracy

    def interval(correct, total):
        z = 1.959963984540054
        p = correct / total
        denominator = 1 + z * z / total
        center = (p + z * z / (2 * total)) / denominator
        half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
        return max(0, center - half), min(1, center + half)

    def mark(px, py, color, shape):
        fill = s.color(color)
        if shape == "circle":
            s.parts.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{fill}"/>')
        elif shape == "square":
            s.box(px - 3, py - 3, 6, 6, color, radius=0)
        else:
            s.parts.append(f'<path d="M{px} {py-4}L{px+4} {py}L{px} {py+4}L{px-4} {py}Z" fill="{fill}"/>')

    for i, (_, color, label, shape) in enumerate(series):
        yy = 22 + i * 23
        s.line(24, yy - 4, 46, yy - 4, color, 2, dashed=shape == "diamond")
        mark(35, yy - 4, color, shape)
        s.text(56, yy, label, 14, color)
    s.text(16, 101, "% correct", 12, "muted")
    for tick in (0, 25, 50, 75, 100):
        yy = y(tick / 100)
        s.line(48, yy, 396, yy, width=1)
        s.text(39, yy + 4, str(tick), 12, "muted", anchor="end")
    for key, label in (("codex_default_window", "272K Codex"),
                       ("advertised_window", "1M")):
        xx = x(data["markers"][key])
        s.line(xx, 110, xx, 368, "muted", 1, dashed=True)
        s.text(xx, 101, label, 12, "muted", anchor="end")

    # Draw all intervals before the observation lines so no band hides a line.
    for name, color, _, _ in series:
        points = data[name]["points"]
        previous = 0
        for tokens, correct, total in points:
            if not (tokens > previous and total > 0 and 0 <= correct <= total):
                raise ValueError(f"Invalid or unsorted counts in {name}")
            previous = tokens
        bounds = [(x(t), interval(k, n)) for t, k, n in points]
        polygon = [(px, y(hi)) for px, (_, hi) in bounds]
        polygon += [(px, y(lo)) for px, (lo, _) in reversed(bounds)]
        coordinates = " ".join(f"{px},{py}" for px, py in polygon)
        s.parts.append(f'<polygon points="{coordinates}" fill="{s.color(color)}" opacity="0.13"/>')
    for name, color, _, shape in series:
        points = data[name]["points"]
        path = "M" + " L".join(f"{x(t)} {y(k/n)}" for t, k, n in points)
        dash = ' stroke-dasharray="5 4"' if shape == "diamond" else ''
        s.parts.append(f'<path d="{path}" fill="none" stroke="{s.color(color)}" '
                       f'stroke-width="2" stroke-linejoin="round"{dash}/>')
        for tokens, correct, total in points:
            s.parts.append(f'<g data-series="{name}" data-tokens="{tokens}" '
                           f'data-correct="{correct}" data-total="{total}">'
                           f'<title>{tokens} tokens: {correct}/{total} correct</title>')
            mark(x(tokens), y(correct / total), color, shape)
            s.parts.append('</g>')
    for tokens, label in ((250, "250"), (1000, "1K"), (16000, "16K"),
                          (128000, "128K"), (maximum, "1M")):
        xx = x(tokens)
        s.line(xx, 368, xx, 373, "muted", 1)
        s.text(xx, 391, label, 12, "muted", anchor="end" if tokens == maximum else "middle")
    s.text(222, 416, "Context length · tokens (log scale)", 13, "muted", anchor="middle")
    s.text(210, 444, "Bands: 95% Wilson intervals · no data at 1M", 12, "muted", anchor="middle")
    s.save("curve", theme)


def two_paths(theme):
    s = SVG(theme, 330, "Two ways to retrieve Kai",
            "Invented example: Kai lives next to the Rijksmuseum. The shared-word question "
            "goes from Rijksmuseum to Kai. The meaning question needs the connection from "
            "Amsterdam to Rijksmuseum before retrieving Kai.")
    s.box(28, 16, 364, 42, "panel", "line", 7)
    s.text(210, 42, "Kai lives next to the Rijksmuseum.", 16, anchor="middle")
    s.text(28, 94, "Shared words", 13, "teal", 700)
    s.text(28, 119, "Who lives next to the Rijksmuseum?", 16)
    s.box(28, 136, 132, 36, "teal_bg", "teal", 7)
    s.text(94, 159, "Rijksmuseum", 15, "teal", anchor="middle")
    s.arrow(168, 154, 314, 154, "teal")
    s.box(322, 136, 70, 36, "teal_bg", "teal", 7)
    s.text(357, 159, "Kai", 16, "teal", anchor="middle")
    s.line(28, 194, 392, 194, width=1)
    s.text(28, 222, "Meaning", 13, "blue", 700)
    s.text(28, 247, "Who has been to Amsterdam?", 16)
    for xx, width, label in ((28, 100, "Amsterdam"), (162, 126, "Rijksmuseum"), (322, 70, "Kai")):
        s.box(xx, 264, width, 36, "blue_bg", "blue", 7)
        s.text(xx + width / 2, 287, label, 14, "blue", anchor="middle")
    s.arrow(135, 282, 155, 282, "blue")
    s.arrow(295, 282, 315, 282, "blue")
    s.save("two-paths", theme)


def experiment(theme):
    s = SVG(theme, 450, "Document length, fact depth, question",
            "Document lengths grow from 250 to 512K tokens, shown schematically. "
            "Separate runs place one fact at 25%, 50%, or 75% depth and ask either "
            "a meaning question or a shared-word control. Removing the fact gives "
            "zero correct names in 28 questions.")

    def document(xx, yy, width, height, depth=None):
        s.box(xx, yy, width, height, "panel", "line", 5)
        for row in range(10, height - 5, 10):
            s.line(xx + 10, yy + row, xx + width - 10, yy + row, width=2)
        if depth is not None:
            fy = yy + height * depth
            s.box(xx + 3, fy - 6, width - 6, 12, "amber_bg", radius=2)
            s.line(xx + 10, fy, xx + width - 10, fy, "amber", 3)

    for xx, height in ((44, 32), (172, 54), (300, 78)):
        document(xx, 96 - height, 76, height)
    s.text(82, 120, "250 tokens", 14, "muted", anchor="middle")
    s.arrow(146, 115, 275, 115)
    s.text(338, 120, "512K", 14, "muted", anchor="middle")
    s.text(28, 145, "Fact depth", 12, "muted")
    for xx, depth in ((44, .25), (172, .5), (300, .75)):
        document(xx, 153, 76, 120, depth)
        s.text(xx + 38, 296, f"{depth:.0%}", 14, "amber", 700, "middle")
        s.line(xx + 38, 304, xx + 38, 316, width=1)
    s.line(82, 316, 338, 316, width=1)
    for xx, width, color, label in ((44, 145, "blue", "Meaning"), (231, 145, "teal", "Shared words")):
        s.arrow(xx + width / 2, 316, xx + width / 2, 333)
        s.box(xx, 340, width, 36, color + "_bg", color, 7)
        s.text(xx + width / 2, 363, label, 15, color, anchor="middle")
    s.text(210, 408, "Fact removed → 0 / 28 correct names", 14, anchor="middle")
    s.text(210, 437, "Document lengths schematic; one fact per run.", 12, "muted", anchor="middle")
    s.save("experiment", theme)


def compaction(theme):
    s = SVG(theme, 364, "Usable context and default compaction",
            "Linear scales. The full one-million-token capacity is measured only through 512K. "
            "The lower scale expands the 272K Codex window: usable context reaches 64K with "
            "reasoning off and about 128K with reasoning high. Default compaction is about "
            "258K, past both. Usable means retaining at least 85% of the short-context score.")
    left, width = 28, 364
    measured_end = left + width * .512
    s.text(left, 28, "Measured", 13, "muted")
    s.text(304, 28, "Unmeasured", 13, "muted", anchor="middle")
    s.box(left, 42, width * .512, 22, "line", radius=0)
    s.box(measured_end, 42, width * .488, 22, "faint", radius=0)
    s.box(measured_end, 42, width * .488, 22, "url(#untested)", radius=0)
    for xx, label, anchor in ((left, "0", "start"), (measured_end, "512K", "middle"), (392, "1M", "end")):
        s.text(xx, 84, label, 13, "muted", anchor=anchor)
    # Expand the first 272K of the capacity bar into the lower linear scale.
    zoom_end = left + width * .272
    s.line(left, 91, zoom_end, 91, "muted", 1)
    s.line(left, 87, left, 95, "muted", 1)
    s.line(zoom_end, 87, zoom_end, 95, "muted", 1)
    s.line(left, 96, left, 124, width=1)
    s.line(zoom_end, 96, 392, 124, width=1)
    s.text(left, 145, "Codex window: 272K", 14, "muted")
    for yy, tokens, color, label in ((186, 64, "amber", "Usable · reasoning off"),
                                    (238, 128, "blue", "Usable · reasoning high")):
        s.text(left, yy - 10, label, 14, color)
        s.box(left, yy, width, 14, "faint", radius=0)
        length = width * tokens / 272
        s.box(left, yy, length, 14, color, radius=0)
        s.text(left + length + 9, yy + 12, f"{tokens}K", 14, color, 700)
    s.line(left, 278, 392, 278, width=1)
    for xx, label, anchor in ((left, "0", "start"), (392, "272K", "end")):
        s.line(xx, 273, xx, 282, "muted", 1)
        s.text(xx, 299, label, 12, "muted", anchor=anchor)
    default = left + width * .95
    s.line(default, 160, default, 278, "ink", 1.5, dashed=True)
    s.line(28, 324, 50, 324, "ink", 1.5, dashed=True)
    s.text(64, 329, "Default compaction ≈258K", 14)
    s.text(210, 353, "Usable: at least 85% of short-context score.", 12, "muted", anchor="middle")
    s.save("compaction", theme)


def main():
    for theme in PALETTES:
        curve(theme)
        two_paths(theme)
        experiment(theme)
        compaction(theme)


if __name__ == "__main__":
    main()
