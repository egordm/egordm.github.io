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


def two_paths(theme):
    s = SVG(theme, 596, "Same fact. Different work.",
            "Invented example: Kai lives next to the Rijksmuseum. Who lives next to "
            "the Rijksmuseum shares words with the fact. Who has been to Amsterdam "
            "shares none: it needs a hop from Amsterdam to the Rijksmuseum before "
            "retrieving Kai.")
    s.header("01", "THE LOOKUP", "Same fact. Different work.", "An invented example, not a benchmark item.")
    s.box(24, 112, 372, 78, stroke="line")
    s.text(40, 136, "BURIED IN THE DOCUMENT", 12, "muted", 700)
    s.text(40, 167, "Kai lives next to the Rijksmuseum.", 19, weight=700)

    s.box(24, 208, 372, 139, "teal_bg")
    s.text(40, 235, "WORD MATCH", 12, "teal", 700)
    s.text(40, 265, "Who lives next to the Rijksmuseum?", 18, weight=700)
    s.box(40, 285, 168, 38, "panel")
    s.text(124, 310, "Rijksmuseum", 16, "teal", 700, "middle")
    s.arrow(218, 304, 276, 304, "teal")
    s.box(286, 285, 92, 38, "panel")
    s.text(332, 310, "Kai", 17, weight=700, anchor="middle")

    s.box(24, 363, 372, 171, "blue_bg")
    s.text(40, 390, "MEANING HOP", 12, "blue", 700)
    s.text(40, 420, "Who has been to Amsterdam?", 20, weight=700)
    s.text(40, 446, "No shared words with the planted fact.", 14, "muted")
    for x, w, text in [(40, 98, "Amsterdam"), (166, 128, "Rijksmuseum"), (322, 56, "Kai")]:
        s.box(x, 465, w, 39, "panel")
        s.text(x + w / 2, 490, text, 14, "blue" if x != 322 else "ink", 700, "middle")
    s.arrow(143, 484, 161, 484, "blue")
    s.arrow(299, 484, 317, 484, "blue")
    s.text(24, 562, "Finding the words is only one kind of recall.", 15, weight=700)
    s.source(582)
    s.save("two-paths", theme)


def experiment(theme):
    s = SVG(theme, 670, "Bury it. Move it. Ask.",
            "Experiment schematic. Grow book context from 250 to 512K tokens. "
            "In separate runs place one fact at 25, 50 or 75 percent depth. "
            "Ask a meaning question and its shared-word control separately, "
            "using 28 question types. Reasoning is off or high, with high on a subset. "
            "No answer uses tools. Removing the fact yields zero correct names out of 28.")
    s.header("02", "THE EXPERIMENT", "Bury it. Move it. Ask.", "GPT-6 Luna · NoLiMa's published questions")

    s.text(24, 126, "1", 18, "blue", 700)
    s.text(49, 126, "Grow the book context", 18, weight=700)
    for x, w in [(49, 42), (107, 92), (215, 181)]:
        s.box(x, 142, w, 46, "panel", "line", 5)
        for y in (153, 164, 175):
            s.line(x + 9, y, x + w - 9, y, width=2)
    s.text(49, 211, "250 tokens", 14, "muted")
    s.arrow(145, 207, 275, 207)
    s.text(396, 211, "512K", 14, "muted", anchor="end")
    s.text(49, 232, "Schematic lengths, not to scale.", 12, "muted")

    s.text(24, 267, "2", 18, "blue", 700)
    s.text(49, 267, "Move one fact between runs", 18, weight=700)
    for x, depth, label in [(49, .25, "25%"), (175, .5, "50%"), (301, .75, "75%")]:
        s.box(x, 285, 95, 120, "panel", "line", 6)
        for y in range(296, 400, 10):
            s.line(x + 10, y, x + 85, y, width=2)
        s.box(x + 3, 285 + 120 * depth - 9, 89, 18, "amber_bg", radius=2)
        s.line(x + 9, 285 + 120 * depth, x + 86, 285 + 120 * depth, "amber", 3)
        s.text(x + 47.5, 427, label, 16, "amber", 700, "middle")
    s.text(49, 451, "Depth = fraction of the document before it.", 13, "muted")

    s.text(24, 487, "3", 18, "blue", 700)
    s.text(49, 487, "Ask both kinds of question", 18, weight=700)
    s.box(49, 502, 163, 39, "blue_bg", radius=6)
    s.text(130.5, 527, "Meaning", 16, "blue", 700, "middle")
    s.box(223, 502, 173, 39, "teal_bg", radius=6)
    s.text(309.5, 527, "Shared words", 16, "teal", 700, "middle")
    s.text(49, 561, "28 question types · no tools used", 14, "muted")
    s.text(49, 583, "Reasoning off; high on selected lengths.", 14, "muted")

    s.box(24, 604, 372, 37, "panel", "line", 6)
    s.text(40, 628, "Fact removed → 0 / 28 correct names", 16, weight=700)
    s.source(660)
    s.save("experiment", theme)


def compaction(theme):
    s = SVG(theme, 656, "Compact before the fall.",
            "Compaction guide for meaning-based work. The full linear bar runs from "
            "zero to the advertised one million tokens, with no measurements beyond 512K. "
            "A second linear bar zooms in on the default 272K window. The usable "
            "ceilings are 64K with reasoning off and about 128K with reasoning high; "
            "the latter varies from 64K to 192K by book. The default compaction point "
            "is 95 percent of 272K, about 258K, already past the decline. Usable means "
            "keeping at least 85 percent of the short-context score at the longest length.")
    s.header("03", "THE COMPACTION GUIDE", "Compact before the fall.", "Meaning-based work · lengths in tokens")

    s.box(24, 110, 179, 88, "teal_bg")
    s.text(40, 135, "REASONING OFF", 12, "teal", 700)
    s.text(40, 167, "64K", 29, "teal", 700)
    s.text(40, 187, "useful ceiling", 13, "muted")
    s.box(215, 110, 181, 88, "blue_bg")
    s.text(231, 135, "REASONING HIGH", 12, "blue", 700)
    s.text(231, 167, "≈128K", 29, "blue", 700)
    s.text(231, 187, "useful ceiling", 13, "muted")

    s.text(24, 232, "THE ADVERTISED CAPACITY", 12, "muted", 700)
    y, w = 252, 372
    for a, b, color in [(0, 64, "teal"), (64, 128, "blue"), (128, 512, "amber")]:
        s.box(24 + w * a / 1000, y, w * (b - a) / 1000, 24, color, radius=0)
    s.box(24 + w * .512, y, w * .488, 24, "faint", radius=0)
    s.box(24 + w * .512, y, w * .488, 24, "url(#untested)", radius=0)
    s.text(24, 297, "0", 13, "muted")
    s.text(24 + w * .512, 297, "512K", 13, "muted", anchor="middle")
    s.text(396, 297, "1M", 15, weight=700, anchor="end")
    s.text(396, 318, "Hatched: beyond the measured range", 13, "muted", anchor="end")

    s.text(24, 353, "ZOOM: THE CODEX DEFAULT WINDOW", 12, "muted", 700)
    s.text(24, 377, "0", 13, "muted")
    s.text(396, 377, "272K", 15, weight=700, anchor="end")
    y = 387
    for a, b, color in [(0, 64, "teal"), (64, 128, "blue"), (128, 272, "amber")]:
        s.box(24 + w * a / 272, y, w * (b - a) / 272, 30, color, radius=0)
    for a, text, color in [(64, "64K", "teal"), (128, "≈128K", "blue")]:
        x = 24 + w * a / 272
        s.line(x, 419, x, 428, color)
        s.text(x, 447, text, 14, color, 700, "middle")
    x = 24 + w * (272 * .95) / 272
    s.line(x, 381, x, 454, "bg", 5)
    s.line(x, 381, x, 454, "ink", 2)
    s.text(396, 478, "≈258K: default compaction", 16, weight=700, anchor="end")
    s.text(396, 500, "95% of 272K. Already past the fall.", 14, "muted", anchor="end")

    s.box(24, 519, 11, 11, "teal", radius=2)
    s.text(42, 530, "Usable: off", 13, "muted")
    s.box(154, 519, 11, 11, "blue", radius=2)
    s.text(172, 530, "+ high", 13, "muted")
    s.box(255, 519, 11, 11, "amber", radius=2)
    s.text(273, 530, "Degrading", 13, "muted")

    s.line(24, 548, 396, 548)
    s.text(24, 571, "Usable = longest length retaining at least", 14, "muted")
    s.text(24, 592, "85% of the short-context score.", 14, "muted")
    s.text(24, 613, "With reasoning high: 64K to 192K by book.", 14, "muted")
    s.source(640, "Luna: this experiment · yardstick: NoLiMa")
    s.save("compaction", theme)


def main():
    with tempfile.TemporaryDirectory(prefix="luna-v-mpl-", dir="/tmp") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        for theme in PALETTES:
            two_paths(theme)
            experiment(theme)
            compaction(theme)


if __name__ == "__main__":
    main()
