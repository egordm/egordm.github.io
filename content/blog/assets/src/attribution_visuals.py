#!/usr/bin/env python3
"""Rebuild this post's five attribution-v figures, offline with standard Python.

All canvases are 420 px wide. Probabilities and rounded weights come from the
post's invented example. Probability bars share a linear scale; signed weights
share another. Mask cells and redundancy scores are schematic, not measurements.
Only the shared palette and drawing primitives are imported; no shared file is
changed. Run: python3 content/blog/assets/src/attribution_visuals.py
"""

import sys
from html import escape
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from luna_v_visuals import ASSETS, PALETTES, SVG  # noqa: E402


class Figure(SVG):
    def math(self, x, y, body, size=18, color="ink", anchor="middle"):
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="Georgia, Times New Roman, serif" font-size="{size}" '
            f'fill="{self.color(color)}">{body}</text>'
        )

    def save(self, name, theme):
        (ASSETS / f"attribution-v-{name}-{theme}.svg").write_text(
            "\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8"
        )


def sub(symbol, index):
    return (f'<tspan font-style="italic">{escape(symbol)}</tspan>'
            f'<tspan baseline-shift="sub" font-size="12">{escape(index)}</tspan>')


def chip(s, x, y, w, label, color, kept=True):
    s.box(x, y, w, 30, color + "_bg" if kept else "bg",
          color if kept else "line", radius=5)
    s.text(x + w / 2, y + 20, label, 13, color if kept else "muted", anchor="middle")
    if not kept:
        s.line(x + 6, y + 15, x + w - 6, y + 15, "muted")


def ablate(theme):
    s = Figure(theme, 362, "Ablating sources",
               'The fixed answer is 3000. With both sources its probability is 90%; '
               'without lsof, 2%; without config, 95%. Bars share a linear scale.')
    s.text(24, 30, 'Fixed answer: “3000”', 16, weight=700)
    s.text(396, 30, 'probability', 13, "muted", anchor="end")
    for y, config, lsof, p in ((56, True, True, 90),
                              (151, True, False, 2),
                              (246, False, True, 95)):
        chip(s, 24, y, 145, "config: 8080", "amber", config)
        chip(s, 183, y, 145, "lsof: 3000", "teal", lsof)
        s.box(24, y + 43, 304, 20, "faint", radius=2)
        s.box(24, y + 43, 304 * p / 100, 20,
              "teal" if lsof else "amber", radius=2)
        s.text(396, y + 59, f"{p}%", 17, weight=700, anchor="end")
    s.text(210, 348, "Crossed out: source removed · invented probabilities", 12,
           "muted", anchor="middle")
    s.save("ablate", theme)


def generate_grade(theme):
    s = Figure(theme, 346, "Generate, then grade",
               'Generate once from the full context. Reuse the recorded answer '
               'in every grading pass, alongside a modified context. No answer is sampled again.')
    s.box(24, 16, 135, 38, stroke="line", radius=7)
    s.text(91.5, 40, "full context", 14, anchor="middle")
    s.arrow(168, 35, 214, 35)
    s.box(224, 16, 172, 38, "blue_bg", "blue", radius=7)
    s.text(310, 40, "Generate once", 14, "blue", 700, "middle")
    s.line(310, 54, 310, 74, "blue")
    s.line(310, 74, 210, 74, "blue")
    s.arrow(210, 74, 210, 91, "blue")
    s.box(144, 97, 132, 40, "blue_bg", "blue", radius=7)
    s.text(210, 123, "fixed answer", 16, "blue", 700, "middle")
    s.line(210, 137, 210, 158, "blue")
    s.line(82, 158, 338, 158, "blue")
    for x, label in ((24, "context A"), (152, "context B"), (280, "context C")):
        cx = x + 58
        s.arrow(cx, 158, cx, 181, "blue")
        s.box(x, 187, 116, 120, stroke="line", radius=7)
        s.text(cx, 211, label, 13, anchor="middle")
        s.text(cx, 235, "+ fixed answer", 13, "blue", 700, "middle")
        s.arrow(cx, 245, cx, 263)
        s.text(cx, 286, "Grade → p", 14, "teal", 700, "middle")
    s.text(210, 331, "Same recorded answer in every grading pass", 12,
           "muted", anchor="middle")
    s.save("generate-grade", theme)


def weights(theme):
    s = Figure(theme, 282, "Signed source weights",
               'Signed log-odds weights: lsof +6.1 and config -0.75. Keeping lsof '
               'multiplies the odds of 3000 by about 440; config by about one half.')
    zero, scale = 110, 43
    s.text(24, 28, 'Weights for “3000”', 14, "muted")
    s.text(24, 60, "lsof result", 15, "teal", 700)
    s.text(396, 60, "+6.1", 16, "teal", 700, "end")
    s.box(zero, 72, 6.1 * scale, 24, "teal", radius=2)
    s.text(396, 121, "odds × about 440", 14, "teal", anchor="end")
    s.text(24, 158, "config file", 15, "amber", 700)
    s.text(396, 158, "−0.75", 16, "amber", 700, "end")
    s.box(zero - .75 * scale, 170, .75 * scale, 24, "amber", radius=2)
    s.text(396, 219, "odds × about one half", 14, "amber", anchor="end")
    s.line(zero, 66, zero, 200, "muted", 1)
    s.text(zero, 219, "0", 12, "muted", anchor="middle")
    s.text(24, 260, "← opposes", 13, "amber")
    s.text(396, 260, "supports →", 13, "teal", anchor="end")
    s.text(210, 260, "log-odds", 13, "muted", anchor="middle")
    s.save("weights", theme)


def masks(theme):
    s = Figure(theme, 345, "Random masks feed a regression",
               'Each row keeps or removes sources and receives a graded score. '
               'The sampled rows feed one regression with a base and a weight per source. '
               'The default is 32 masks; the diagram shows schematic rows and scores.')
    s.math(32, 27, '<tspan font-style="italic">d</tspan>', 17)
    s.text(45, 27, "sources", 13, "muted")
    s.text(259, 27, "grade", 13, "muted", anchor="middle")
    rows = ((1, 0, 1, 1, 0), (0, 1, 1, 0, 1),
            (1, 1, 0, 0, 1), (0, 0, 1, 1, 1))
    for i, row in enumerate(rows):
        y = 44 + i * 36
        for j, kept in enumerate(row):
            s.box(24 + j * 31, y, 23, 23, "teal" if kept else "bg",
                  None if kept else "line", radius=3)
        s.arrow(185, y + 12, 220, y + 12)
        s.math(259, y + 18, sub("s", str(i + 1)), 18, "blue")
        s.line(292, y + 12, 314, y + 12)
    s.text(94, 200, "…", 19, "muted", anchor="middle")
    s.text(259, 200, "…", 19, "muted", anchor="middle")
    # All measured rows converge on the same regression.
    s.line(314, 56, 314, 164)
    s.arrow(314, 164, 314, 240)
    s.box(24, 246, 372, 65, "blue_bg", "blue", radius=7)
    s.text(40, 269, "one regression", 14, "blue", 700)
    s.math(379, 270, '<tspan font-style="italic">d</tspan> + 1 unknowns', 17, "blue", "end")
    s.math(210, 296, 'base + ' + sub("w", "1") + ' · ' + sub("x", "1")
           + ' + … + ' + sub("w", "d") + ' · ' + sub("x", "d"), 18)
    s.text(210, 332, "Filled: kept · empty: removed · default: 32 masks", 12,
           "muted", anchor="middle")
    s.save("masks", theme)


def redundancy(theme):
    s = Figure(theme, 322, "Redundant sources",
               'A file and a tool result carry the same fact. Keeping either or both '
               'leaves the score high. Removing both collapses it. Qualitative scores, '
               'not numerical measurements: this OR cannot be represented by an additive model.')
    s.text(24, 27, "file", 13, "muted")
    s.text(132, 27, "tool result", 13, "muted")
    s.text(258, 27, "score", 13, "muted")
    for y, a, b in ((44, True, True), (104, False, True),
                    (164, True, False), (224, False, False)):
        chip(s, 24, y, 93, "same fact", "teal", a)
        chip(s, 132, y, 93, "same fact", "teal", b)
        s.arrow(234, y + 15, 248, y + 15)
        s.box(258, y + 5, 138, 20, "faint", radius=2)
        s.box(258, y + 5, 138 if a or b else 5, 20,
              "teal" if a or b else "amber", radius=2)
    s.text(210, 290, "Either copy is enough; remove both and it collapses.", 12,
           "muted", anchor="middle")
    s.save("redundancy", theme)


def main():
    for theme in PALETTES:
        for draw in (ablate, generate_grade, weights, masks, redundancy):
            draw(theme)


if __name__ == "__main__":
    main()
