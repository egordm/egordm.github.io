# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Build this post's three figures offline at 420 CSS pixels wide.

Run: python3 content/blog/assets/src/recall_visuals.py
Writes only recall-v-{pair,curve,position}-{light,dark}.svg.
The verified N5 fact sheet supplies all measurements: 14 scored items,
one draw per item/length/position, five positions per length. The Greta
example illustrates construction and was dropped from the scored set.
Curve segments connect observations; they do not estimate untested lengths.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from luna_v_visuals import ASSETS, PALETTES, SVG  # noqa: E402

CURVE = [(1000, 1.000), (4000, 1.000), (8000, 1.000),
         (16000, 0.900), (24000, 0.757)]
WINDOW = 32768


def save(svg, name, theme):
    path = ASSETS / f"recall-v-{name}-{theme}.svg"
    path.write_text("\n".join(svg.parts + ["</svg>"]) + "\n", encoding="utf-8")


def circle(svg, x, y, radius, fill, stroke=None):
    svg.parts.append(
        f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{svg.color(fill)}"'
        + (f' stroke="{svg.color(stroke)}" stroke-width="1.5"' if stroke else '')
        + '/>'
    )


def pair(theme):
    s = SVG(theme, 290, "Finding the same name with different clues",
            "Authored example, later dropped from scoring. The meaning question "
            "connects Shakespeare to Hamlet to Greta Simeon. Adding Shakespeare "
            "to the fact allows a direct word match. Arrows show the required "
            "relations, not measured model internals.")
    s.text(24, 31, "Which character is a Shakespeare devotee?", 16, weight=700)

    s.text(24, 78, "By meaning", 14, "blue", 700)
    for x, width, label in [(24, 106, "Shakespeare"), (156, 80, "Hamlet"),
                             (262, 134, "Greta Simeon")]:
        s.box(x, 93, width, 46, "blue_bg", radius=6)
        s.text(x + width / 2, 121, label, 15, "blue", 700, "middle")
    s.arrow(136, 116, 150, 116, "blue")
    s.arrow(242, 116, 256, 116, "blue")
    s.text(137, 160, "play knowledge", 12, "muted", anchor="middle")
    s.text(272, 160, "document fact", 12, "muted", anchor="middle")

    s.text(24, 202, "Shared word", 14, "teal", 700)
    for x, width, label in [(24, 106, "Shakespeare"), (262, 134, "Greta Simeon")]:
        s.box(x, 217, width, 46, "teal_bg", radius=6)
        s.text(x + width / 2, 245, label, 15, "teal", 700, "middle")
    s.arrow(138, 240, 254, 240, "teal")
    save(s, "pair", theme)


def curve(theme):
    s = SVG(theme, 390, "Recall and the usable-length bar",
            "Qwen3-8B with thinking on, 14 scored items and five positions. "
            "Fact recovery is 100% at 1K, 4K and 8K, 90% at 16K, and 75.7% at "
            "24K. The 85% bar is crossed between the final two tested lengths. "
            "There are no measurements beyond 24K, up to the native 32,768 limit.")
    left, right, top, bottom = 52, 388, 76, 316

    def x(tokens):
        return left + (right - left) * tokens / WINDOW

    def y(share):
        return bottom - (bottom - top) * share

    s.text(24, 26, "Facts found · 70 answers per length", 14, "muted")
    s.box(x(24000), top, right - x(24000), bottom - top, "faint", radius=0)
    s.text((x(24000) + right) / 2, 230, "untested", 12, "muted", anchor="middle")
    for value in (0, 0.5, 1):
        s.line(left, y(value), right, y(value), "line", 1)
        s.text(left - 8, y(value) + 5, f"{int(value * 100)}%", 12, "muted", anchor="end")
    s.line(left, y(.85), right, y(.85), "amber", 1.5, dashed=True)
    s.text(70, y(.85) + 20, "85% bar", 13, "amber", 700)
    s.line(right, top, right, bottom, "muted", 1.5, dashed=True)
    s.text(right, 48, "Native limit", 12, "muted", anchor="end")
    s.text(right, 65, "32,768", 12, "muted", anchor="end")

    points = " L".join(f"{x(t):.2f} {y(v):.2f}" for t, v in CURVE)
    s.parts.append(f'<path d="M{points}" fill="none" stroke="{s.c["blue"]}" '
                   'stroke-width="2.5" stroke-linejoin="round"/>')
    for t, value in CURVE:
        circle(s, x(t), y(value), 4, "blue")
        s.line(x(t), bottom, x(t), bottom + 5, "muted", 1)
        s.text(x(t), bottom + 23, f"{t // 1000}K", 12, "muted", anchor="middle")
    s.text(x(16000), y(.9) - 13, "90%", 14, "blue", 700, "middle")
    s.text(x(24000) - 8, y(.757) + 24, "76%", 14, "blue", 700, "end")
    s.text(220, 367, "Document length (tokens)", 14, "muted", anchor="middle")
    save(s, "curve", theme)


def position(theme):
    s = SVG(theme, 352, "Fact position changes recall at 24K",
            "At 24K tokens, middle placement recovered 6 of 14 facts; placement "
            "90% through the document recovered 12 of 14. Equal-length document "
            "bars mark the fact position. Each circle represents an item; filled "
            "circles count recoveries and are grouped by outcome, not item identity.")
    s.text(24, 26, "Same document length: 24K tokens", 14, "muted")
    for y, label, depth, count in [(64, "Middle", .5, 6),
                                   (205, "Near the end", .9, 12)]:
        s.text(24, y, label, 16, weight=700)
        s.text(396, y, f"{count} of 14 found", 15, "blue", 700, "end")
        s.box(24, y + 15, 372, 28, "faint", "line", 4)
        marker = 24 + 372 * depth
        s.box(marker - 3, y + 12, 6, 34, "amber", radius=2)
        s.text(marker, y + 65, f"{int(depth * 100)}%", 12, "amber", anchor="middle")
        for i in range(14):
            circle(s, 35 + i * 27, y + 90, 7,
                   "blue" if i < count else "bg",
                   None if i < count else "muted")
    circle(s, 30, 332, 5, "blue")
    s.text(43, 336, "found", 12, "muted")
    circle(s, 116, 332, 5, "bg", "muted")
    s.text(129, 336, "missed", 12, "muted")
    s.box(289, 324, 4, 14, "amber", radius=1)
    s.text(302, 336, "fact position", 12, "muted")
    save(s, "position", theme)


def main():
    for theme in PALETTES:
        pair(theme)
        curve(theme)
        position(theme)


if __name__ == "__main__":
    main()
