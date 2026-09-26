# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Rebuild the two figures of "Is Half Your Context Window Just Marketing?", offline.

Run: python3 content/blog/assets/src/recall_visuals.py
Writes recall-v-{pair,curve}-{light,dark}.svg into assets/, in the style of luna_v_visuals.
Data: the Qwen3-8B recall grid (14 used needles x 5 positions per length), its item-bootstrapped
95% intervals, and the keyword twin's two tested cells (14 of 14 each on the used set).
"""

import sys
from html import escape
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ under content/: the site build copies it
sys.path.insert(0, str(Path(__file__).resolve().parent))
from luna_v_visuals import ASSETS, PALETTES, SVG  # noqa: E402

CURVE = [(1000, 1.000, 1.000, 1.000), (4000, 1.000, 1.000, 1.000), (8000, 1.000, 1.000, 1.000),
         (16000, 0.900, 0.829, 0.957), (24000, 0.757, 0.657, 0.857)]
WINDOW = 32768


def save(svg, name, theme):
    path = ASSETS / f"recall-v-{name}-{theme}.svg"
    path.write_text("\n".join(svg.parts + ["</svg>"]) + "\n", encoding="utf-8")


def raw(svg, markup):
    svg.parts.append(markup)


def pair(theme):
    s = SVG(theme, 560, "One fact, two ways to plant it.",
            "The question asks which character is a Shakespeare devotee. The original needle says "
            "Greta Simeon can recite Hamlet's soliloquy; answering needs the hop Shakespeare to "
            "Hamlet to Greta Simeon. The keyword twin says Shakespeare's Hamlet, so the question's "
            "own word sits next to the answer.")
    s.header("01", "THE PAIR", "One fact, two ways to plant it.", "A real item from the test set.")
    s.box(24, 112, 372, 70, stroke="line")
    s.text(40, 136, "THE QUESTION", 12, "muted", 700)
    s.text(40, 164, "Which character is a Shakespeare devotee?", 15.5, weight=700)

    s.box(24, 198, 372, 160, "blue_bg")
    s.text(40, 224, "ORIGINAL: NO SHARED WORD", 12, "blue", 700)
    s.text(40, 252, "Greta Simeon can recite Hamlet's soliloquy", 15)
    s.text(40, 274, "from memory at the slightest invitation.", 15)
    for x, w, label, color in [(40, 112, "Shakespeare", "blue"), (180, 84, "Hamlet", "blue"),
                               (292, 88, "Greta", "ink")]:
        s.box(x, 300, w, 38, "panel")
        s.text(x + w / 2, 324, label, 14, color, 700, "middle")
    s.arrow(156, 319, 175, 319, "blue")
    s.arrow(268, 319, 287, 319, "blue")

    s.box(24, 374, 372, 136, "teal_bg")
    s.text(40, 400, "KEYWORD TWIN: SAME FACT, SHARED WORD", 12, "teal", 700)
    s.text(40, 428, "Greta Simeon can recite Shakespeare's Hamlet", 15)
    s.text(40, 450, "soliloquy from memory ...", 15)
    s.box(40, 464, 112, 34, "panel")
    s.text(96, 486, "Shakespeare", 14, "teal", 700, "middle")
    s.arrow(160, 481, 280, 481, "teal")
    s.box(292, 464, 88, 34, "panel")
    s.text(336, 486, "Greta", 14, weight=700, anchor="middle")

    s.text(24, 538, "Same answer. Only the clue changes.", 15, weight=700)
    save(s, "pair", theme)


def curve(theme):
    s = SVG(theme, 512, "Perfect to 8K, then a slide.",
            "Share of needles the model found, by document length, averaged over five positions: "
            "100% at 1K, 4K and 8K, 90% at 16K, 76% at 24K, with 95% intervals. It crosses the 85% "
            "bar between 16K and 24K, inside the advertised 32K window. The keyword twin, tested at "
            "1K and at 24K, found 14 of 14 at both.")
    s.header("02", "THE RESULT", "Perfect to 8K, then a slide.", "Qwen3-8B, thinking on · 70 answers per length")
    left, right, top, bottom = 56, 380, 150, 390

    def x(tokens):
        return left + (right - left) * tokens / WINDOW

    def y(share):
        return bottom - (bottom - top) * share

    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.line(left, y(v), right, y(v), "faint", 1)
        s.text(left - 8, y(v) + 5, f"{int(v * 100)}%", 12, "muted", anchor="end")
    raw(s, f'<rect x="{x(16000):.1f}" y="{top - 8}" width="{x(24000) - x(16000):.1f}" '
           f'height="{bottom - top + 8}" fill="{s.c["amber"]}" fill-opacity="0.13"/>')
    s.text((x(16000) + x(24000)) / 2, top - 14, "crosses here", 12, "amber", 700, "middle")
    s.line(left, y(0.85), right, y(0.85), "amber", 1.5, dashed=True)
    s.text(left + 6, y(0.85) + 17, "85% bar", 12, "amber", 700)
    s.line(x(WINDOW), top - 30, x(WINDOW), bottom, "muted", 1.5, dashed=True)
    s.text(x(WINDOW), top - 36, "32K window", 12, "muted", 700, "end")

    c = s.c
    band = [(x(t), y(hi)) for t, _, _, hi in CURVE] + [(x(t), y(lo)) for t, _, lo, _ in reversed(CURVE)]
    raw(s, f'<path d="M{" L".join(f"{a:.1f} {b:.1f}" for a, b in band)} Z" fill="{c["blue"]}" fill-opacity="0.18"/>')
    points = " L".join(f"{x(t):.1f} {y(v):.1f}" for t, v, _, _ in CURVE)
    raw(s, f'<path d="M{points}" fill="none" stroke="{c["blue"]}" stroke-width="2.5"/>')
    for t, v, _, _ in CURVE:
        raw(s, f'<circle cx="{x(t):.1f}" cy="{y(v):.1f}" r="4" fill="{c["blue"]}"/>')
    for t in (1000, 24000):
        cx, cy = x(t), y(1.0)
        raw(s, f'<path d="M{cx:.1f} {cy - 9:.1f} L{cx + 9:.1f} {cy:.1f} L{cx:.1f} {cy + 9:.1f} '
               f'L{cx - 9:.1f} {cy:.1f} Z" fill="none" stroke="{c["teal"]}" stroke-width="2.5"/>')
    s.text(x(24000) + 10, y(0.757) + 5, "76%", 13, "blue", 700)
    s.text(x(16000) + 8, y(0.90) - 10, "90%", 13, "blue", 700)

    for t in (0, 8000, 16000, 24000):
        s.text(x(t), bottom + 20, "0" if t == 0 else f"{t // 1000}K", 12, "muted", anchor="middle")
    s.text((left + right) / 2, bottom + 42, "document length (tokens)", 13, "muted", anchor="middle")

    raw(s, f'<circle cx="32" cy="466" r="5" fill="{c["blue"]}"/>')
    s.text(44, 471, "Original: share found, 95% interval shaded", 13, "ink")
    cx, cy = 32, 492
    raw(s, f'<path d="M{cx} {cy - 7} L{cx + 7} {cy} L{cx} {cy + 7} L{cx - 7} {cy} Z" fill="none" '
           f'stroke="{c["teal"]}" stroke-width="2.5"/>')
    s.text(44, 497, "Keyword twin: 14 of 14 at both cells tested", 13, "ink")
    save(s, "curve", theme)


def main():
    for theme in PALETTES:
        pair(theme)
        curve(theme)


if __name__ == "__main__":
    main()
