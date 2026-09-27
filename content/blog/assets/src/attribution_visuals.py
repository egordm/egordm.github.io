# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Rebuild the two figures of "Did Your LLM Actually Read That File?", offline.

Run: python3 content/blog/assets/src/attribution_visuals.py
Writes attribution-v-{ablate,weights}-{light,dark}.svg into assets/, in the style of luna_v_visuals.
The probabilities are the post's invented toy values; the weights are the exact least-squares fit
of their log-odds (logit), the quantity ContextCite regresses.
"""

import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ under content/: the site build copies it
sys.path.insert(0, str(Path(__file__).resolve().parent))
from luna_v_visuals import ASSETS, PALETTES, SVG  # noqa: E402

# P("3000") with (config kept, lsof kept)
PROB = {(1, 1): 0.90, (0, 1): 0.95, (1, 0): 0.02}


def logit(p):
    return math.log(p / (1 - p))


W_LSOF = logit(PROB[1, 1]) - logit(PROB[1, 0])
W_CONFIG = logit(PROB[1, 1]) - logit(PROB[0, 1])


def save(svg, name, theme):
    path = ASSETS / f"attribution-v-{name}-{theme}.svg"
    path.write_text("\n".join(svg.parts + ["</svg>"]) + "\n", encoding="utf-8")


def ablate(theme):
    s = SVG(theme, 540, "Lock the answer.",
            "Two sources: a config file saying port 8080 and an lsof tool result saying the server "
            "listens on 3000. The recorded answer 3000 stays fixed. The model assigns it 90% with "
            "both sources, 95% with the config removed, and 2% with the lsof result removed. "
            "Invented numbers.")
    s.header("01", "THE EXPERIMENT", "Lock the answer.", "Then remove sources, one at a time. Invented numbers.")
    s.box(24, 112, 180, 70, "amber_bg")
    s.text(38, 136, "CONFIG FILE", 12, "amber", 700)
    s.text(38, 162, "port = 8080", 16, weight=700)
    s.box(216, 112, 180, 70, "teal_bg")
    s.text(230, 136, "LSOF TOOL RESULT", 12, "teal", 700)
    s.text(230, 162, "listening on 3000", 16, weight=700)
    s.box(24, 196, 372, 50, stroke="line")
    s.text(40, 227, "Recorded answer:", 15, "muted")
    s.text(170, 227, "\"3000\"  (locked)", 16, weight=700)

    s.text(24, 280, "PROBABILITY THE MODEL GIVES \"3000\"", 12, "muted", 700)
    rows = [("Both sources", PROB[1, 1], "blue"), ("Config removed", PROB[0, 1], "blue"),
            ("lsof removed", PROB[1, 0], "amber")]
    bar_x, bar_w = 150, 190
    for i, (label, p, color) in enumerate(rows):
        y = 300 + i * 56
        s.text(24, y + 25, label, 15)
        s.box(bar_x, y + 8, bar_w, 24, "faint", radius=6)
        s.box(bar_x, y + 8, max(bar_w * p, 6), 24, color, radius=6)
        s.text(bar_x + bar_w + 10, y + 26, f"{round(p * 100)}%", 15, color, 700)
    s.text(24, 490, "Removing the lsof result makes the same", 15, weight=700)
    s.text(24, 512, "answer 45 times less likely.", 15, weight=700)
    save(s, "ablate", theme)


def weights(theme):
    s = SVG(theme, 400, "One signed number per source.",
            f"Fitted ContextCite weights in log-odds: the lsof result +{W_LSOF:.1f}, the config file "
            f"{W_CONFIG:.2f}. Positive supports the recorded answer, negative pushes against it.")
    s.header("02", "THE ATTRIBUTION", "One signed number per source.", "Weights on the log-odds of the recorded answer.")
    zero, scale = 150, 34  # x of zero, pixels per log-odds unit
    top, bottom = 124, 290
    for v in range(-1, 7):
        x = zero + v * scale
        s.line(x, top, x, bottom, "faint", 1)
        s.text(x, bottom + 20, f"{v:+d}" if v else "0", 12, "muted", anchor="middle")
    s.line(zero, top - 6, zero, bottom, "ink", 2)
    s.text(zero + 3 * scale, bottom + 42, "change in log-odds when the source is kept", 13, "muted", anchor="middle")

    s.text(24, 170, "lsof result", 15, weight=700)
    s.box(zero, 150, W_LSOF * scale, 30, "teal", radius=4)
    s.text(zero + W_LSOF * scale - 8, 171, f"+{W_LSOF:.1f}", 14, "bg", 700, "end")
    s.text(24, 250, "config file", 15, weight=700)
    s.box(zero + W_CONFIG * scale, 230, -W_CONFIG * scale, 30, "amber", radius=4)
    s.text(zero + 8, 251, f"{W_CONFIG:.2f}", 14, "amber", 700)

    s.text(24, 364, f"Odds of \"3000\": x{math.exp(W_LSOF):.0f} with the lsof result,", 14, "ink")
    s.text(24, 384, f"x{math.exp(W_CONFIG):.2f} with the config file.", 14, "ink")
    save(s, "weights", theme)


def main():
    for theme in PALETTES:
        ablate(theme)
        weights(theme)


if __name__ == "__main__":
    main()
