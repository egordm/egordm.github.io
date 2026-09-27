# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.11"]
# ///
"""Render Luna recall from the adjacent JSON, without fetching any data.

Run: python3 content/blog/assets/src/make_luna_context_curve.py
Writes opaque light/dark SVGs and PNGs into assets/, plus 420px QA previews.
Both variants remain readable on either page background; no theme CSS is needed.
Shading shows pointwise 95% Wilson binomial intervals, not variability across
books, depths, or runs. Straight segments connect observations only; no fits or
extrapolation. K and M are decimal token units, as in the source JSON.
"""

import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

ASSETS = Path(__file__).resolve().parent.parent
# Keep temporary caches inside assets and remove them on exit.
_cache = TemporaryDirectory(prefix="luna-mpl-")  # outside content/: the Quartz watcher copies anything there
os.environ["MPLCONFIGDIR"] = _cache.name
os.environ["XDG_CACHE_HOME"] = _cache.name

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

PALETTES = {
    "dark": dict(bg="#1e1e1e", text="#efefeb", muted="#bdbdb8", grid="#41413f",
                 literal="#7bd7c8", off="#ffa96e", high="#a6b8ff"),
    "light": dict(bg="#ffffff", text="#252725", muted="#595e5b", grid="#dedfd9",
                  literal="#00796b", off="#b34909", high="#4557ba"),
}


def wilson(correct: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval in percentage points."""
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denom
    return 100 * max(0, center - half), 100 * min(1, center + half)


def read_data() -> dict:
    data = json.loads(Path(__file__).with_name("luna_context_curve.json").read_text())
    for name in ("latent_reasoning_off", "literal_reasoning_off", "latent_reasoning_high"):
        previous = 0
        for tokens, correct, total in data[name]["points"]:
            if not (tokens > previous and 0 <= correct <= total and total > 0):
                raise ValueError(f"Invalid or unsorted counts in {name}")
            previous = tokens
    return data


def score(data: dict, series: str, tokens: int) -> float:
    return next(100 * k / n for x, k, n in data[series]["points"] if x == tokens)


def render(data: dict, theme: str) -> None:
    c = PALETTES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "svg.fonttype": "path", "svg.hashsalt": "luna-context-curve"})
    fig, ax = plt.subplots(figsize=(7, 8.7), facecolor=c["bg"])
    fig.subplots_adjust(left=.13, right=.955, bottom=.29, top=.80)
    ax.set_facecolor(c["bg"])
    ax.set_xscale("log")
    ax.set_xlim(210, 1_100_000)
    ax.set_ylim(0, 103)
    codex = data["markers"]["codex_default_window"]
    advertised = data["markers"]["advertised_window"]
    ax.axvspan(codex, advertised, color=c["muted"], alpha=.075, linewidth=0)
    for value in (codex, advertised):
        ax.axvline(value, color=c["muted"], linewidth=1.2, linestyle=(0, (3, 3)), zorder=1)

    for name, key, marker, linestyle in (
        ("latent_reasoning_off", "off", "o", "-"),
        ("latent_reasoning_high", "high", "D", "--"),
        ("literal_reasoning_off", "literal", "s", "-"),
    ):
        points = data[name]["points"]
        xs = [x for x, _, _ in points]
        ys = [100 * k / n for _, k, n in points]
        intervals = [wilson(k, n) for _, k, n in points]
        ax.fill_between(xs, [a for a, _ in intervals], [b for _, b in intervals],
                        color=c[key], alpha=.13, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=c[key], marker=marker, linestyle=linestyle,
                linewidth=2.3, markersize=5.5, markeredgecolor=c["bg"],
                markeredgewidth=.6, zorder=3)

    def label(text, xy, xytext, color, ha="left"):
        ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=15,
                    ha=ha, va="bottom", linespacing=1.25,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.1,
                                    connectionstyle="angle,angleA=0,angleB=90,rad=4"),
                    zorder=5)

    label("Shared words\nreasoning off", (8000, score(data, "literal_reasoning_off", 8000)),
          (250, 110), c["literal"])
    label(f"{codex / 1000:g}K\nCodex default", (codex, 101), (codex, 110), c["text"], "right")
    label(f"{advertised / 1_000_000:g}M advertised", (advertised, 101),
          (advertised, 128), c["muted"], "right")
    label("By meaning\nreasoning off", (224000, score(data, "latent_reasoning_off", 224000)),
          (1200, 39), c["off"])
    label("By meaning\nreasoning high", (512000, score(data, "latent_reasoning_high", 512000)),
          (20000, 10), c["high"])

    ax.set_xticks([250, 1000, 16000, 128000, 1000000], ["250", "1K", "16K", "128K", "1M"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
    ax.tick_params(colors=c["muted"], labelsize=14, length=0, pad=8)
    ax.set_xlabel("Context length · tokens (log scale)", color=c["text"], fontsize=15, labelpad=14)
    ax.text(-.105, 1.02, "% correct", transform=ax.transAxes, color=c["text"], fontsize=15)
    ax.yaxis.grid(True, color=c["grid"], linewidth=.8)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(c["grid"])

    at128 = score(data, "latent_reasoning_high", 128000)
    at512 = score(data, "latent_reasoning_high", 512000)
    fig.text(.055, .18, "Meaning fades around 64K–128K.",
             color=c["text"], fontsize=14)
    fig.text(.055, .135, f"Reasoning high: {at128:.0f}% at 128K → {at512:.0f}% at 512K",
             color=c["high"], fontsize=14)
    totals = [n for name in ("latent_reasoning_off", "latent_reasoning_high", "literal_reasoning_off")
              for _, _, n in data[name]["points"]]
    fig.text(.055, .09, f"Bands: 95% Wilson intervals · {min(totals)}–{max(totals)} calls / point",
             color=c["muted"], fontsize=13)
    fig.text(.055, .045, "Shared-word control ends at 224K; no data at 1M.",
             color=c["muted"], fontsize=13)
    metadata = {"Title": "GPT-6 Luna recall by context length",
                "Description": data["source"] + " Pointwise 95% Wilson binomial intervals.",
                "Date": None}
    stem = ASSETS / f"luna-context-curve-{theme}"
    fig.savefig(stem.with_suffix(".svg"), facecolor=c["bg"], metadata=metadata)
    fig.savefig(stem.with_suffix(".png"), dpi=120, facecolor=c["bg"])
    fig.savefig(ASSETS / f"luna-context-curve-{theme}-420.png", dpi=60, facecolor=c["bg"])
    plt.close(fig)


if __name__ == "__main__":
    data = read_data()
    for theme in PALETTES:
        render(data, theme)
