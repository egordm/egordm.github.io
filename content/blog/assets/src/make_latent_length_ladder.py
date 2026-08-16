# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Render the latent-length ladder chart: access/score vs haystack material size.

Matches the style of the original latent-length-ladder.png (dark background,
blue-solid-circle / orange-dashed-square / green-dotted-triangle series, value
labels on points, legend below the axes). Differs from that original in three
ways: the x-axis is haystack material tokens (not context window), a vertical
marker line calls out the native-window limit at the deepest material point
(no shaded band), and a horizontal chance-floor line is added.

Self-contained: all data is plain lists at the top of this file, each with a
one-line provenance comment. Run with `uv run make_latent_length_ladder.py`
from any checkout; it writes the PNG next to this script's `assets/` parent,
so it renders correctly from a fresh clone.
"""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Data (plain lists; edit here to update the chart)
# ---------------------------------------------------------------------------

# (haystack material tokens, score) -- blog part 4's published numbers.
ONE_FACT_TASK: list[tuple[float, float]] = [
    (1000, 1.000),
    (4000, 1.000),
    (8000, 1.000),
    (16000, 0.900),
    (24000, 0.757),
]

# (haystack material tokens, point, lo, hi) -- N20's four cells
# b1-cell-a-anchor-4096 / b1-cell-c-mid-7558 / b1-cell-b-deep-15360 /
# b1-cell-d-tail-23161, accept-form join subset; lo/hi are the full range
# over 9 runs per needle pair.
TWO_FACT_JOIN: list[tuple[float, float, float, float]] = [
    (4096, 0.538, 0.523, 0.550),
    (7558, 0.446, 0.436, 0.459),
    (15360, 0.306, 0.298, 0.325),
    (23161, 0.066, 0.064, 0.094),
]

# (haystack material tokens, defined-only rate, strict extreme, permissive extreme).
# N20 Addendum 6 twin co-report, vault 54562518, reader code 4d5a444, four cells
# n=342 each. Set to None (and the series is simply omitted) while this slot
# awaits a registered co-report; currently filled with the landed twin numbers.
ONE_HOP_CONTROL: list[tuple[float, float, float, float]] | None = [
    (4096, 0.7207, 0.7018, 0.7281),
    (7558, 0.5215, 0.4971, 0.5439),
    (15360, 0.2692, 0.2661, 0.2778),
    (23161, 0.0917, 0.0906, 0.1023),
]

NATIVE_WINDOW_X_TOKENS = 23161
LABEL_COLLISION_ZONE_TOKENS = 2000  # x-distance within which a value label nudges aside
NATIVE_WINDOW_LABEL = "native-window limit: deepest material at 32,768 with the reasoning budget"
NATIVE_WINDOW_LABEL_X_PAD_FRACTION = 0.012  # clearance from the line so tail labels don't cross it
CHANCE_FLOOR_Y = 0.1619
CHANCE_FLOOR_LABEL = "chance floor (2-fact join)"
ERROR_BAR_FOOTNOTE = "error bars: full range over 9 runs per needle pair"

X_AXIS_MAX_TOKENS = (
    27_000  # padded past the rightmost point (24,000) so its label sits clear of the edge
)
X_TICKS_TOKENS = [0, 4_000, 8_000, 12_000, 16_000, 20_000, 24_000]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "latent-length-ladder.png"

# ---------------------------------------------------------------------------
# Style (matched to the original chart by pixel sampling)
# ---------------------------------------------------------------------------

BACKGROUND = "#1e1e1e"
GRID_COLOR = "#2d2d31"
SPINE_COLOR = "#55555c"
TEXT_COLOR = "#e4e4e6"
MUTED_TEXT_COLOR = "#a2a2a8"
BLUE = "#6db1f7"
ORANGE = "#f0a04b"
GREEN = "#3ecf9b"

LINE_WIDTH = 3.0
MARKER_SIZE = 11.0
VALUE_LABEL_FONTSIZE = 15.0
AXIS_LABEL_FONTSIZE = 20.0
TICK_LABEL_FONTSIZE = 16.0
LEGEND_FONTSIZE = 15.0
ANNOTATION_FONTSIZE = 13.0
NATIVE_WINDOW_LABEL_FONTSIZE = 10.5
FOOTNOTE_FONTSIZE = 11.0
VALUE_LABEL_LIGHTEN_AMOUNT = 0.35  # blend toward white; keeps the series hue but stays legible


def _lighten(hex_color: str, amount: float) -> str:
    """Blend a hex color toward white by `amount` (0 keeps it unchanged, 1 gives white).

    Derives a series' value-label color from its line color, so a label is attachable to
    its series by color alone (not just its above/below position) while staying readable
    on the dark background.
    """
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(channel + (255 - channel) * amount) for channel in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _format_tokens(value: float, _pos: int) -> str:
    """Format an x-axis tick as '0' or e.g. '4K'."""
    if value == 0:
        return "0"
    return f"{value / 1000:g}K"


def _format_score(value: float) -> str:
    """Format a score to 3 decimals with round-half-up (matches the blog prose's rounding).

    `f"{value:.3f}"` rounds via the value's binary float representation, which rounds
    0.5215 down to "0.521" instead of the decimal round-half-up "0.522" the prose uses.
    """
    return str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _value_label_offset(x: float) -> tuple[float, str]:
    """Nudge a value label sideways when its point sits close to the native-window line.

    Points within the collision zone of the vertical marker line would otherwise have
    their value label overlap the line's rotated caption; push those labels left or
    right of the marker instead of centering them on it.
    """
    distance = x - NATIVE_WINDOW_X_TOKENS
    if abs(distance) >= LABEL_COLLISION_ZONE_TOKENS:
        return 0.0, "center"
    return (-14.0, "right") if distance <= 0 else (14.0, "left")


def _plot_line_series(
    ax: Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    """Plot a plain (x, y) series with value labels above each point, colored to match it."""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    ax.plot(
        xs,
        ys,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=3,
    )
    label_color = _lighten(color, VALUE_LABEL_LIGHTEN_AMOUNT)
    for x, y in zip(xs, ys, strict=True):
        dx, ha = _value_label_offset(x)
        ax.annotate(
            _format_score(y),
            xy=(x, y),
            xytext=(dx, 12),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            color=label_color,
            fontsize=VALUE_LABEL_FONTSIZE,
            zorder=4,
        )


def _plot_error_bar_series(  # noqa: PLR0913 -- five keyword-only style params + the ax it draws on
    ax: Axes,
    points: list[tuple[float, float, float, float]],
    *,
    color: str,
    marker: str,
    linestyle: str,
    label_side: Literal["above", "below"],
) -> None:
    """Plot an (x, point, lo, hi) series with asymmetric error bars and value labels.

    `label_side` places every label consistently on one side of its point (above the
    high whisker, or below the low whisker): the two error-bar series (2-fact join,
    1-hop control) share all four x positions, so pinning one above and the other below
    keeps their labels apart at every shared x rather than only where they happen to differ.
    """
    xs = [x for x, _, _, _ in points]
    ys = [y for _, y, _, _ in points]
    los = [y - lo for _, y, lo, _ in points]
    his = [hi - y for _, y, _, hi in points]
    ax.plot(
        xs,
        ys,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=3,
    )
    ax.errorbar(
        xs,
        ys,
        yerr=[los, his],
        fmt="none",
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        alpha=0.7,
        zorder=2,
    )
    dy = 10.0 if label_side == "above" else -10.0
    va = "bottom" if label_side == "above" else "top"
    label_color = _lighten(color, VALUE_LABEL_LIGHTEN_AMOUNT)
    for x, y, lo, hi in points:
        anchor_y = hi if label_side == "above" else lo
        dx, ha = _value_label_offset(x)
        ax.annotate(
            _format_score(y),
            xy=(x, anchor_y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=label_color,
            fontsize=VALUE_LABEL_FONTSIZE,
            zorder=4,
        )


def render_ladder(output_path: Path) -> None:
    """Render the latent-length ladder chart to `output_path`."""
    fig, ax = plt.subplots(figsize=(10.96, 8.83), dpi=100)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    legend_handles: list[Line2D] = []

    _plot_line_series(ax, ONE_FACT_TASK, color=BLUE, marker="o", linestyle="-")
    legend_handles.append(
        Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE * 0.7,
            label="1-fact task, depth-aggregated (Qwen3-8B, BF16)",
        ),
    )

    _plot_error_bar_series(
        ax,
        TWO_FACT_JOIN,
        color=ORANGE,
        marker="s",
        linestyle="--",
        label_side="below",
    )
    legend_handles.append(
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="s",
            linestyle="--",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE * 0.7,
            label="2-fact join, accept-form (Qwen3-8B, BF16, thinking on)",
        ),
    )

    has_error_bars = True  # TWO_FACT_JOIN always carries error bars.

    if ONE_HOP_CONTROL is not None:
        _plot_error_bar_series(
            ax,
            ONE_HOP_CONTROL,
            color=GREEN,
            marker="^",
            linestyle=(0, (1, 1)),
            label_side="above",
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color=GREEN,
                marker="^",
                linestyle=(0, (1, 1)),
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE * 0.7,
                label="1-hop control, accept-form (Qwen3-8B, BF16, thinking on)",
            ),
        )
        has_error_bars = True

    ax.axvline(
        x=NATIVE_WINDOW_X_TOKENS,
        color=MUTED_TEXT_COLOR,
        linestyle=":",
        linewidth=1.5,
        zorder=1,
    )
    ax.text(
        NATIVE_WINDOW_X_TOKENS + X_AXIS_MAX_TOKENS * NATIVE_WINDOW_LABEL_X_PAD_FRACTION,
        -0.015,
        NATIVE_WINDOW_LABEL,
        rotation=90,
        ha="left",
        va="bottom",
        color=MUTED_TEXT_COLOR,
        fontsize=NATIVE_WINDOW_LABEL_FONTSIZE,
        zorder=2,
        clip_on=False,
    )

    ax.axhline(y=CHANCE_FLOOR_Y, color=MUTED_TEXT_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(
        X_AXIS_MAX_TOKENS * 0.03,
        CHANCE_FLOOR_Y + 0.015,
        CHANCE_FLOOR_LABEL,
        ha="left",
        va="bottom",
        color=MUTED_TEXT_COLOR,
        fontsize=ANNOTATION_FONTSIZE,
        zorder=2,
    )

    ax.set_xlim(0, X_AXIS_MAX_TOKENS)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(X_TICKS_TOKENS)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_tokens))
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

    ax.set_xlabel("haystack material (tokens)", color=TEXT_COLOR, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("access / score", color=TEXT_COLOR, fontsize=AXIS_LABEL_FONTSIZE)

    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=TICK_LABEL_FONTSIZE, length=4)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(SPINE_COLOR)

    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1.0, zorder=0)  # noqa: FBT003 -- matplotlib's own Axis.grid signature
    ax.xaxis.grid(False)  # noqa: FBT003 -- matplotlib's own Axis.grid signature
    ax.set_axisbelow(True)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.135),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        labelcolor=TEXT_COLOR,
        handlelength=2.6,
        borderaxespad=0.0,
    )

    if has_error_bars:
        footnote_y = -0.135 - 0.045 * len(legend_handles) - 0.05
        ax.text(
            0.0,
            footnote_y,
            ERROR_BAR_FOOTNOTE,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=MUTED_TEXT_COLOR,
            fontsize=FOOTNOTE_FONTSIZE,
            fontstyle="italic",
        )

    fig.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.30)
    fig.savefig(output_path, facecolor=BACKGROUND)
    plt.close(fig)


if __name__ == "__main__":
    render_ladder(OUTPUT_PATH)
