# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Rebuild the plateau figures offline with standard-library Python.

Run: python3 content/blog/assets/src/plateau_visuals.py
Writes plateau-v-{lottery,resumes,screens,inside}-{light,dark}.svg
and plateau-v-{graded-halves,muon-scratch,playbook}-{light,dark}.svg
and plateau-v-{forced-read,two-factors}-{light,dark}.svg
beside this src directory.
SVG helpers and palettes follow luna_v_visuals.py; text stays selectable.
"""

import json
from html import escape
from math import exp
from pathlib import Path
from statistics import median
from typing import Literal, TypedDict, cast

ASSETS = Path(__file__).resolve().parent.parent
PALETTES = {
    "light": dict(
        bg="#ffffff",
        panel="#ffffff",
        ink="#252725",
        muted="#595e5b",
        line="#d8dcd7",
        teal="#00796b",
        teal_bg="#e4f3ee",
        blue="#4557ba",
        blue_bg="#e9edfc",
        amber="#a8460c",
        amber_bg="#faebdf",
        faint="#eeeFEB",
    ),
    "dark": dict(
        bg="#1e1e1e",
        panel="#292b2a",
        ink="#efefeb",
        muted="#bdbdb8",
        line="#4c504d",
        teal="#7bd7c8",
        teal_bg="#253e38",
        blue="#a6b8ff",
        blue_bg="#30374f",
        amber="#ffa96e",
        amber_bg="#443226",
        faint="#323532",
    ),
}


class RunData(TypedDict):
    """Recorded observations and the takeoff update for one run."""

    step: list[int]
    accuracy: list[float]
    takeoff: int


class CurveData(TypedDict):
    """Fields consumed from the supplied curve data."""

    copy_level: float
    runs: dict[str, RunData]


class ResumeRunData(TypedDict):
    """Recorded observations and takeoff for one resumed training run."""

    arm: Literal["off", "rep2"]
    data_seed: int
    takeoff: int
    update: list[int]
    accuracy: list[float]


class ResumeData(TypedDict):
    """Fields consumed from the supplied resume data."""

    copy_level: float
    runs: list[ResumeRunData]


class GradedEvaluation(TypedDict):
    """Pooled accuracy at one recorded update."""

    update: int
    restatement: float


class GradedRun(TypedDict):
    """One from-scratch recipe and seed."""

    recipe: str
    seed: int
    evaluations: list[GradedEvaluation]


class GradedRead(TypedDict):
    """Two logit reads and accuracy on the same checkpoint."""

    forced_lead: float
    stock_lead: float
    accuracy: float


class GradedCheckpoint(TypedDict):
    """Restatement reads at one saved update."""

    update: int
    restatement: GradedRead


class GradedSeed(TypedDict):
    """Checkpoint reads for one AdamW baseline."""

    seed: int
    checkpoints: list[GradedCheckpoint]


class GradedData(TypedDict):
    """Fields consumed from the supplied graded data."""

    warm_updates: int
    runs: list[GradedRun]
    two_half: list[GradedSeed]


class SVG:
    """Small SVG drawing helpers shared by the themed figures."""

    def __init__(self, theme: str, height: int, title: str, description: str) -> None:
        self.c = PALETTES[theme]
        self.parts = [
            (
                '<svg xmlns="http://www.w3.org/2000/svg" width="420" '
                f'height="{height}" '
                f'viewBox="0 0 420 {height}" role="img" aria-labelledby="title desc">'
            ),
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
        ]
        self.box(0, 0, 420, height, "bg", radius=0)

    def color(self, name: str) -> str:
        return self.c.get(name, name)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "panel",
        stroke: str | None = None,
        radius: float = 12,
    ) -> None:
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{radius}" fill="{self.color(fill)}"'
            + (f' stroke="{self.color(stroke)}"' if stroke is not None else "")
            + "/>",
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 15,
        color: str = "ink",
        weight: int = 400,
        anchor: str = "start",
    ) -> None:
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{self.color(color)}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>',
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "line",
        width: float = 1.5,
        *,
        dashed: bool = False,
    ) -> None:
        self.parts.append(
            f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" '
            f'stroke="{self.color(color)}" stroke-width="{width}" '
            'stroke-linecap="round"'
            + (' stroke-dasharray="4 4"' if dashed else "")
            + "/>",
        )

    def save(self, name: str, theme: str) -> None:
        path = ASSETS / f"plateau-v-{name}-{theme}.svg"
        path.write_text("\n".join([*self.parts, "</svg>"]) + "\n", encoding="utf-8")


def lottery(theme: str) -> None:
    """Connect every recorded observation and mark each run's takeoff."""
    data = cast(
        CurveData,
        json.loads(
            Path(__file__).with_name("plateau_curves.json").read_text(encoding="utf-8"),
        ),
    )
    s = SVG(
        theme,
        460,
        "Two training runs leave the plateau at different times.",
        "Accuracy over training updates for two runs with the same initialization "
        "and training rows: both linger near the copy level of 0.1003 before run 1 "
        "takes off at 14,000 updates and run 2 at 21,750, with all recorded "
        "observations connected through each run's end.",
    )

    def x(step: int) -> float:
        return 48 + 320 * step / 24000

    def y(accuracy: float) -> float:
        return 368 - 320 * accuracy

    s.text(16, 29, "accuracy", size=12, color="muted")
    for tick in (0, 0.5, 1):
        yy = y(tick)
        s.line(48, yy, 368, yy, width=1)
        s.text(39, yy + 4, f"{tick:g}", size=12, color="muted", anchor="end")
    s.line(48, 48, 48, 368, width=1)
    for tick in range(0, 24001, 4000):
        xx = x(tick)
        s.line(xx, 368, xx, 373, color="muted", width=1)
        label = f"{tick // 1000}k" if tick else "0"
        s.text(xx, 391, label, size=12, color="muted", anchor="middle")
    s.text(208, 416, "updates", size=13, color="muted", anchor="middle")

    copy_y = y(data["copy_level"])
    s.line(48, copy_y, 368, copy_y, color="muted", width=1, dashed=True)
    s.text(60, copy_y - 8, "copy level", size=12, color="muted")

    for name, color, label in (
        ("first", "teal", "run 1"),
        ("second", "amber", "run 2"),
    ):
        run = data["runs"][name]
        points = list(zip(run["step"], run["accuracy"], strict=True))
        path = "M" + " L".join(f"{x(step)} {y(accuracy)}" for step, accuracy in points)
        s.parts.append(
            f'<path id="{name}-curve" d="{path}" fill="none" '
            f'stroke="{s.color(color)}" stroke-width="2" stroke-linejoin="round"/>',
        )
        takeoff_index = run["step"].index(run["takeoff"])
        marker_x = x(run["takeoff"])
        marker_y = y(run["accuracy"][takeoff_index])
        s.parts.append(
            f'<circle id="{name}-takeoff" cx="{marker_x}" cy="{marker_y}" '
            f'r="3.5" fill="{s.color(color)}"/>',
        )
        last_step, last_accuracy = points[-1]
        s.text(x(last_step) + 8, y(last_accuracy) + 4, label, size=12, color=color)
    s.save("lottery", theme)


def resumes(theme: str) -> None:
    """Connect all six resume traces and mark their recorded takeoffs."""
    data = cast(
        ResumeData,  # noqa: TC006 - Workspace convention keeps runtime types explicit.
        json.loads(
            Path(__file__)
            .with_name("plateau_resumes.json")
            .read_text(encoding="utf-8"),
        ),
    )
    s = SVG(
        theme,
        460,
        "Repeated keys bring all three branches out of the plateau before control.",
        "Accuracy after branching from one plateau checkpoint with six fresh data orders: "
        "three control branches take off at 9,000 to 17,500 updates, while three "
        "repeated-key branches take off at 750 to 2,500. Dots mark takeoff; all "
        "recorded observations are connected through each branch's end. "
        "The dashed line marks the copy level of 0.1003.",
    )

    def x(update: int) -> float:
        return 48 + 320 * update / 20000

    def y(accuracy: float) -> float:
        return 368 - 320 * accuracy

    s.text(16, 29, "accuracy", size=12, color="muted")
    for tick in (0, 0.5, 1):
        yy = y(tick)
        s.line(48, yy, 368, yy, width=1)
        s.text(39, yy + 4, f"{tick:g}", size=12, color="muted", anchor="end")
    s.line(48, 48, 48, 368, width=1)
    for tick in range(0, 20001, 4000):
        xx = x(tick)
        s.line(xx, 368, xx, 373, color="muted", width=1)
        label = f"{tick // 1000}k" if tick else "0"
        s.text(xx, 391, label, size=12, color="muted", anchor="middle")
    s.text(208, 416, "updates after branching", size=13, color="muted", anchor="middle")

    copy_y = y(data["copy_level"])
    s.line(48, copy_y, 368, copy_y, color="muted", width=1, dashed=True)
    s.text(276, copy_y - 8, "copy level", size=12, color="muted")

    for run in data["runs"]:
        color = "muted" if run["arm"] == "off" else "teal"
        width = 1.8 if run["arm"] == "off" else 2
        name = f"{run['arm']}-{run['data_seed']}"
        points = zip(run["update"], run["accuracy"], strict=True)
        path = "M" + " L".join(
            f"{x(update)} {y(accuracy)}" for update, accuracy in points
        )
        s.parts.append(
            f'<path id="{name}-curve" d="{path}" fill="none" '
            f'stroke="{s.color(color)}" stroke-width="{width}" '
            'stroke-linejoin="round"/>',
        )
        takeoff_index = run["update"].index(run["takeoff"])
        marker_x = x(run["takeoff"])
        marker_y = y(run["accuracy"][takeoff_index])
        s.parts.append(
            f'<circle id="{name}-takeoff" cx="{marker_x}" cy="{marker_y}" '
            f'r="3.5" fill="{s.color(color)}"/>',
        )

    s.text(112, 141, "repeated keys", size=12, color="teal")
    s.text(352, 155, "control", size=12, color="muted")
    s.save("resumes", theme)


def screens(theme: str) -> None:
    """Compare each lever's takeoffs with its own stock resume range."""

    class ScreenRow(TypedDict):
        label: str
        verdict: Literal["pass", "fail"]
        takeoff: list[int | None]
        reference: list[int | None]

    class ScreenData(TypedDict):
        cap: int
        rows: list[ScreenRow]

    data = cast(
        ScreenData,  # noqa: TC006 - Workspace convention keeps runtime types explicit.
        json.loads(
            Path(__file__).with_name("plateau_screens.json").read_text(encoding="utf-8"),
        ),
    )
    s = SVG(
        theme,
        420,
        "Lever takeoffs beside their own control branches.",
        "Seventeen tests in source order. Each grey bar spans the fastest to "
        "slowest control takeoff for that test; an open end reaches the 27,500 "
        "update cap when control is censored. Three dots show each lever's takeoffs, "
        "teal for the supplied pass verdict and warm for fail. Hollow dots with "
        "right chevrons mark branches that never took off before the cap. Nearby "
        "dots are separated vertically. Verdicts follow the source data.",
    )
    row_height = 19
    minimum_dot_separation = 6.5
    last_pass = max(i for i, row in enumerate(data["rows"]) if row["verdict"] == "pass")

    def x(update: int) -> float:
        return 166 + 228 * update / data["cap"]

    for tick in range(0, 25001, 5000):
        xx = x(tick)
        s.line(xx, 11, xx, 343, color="faint", width=0.7)
        s.line(xx, 343, xx, 347, color="muted", width=0.7)
        s.text(
            xx,
            360,
            f"{tick // 1000}k" if tick else "0",
            size=10,
            color="muted",
            anchor="middle",
        )

    for i, row in enumerate(data["rows"]):
        yy = 20 + row_height * i + (7 if i > last_pass else 0)
        color = s.color("teal" if row["verdict"] == "pass" else "amber")
        s.parts.append(f'<g id="test-{i + 1}" data-verdict="{row["verdict"]}">')
        s.text(16, yy + 3.5, row["label"], size=10, color="muted")
        stock = [data["cap"] if value is None else value for value in row["reference"]]
        left, right = x(min(stock)), x(max(stock))
        # A stroked, unfilled range keeps a censored right end visibly open.
        censored = None in row["reference"]
        s.parts.append(
            f'<path class="control-range" d="M{right} {yy - 3}H{left}V{yy + 3}'
            f'H{right}' + ("" if censored else "Z") + '" '
            f'fill="{s.color("bg") if censored else s.color("line")}" '
            f'stroke="{s.color("line")}" stroke-width="1"/>',
        )

        positions = sorted(
            (x(data["cap"] if value is None else value), j)
            for j, value in enumerate(row["takeoff"])
        )
        clusters: list[list[tuple[float, int]]] = []
        for position in positions:
            if (
                not clusters
                or position[0] - clusters[-1][-1][0] >= minimum_dot_separation
            ):
                clusters.append([])
            clusters[-1].append(position)
        for cluster in clusters:
            for offset, (xx, j) in enumerate(cluster):
                dot_y = yy + 6 * (offset - (len(cluster) - 1) / 2)
                censored = row["takeoff"][j] is None
                s.parts.append(
                    f'<circle id="test-{i + 1}-takeoff-{j + 1}" '
                    f'cx="{xx}" cy="{dot_y}" r="2.4" '
                    f'fill="{s.color("bg") if censored else color}"'
                    + (f' stroke="{color}" stroke-width="1.2"' if censored else "")
                    + "/>",
                )
                if censored:
                    s.parts.append(
                        f'<path d="M{xx + 5} {dot_y - 2}l2.5 2 -2.5 2" '
                        f'fill="none" stroke="{color}" stroke-width="1" '
                        'stroke-linecap="round" stroke-linejoin="round"/>',
                    )
        s.parts.append("</g>")

    s.text(
        280,
        383,
        "updates to takeoff after branching",
        size=11,
        color="muted",
        anchor="middle",
    )
    s.box(166, 401, 16, 6, fill="line", radius=0)
    s.text(188, 408, "control branches, same test", size=10, color="muted")
    s.save("screens", theme)


def inside(theme: str) -> None:
    """Align eight seeds' attention and copy leads by plateau progress."""

    class Evaluation(TypedDict):
        update: int
        attention_to_value: float

    class Checkpoint(TypedDict):
        update: int
        forced_lead: float
        stock_lead: float

    class Seed(TypedDict):
        P: int
        T: int
        evaluations: list[Evaluation]
        checkpoints: list[Checkpoint]

    class InsideData(TypedDict):
        seeds: list[Seed]

    data = cast(
        InsideData,  # noqa: TC006 - Workspace convention keeps runtime types explicit.
        json.loads(
            Path(__file__).with_name("plateau_inside.json").read_text(encoding="utf-8"),
        ),
    )
    s = SVG(
        theme,
        330,
        "Copy lead grows while attention remains nearly flat through the plateau.",
        "Eight seeds aligned by plateau progress, (update - P) / (T - P), "
        "from -0.15 to 1. The shaded interval runs from plateau start to takeoff. "
        "Above, warm lines show attention to the correct value as a log ratio; "
        "zero means no preference. Below, teal lines show the answer's lead in "
        "logits with attention forced onto the correct value; dashed grey lines "
        "show the unforced lead. Only recorded points within the window are "
        "connected. The lower axis starts at zero, clipping small negative leads.",
    )
    attention: list[list[tuple[float, float]]] = []
    forced: list[list[tuple[float, float]]] = []
    stock: list[list[tuple[float, float]]] = []
    progress_min = -0.15
    for seed in data["seeds"]:
        duration = seed["T"] - seed["P"]
        attention.append(
            [
                (progress, row["attention_to_value"])
                for row in seed["evaluations"]
                if progress_min
                <= (progress := (row["update"] - seed["P"]) / duration)
                <= 1
            ],
        )
        checkpoints = [
            (progress, row)
            for row in seed["checkpoints"]
            if progress_min <= (progress := (row["update"] - seed["P"]) / duration) <= 1
        ]
        forced.append([(progress, row["forced_lead"]) for progress, row in checkpoints])
        stock.append([(progress, row["stock_lead"]) for progress, row in checkpoints])

    attention_min = min(value for trace in attention for _, value in trace)
    attention_max = max(value for trace in attention for _, value in trace)
    lead_max = max(value for trace in [*forced, *stock] for _, value in trace)
    panel_height = 126

    def x(progress: float) -> float:
        return 48 + 320 * (progress - progress_min) / (1 - progress_min)

    for name, top, minimum, maximum, label, ticks in (
        (
            "attention",
            16,
            attention_min,
            attention_max,
            "attention to the right value",
            (0, 2),
        ),
        ("copy", 168, 0, lead_max, "copy lead, attention forced", (0, 4, 8)),
    ):
        bottom = top + panel_height
        scale = panel_height / (maximum - minimum)
        s.parts.append(
            f'<defs><clipPath id="{name}-clip"><rect x="48" y="{top}" '
            f'width="320" height="{panel_height}"/></clipPath></defs>',
        )
        s.parts.append(
            f'<rect x="{x(0)}" y="{top}" width="{x(1) - x(0)}" '
            f'height="{panel_height}" fill="{s.color("faint")}" opacity="0.55"/>',
        )
        for tick in ticks:
            yy = bottom - (tick - minimum) * scale
            s.line(48, yy, 368, yy, color="line", width=0.6)
            s.text(39, yy + 3.5, f"{tick:g}", size=10, color="muted", anchor="end")
        s.line(48, top, 48, bottom, width=0.7)
        s.line(x(1), top, x(1), bottom, color="muted", width=0.7)

        series = (
            (("attention", attention, "amber", False),)
            if name == "attention"
            else (("stock", stock, "muted", True), ("forced", forced, "teal", False))
        )
        s.parts.append(f'<g clip-path="url(#{name}-clip)">')
        for kind, traces, color, dashed in series:
            for index, trace in enumerate(traces):
                path = "M" + " L".join(
                    f"{x(progress):.3f} {bottom - (value - minimum) * scale:.3f}"
                    for progress, value in trace
                )
                s.parts.append(
                    f'<path id="{kind}-{index + 1}" class="{kind}-curve" '
                    f'd="{path}" fill="none" stroke="{s.color(color)}" '
                    'stroke-width="1.1" stroke-opacity="0.55" '
                    'stroke-linecap="round" stroke-linejoin="round"'
                    + (' stroke-dasharray="3 3"' if dashed else "")
                    + "/>",
                )
        s.parts.append("</g>")
        s.text(58, top + 16, label, size=12, color="muted")

    s.parts.append(
        f'<line x1="210" y1="279" x2="224" y2="279" '
        f'stroke="{s.color("muted")}" stroke-width="1.1" '
        'stroke-dasharray="3 3" stroke-opacity="0.55"/>',
    )
    s.text(230, 282, "unforced", size=10, color="muted")
    for progress, label, anchor in (
        (0, "plateau starts", "middle"),
        (1, "takeoff", "end"),
    ):
        s.line(x(progress), 294, x(progress), 299, color="muted", width=0.7)
        s.text(x(progress), 315, label, size=10, color="muted", anchor=anchor)
    s.save("inside", theme)


def graded_halves(theme: str) -> None:
    """Show both reads only on post-warm restatement plateau checkpoints."""
    window_end = 56000
    plateau_limit = 0.45
    data = cast(
        GradedData,  # noqa: TC006 - Workspace convention keeps runtime types explicit.
        json.loads(
            Path(__file__).with_name("plateau_graded.json").read_text(encoding="utf-8"),
        ),
    )
    s = SVG(
        theme,
        330,
        "The copy half is ready while the graded match lags.",
        "Three AdamW 1e-4 seeds after the 20,000-update verbatim warm stage. "
        "Teal lines show restatement forced lead; dashed grey lines show stock "
        "lead. Only checkpoints with restatement accuracy below 0.45 are drawn, "
        "ending with dots at each seed's last plateau checkpoint. The vertical "
        "axis starts at zero. The faint band marks 1 to 3 logits, where MQAR "
        "took off; graded forced leads can pass this range while stock stays flat.",
    )
    traces = [
        (
            seed["seed"],
            [
                row
                for row in seed["checkpoints"]
                if data["warm_updates"] < row["update"] <= window_end
                and row["restatement"]["accuracy"] < plateau_limit
            ],
        )
        for seed in data["two_half"]
    ]
    maximum = max(
        row["restatement"][kind]
        for _, trace in traces
        for row in trace
        for kind in ("forced_lead", "stock_lead")
    )

    def x(update: int) -> float:
        return 48 + 320 * (update - 20000) / 36000

    def y(lead: float) -> float:
        return 254 - 206 * lead / maximum

    s.box(48, y(3), 320, y(1) - y(3), fill="faint", radius=0)
    s.text(16, 29, "lead (logits)", size=12, color="muted")
    for tick in (0, 2, 4, 6):
        yy = y(tick)
        s.line(48, yy, 368, yy, width=0.6)
        s.text(39, yy + 4, f"{tick:g}", size=12, color="muted", anchor="end")
    s.line(48, 48, 48, 254, width=0.7)
    for tick in (20000, 30000, 40000, 50000):
        xx = x(tick)
        s.line(xx, 254, xx, 259, color="muted", width=0.7)
        s.text(xx, 277, f"{tick // 1000}k", size=12, color="muted", anchor="middle")
    s.text(208, 302, "updates", size=13, color="muted", anchor="middle")

    for kind, color, dashed in (
        ("stock_lead", "muted", True),
        ("forced_lead", "teal", False),
    ):
        for seed, trace in traces:
            path = "M" + " L".join(
                f"{x(row['update']):.3f} {y(row['restatement'][kind]):.3f}"
                for row in trace
            )
            s.parts.append(
                f'<path id="{kind}-{seed}" class="{kind}-curve" '
                f'd="{path}" fill="none" stroke="{s.color(color)}" '
                'stroke-width="1.1" stroke-opacity="0.65" '
                'stroke-linecap="round" stroke-linejoin="round"'
                + (' stroke-dasharray="3 3"' if dashed else "")
                + "/>",
            )
            last = trace[-1]
            s.parts.append(
                f'<circle cx="{x(last["update"]):.3f}" '
                f'cy="{y(last["restatement"][kind]):.3f}" r="2.2" '
                f'fill="{s.color(color)}" fill-opacity="0.8"/>',
            )
    s.text(205, y(1.5), "MQAR took off here", size=10, color="muted")
    s.line(282, 29, 296, 29, color="muted", width=1.1, dashed=True)
    s.text(302, 32, "unforced", size=10, color="muted")
    s.save("graded-halves", theme)


def muon_scratch(theme: str) -> None:
    """Connect all nine from-scratch restatement accuracy traces."""
    plateau_limit = 0.45
    data = cast(
        GradedData,  # noqa: TC006 - Workspace convention keeps runtime types explicit.
        json.loads(
            Path(__file__).with_name("plateau_graded.json").read_text(encoding="utf-8"),
        ),
    )
    copy_level = median(
        row["restatement"]
        for run in data["runs"]
        if run["recipe"] == "AdamW 1e-4"
        for row in run["evaluations"]
        if row["update"] > data["warm_updates"] and row["restatement"] < plateau_limit
    )
    s = SVG(
        theme,
        460,
        "Muon learns restatement immediately after the warm stage in all three seeds.",
        "Restatement accuracy for all nine from-scratch runs, three per recipe. "
        "Muon 3e-4 is teal, AdamW 2.5e-4 muted blue, and AdamW 1e-4 warm. "
        "All recorded evaluations are connected through each run's end. The "
        "shaded first 20,000 updates train verbatim only. The dashed copy level "
        f"is {copy_level:.4f}, the median of AdamW 1e-4 post-warm evaluations "
        "with restatement accuracy below 0.45.",
    )

    def x(update: int) -> float:
        return 48 + 320 * update / 56000

    def y(accuracy: float) -> float:
        return 368 - 320 * accuracy

    s.box(48, 48, x(data["warm_updates"]) - 48, 320, fill="faint", radius=0)
    s.text(16, 29, "accuracy", size=12, color="muted")
    for tick in (0, 0.5, 1):
        yy = y(tick)
        s.line(48, yy, 368, yy, width=1)
        s.text(39, yy + 4, f"{tick:g}", size=12, color="muted", anchor="end")
    s.line(48, 48, 48, 368, width=1)
    for tick in (0, 20000, 40000):
        xx = x(tick)
        s.line(xx, 368, xx, 373, color="muted", width=1)
        s.text(
            xx,
            391,
            f"{tick // 1000}k" if tick else "0",
            size=12,
            color="muted",
            anchor="middle",
        )
    s.text(208, 416, "updates", size=13, color="muted", anchor="middle")
    s.line(48, y(copy_level), 368, y(copy_level), color="muted", width=1, dashed=True)

    for recipe, color, opacity in (
        ("AdamW 2.5e-4", "blue", 0.5),
        ("AdamW 1e-4", "amber", 0.65),
        ("Muon 3e-4", "teal", 0.65),
    ):
        for run in data["runs"]:
            if run["recipe"] != recipe:
                continue
            path = "M" + " L".join(
                f"{x(row['update']):.3f} {y(row['restatement']):.3f}"
                for row in run["evaluations"]
            )
            s.parts.append(
                f'<path id="{recipe.lower().replace(" ", "-")}-{run["seed"]}" '
                f'class="run-curve" d="{path}" fill="none" '
                f'stroke="{s.color(color)}" stroke-width="1.4" '
                f'stroke-opacity="{opacity}" stroke-linecap="round" '
                'stroke-linejoin="round"/>',
            )
    s.text(58, 71, "verbatim only", size=10, color="muted")
    s.text(58, y(copy_level) - 8, "copy level", size=10, color="muted")
    s.text(218, 32, "Muon 3e-4", size=12, color="teal")
    s.line(250, 36, x(36000), y(0.999), color="teal", width=0.7)
    s.text(298, 133, "AdamW 2.5e-4", size=11, color="blue")
    s.text(270, 271, "AdamW 1e-4", size=12, color="amber")
    s.save("muon-scratch", theme)


def playbook(theme: str) -> None:
    """Draw a straight diagnosis flow ending in a test against control branches."""
    s = SVG(
        theme,
        310,
        "Diagnose the slow half, feed it, and test against control branches.",
        "Accuracy flat at 1/N leads to save checkpoints, stop waiting, then "
        "two reads: which half is flat? Feed the slow half, then test against "
        "control branches.",
    )
    s.parts.append(
        '<defs><marker id="arrow" viewBox="0 0 6 6" refX="6" refY="3" '
        'markerWidth="6" markerHeight="6" orient="auto">'
        f'<path d="M0 0L6 3L0 6" fill="none" stroke="{s.color("muted")}" '
        'stroke-width="1"/></marker></defs>',
    )
    for path in (
        "M210 54V76",
        "M210 114V136",
        "M210 174V196",
        "M210 234V256",
    ):
        s.parts.append(
            f'<path d="{path}" fill="none" stroke="{s.color("muted")}" '
            'stroke-width="1" stroke-linejoin="round" marker-end="url(#arrow)"/>',
        )
    for left, top, width, height, fill, stroke, lines in (
        (60, 16, 300, 38, "panel", "line", ("accuracy flat at 1/N",)),
        (60, 76, 300, 38, "panel", "line", ("save checkpoints, stop waiting",)),
        (60, 136, 300, 38, "panel", "line", ("two reads: which half is flat?",)),
        (60, 196, 300, 38, "teal_bg", "teal", ("feed the slow half",)),
        (60, 256, 300, 38, "blue_bg", "blue", ("test against control branches",)),
    ):
        s.box(left, top, width, height, fill=fill, stroke=stroke, radius=8)
        for index, label in enumerate(lines):
            baseline = top + height / 2 + 5 + (index - (len(lines) - 1) / 2) * 19
            s.text(left + width / 2, baseline, label, size=15, anchor="middle")
    s.save("playbook", theme)


def forced_read(theme: str) -> None:
    """Contrast ordinary attention with a forced read of the correct value."""
    s = SVG(
        theme,
        244,
        "Forcing attention onto the correct value reveals the copy path's lead.",
        "Schematic, not measured data. Both rows contain A 3 B 7 C 1 A ?. "
        "Ordinary attention is spread across the seven earlier tokens and the "
        "output scores for 3, 7 and 1 are equal. Forced attention permits only "
        "the first 3; all other earlier positions are blocked. The score for 3 "
        "then exceeds the mean score for 7 and 1, marked by a dashed line. "
        "The bracket marks that difference, the answer's lead.",
    )
    tokens = ("A", "3", "B", "7", "C", "1", "A", "?")
    for label, top, scores in (
        ("ordinary", 80, (1.0, 1.0, 1.0)),
        ("forced", 188, (3.0, 0.6, 0.6)),
    ):
        forced = label == "forced"
        bottom = top + 24
        attention_bottom = top - 6
        s.text(16, top + 16, label, size=11, color="muted")
        for index, token in enumerate(tokens):
            left = 80 + 24 * index
            answer = index == len(tokens) - 1
            selected = forced and index == 1
            blocked = forced and not answer and not selected
            color = "teal" if answer or selected else "muted"
            fill = "teal_bg" if answer or selected else "faint" if blocked else "panel"
            s.box(
                left,
                top,
                21,
                24,
                fill=fill,
                stroke="teal" if answer or selected else "line",
                radius=3,
            )
            if blocked:
                # Grey blocked tokens and cells; only the permitted value has a bar.
                s.parts.append('<g opacity="0.4">')
            s.parts.append(
                f'<text x="{left + 10.5}" y="{top + 16}" '
                f'fill="{s.color(color)}" font-family="Menlo, Consolas, monospace" '
                'font-size="12" text-anchor="middle">'
                f"{escape(token)}</text>",
            )
            if blocked:
                s.parts.append("</g>")
            if not answer and not blocked:
                height = 48 if forced else 48 / 7
                s.box(
                    left + 8,
                    attention_bottom - height,
                    5,
                    height,
                    fill="teal" if forced else "muted",
                    radius=1,
                )

        s.line(280, top + 12, 293, top + 12, color="muted", width=1)
        s.line(290, top + 9, 293, top + 12, color="muted", width=1)
        s.line(290, top + 15, 293, top + 12, color="muted", width=1)
        for index, (value, score) in enumerate(
            zip(("3", "7", "1"), scores, strict=True),
        ):
            center = 310 + 30 * index
            height = score * 64 / 3
            s.box(
                center - 6,
                bottom - height,
                12,
                height,
                fill="teal" if forced and index == 0 else "muted",
                radius=0,
            )
            s.text(center, bottom + 16, value, size=11, color="muted", anchor="middle")
        mean_y = bottom - (scores[1] + scores[2]) / 2 * 64 / 3
        s.line(300, mean_y, 380, mean_y, color="ink", width=0.8, dashed=True)
        if forced:
            lead_top = bottom - scores[0] * 64 / 3
            s.line(387, lead_top, 387, mean_y, color="teal", width=1)
            s.line(383, lead_top, 391, lead_top, color="teal", width=1)
            s.line(383, mean_y, 391, mean_y, color="teal", width=1)
            middle = (lead_top + mean_y) / 2
            s.parts.append(f'<g transform="rotate(-90 400 {middle})">')
            s.text(400, middle + 4, "lead", size=11, color="teal", anchor="middle")
            s.parts.append("</g>")
    s.save("forced-read", theme)


def two_factors(theme: str) -> None:
    """Compare the logistic product's response to starting strength and rate."""
    s = SVG(
        theme,
        250,
        "A hundredfold larger start and a doubled rate halve the waiting time.",
        "Schematic logistic solutions for product u = ab with target s = 1. "
        "Baseline: tau = 1, u0 = 0.0001. A 100 times larger start: tau = 1, "
        "u0 = 0.01. A 2 times faster rate: tau = 0.5, u0 = 0.0001. "
        "They reach u = 0.5 at about 9.21, 4.60 and 4.61 respectively; the "
        "faster-rate curve rises more steeply than the larger-start curve. "
        "These are analytic illustrations, not training measurements.",
    )

    def x(time: float) -> float:
        return 48 + 340 * time / 14

    def y(product: float) -> float:
        return 204 - 168 * product

    s.text(16, 22, "product u = ab", size=12, color="muted")
    s.line(48, 36, 48, 204, width=1)
    s.line(48, 204, 388, 204, width=1)
    for product in (0, 1):
        s.text(39, y(product) + 4, str(product), size=11, color="muted", anchor="end")
    for time in (0, 14):
        s.line(x(time), 204, x(time), 208, color="muted", width=1)
        s.text(x(time), 222, str(time), size=11, color="muted", anchor="middle")
    s.text(218, 237, "time", size=12, color="muted", anchor="middle")
    for name, tau, start, color in (
        ("baseline", 1.0, 1e-4, "muted"),
        ("larger-start", 1.0, 1e-2, "teal"),
        ("faster-rate", 0.5, 1e-4, "blue"),
    ):
        points: list[str] = []
        for index in range(561):
            time = 14 * index / 560
            growth = exp(time / tau)
            product = growth / (growth - 1 + 1 / start)
            points.append(f"{x(time):.3f} {y(product):.3f}")
        path = "M" + " L".join(points)
        s.parts.append(
            f'<path id="{name}-curve" d="{path}" fill="none" '
            f'stroke="{s.color(color)}" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/>',
        )
    s.text(64, 98, "100× larger start", size=12, color="teal")  # noqa: RUF001 - Multiplication.
    s.line(170, 94, 180, 87, color="teal", width=0.8)
    s.text(211, 25, "2× faster rate", size=12, color="blue")  # noqa: RUF001 - Multiplication.
    s.line(212, 30, 196, 45, color="blue", width=0.8)
    s.text(291, 145, "baseline", size=12, color="muted")
    s.line(287, 132, 281, 105, color="muted", width=0.8)
    s.save("two-factors", theme)


def main() -> None:
    """Write both themes of each plateau figure."""
    for theme in PALETTES:
        lottery(theme)
        resumes(theme)
        screens(theme)
        inside(theme)
        graded_halves(theme)
        muon_scratch(theme)
        playbook(theme)
        forced_read(theme)
        two_factors(theme)


if __name__ == "__main__":
    main()
