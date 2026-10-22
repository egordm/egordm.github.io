"""Blog data: the graded language's plateau (nightjar N586 stage A, N593, N598).

Run: <python with pandas and pyarrow> <this file> <nightjar-data root>
Writes plateau_graded.json beside this file.

- `runs`: pooled accuracy per form (verbatim, restatement) at every evaluation of the from-scratch
  runs: N586 stage A (AdamW 1e-4, seeds 1 to 3, every 4,000 updates to 60,000) and N593 (Muon 3e-4
  and AdamW 2.5e-4, seeds 1 to 3, every 2,000 updates to 44,000). All share the recipe: 4 x 128,
  16 properties, 144 carriers, batch 64, 20,000 verbatim-only warm updates, then a 50/50 mix.
  Pooled accuracy is the share of correct answers over every evaluation row of that form.
- `two_half`: N598's read on stage A's checkpoints: per seed, form and checkpoint, the stock lead L
  and the forced lead F (attention forced onto the target name's positions), in logits.
"""

import json
import sys
from pathlib import Path

import pandas as pd

RECIPES = {("adamw", 0.0001): "AdamW 1e-4", ("adamw", 0.00025): "AdamW 2.5e-4",
           ("muon", 0.0001): "Muon 3e-4"}


def pooled(home: Path) -> list[dict]:
    points = []
    for path in home.glob("step-*.parquet"):
        frame = pd.read_parquet(path, columns=["form", "correct", "step"])
        accuracy = frame.groupby("form")["correct"].mean()
        points.append({"update": int(frame["step"].iloc[0]),
                       "verbatim": round(float(accuracy["verbatim"]), 4),
                       "restatement": round(float(accuracy["restatement"]), 4)})
    return sorted(points, key=lambda p: p["update"])


def main() -> None:
    lake = Path(sys.argv[1])
    runs = []
    homes = [*(lake / "N586-graded-technique-resume" / "stage-a").glob("2026*"),
             *(lake / "N593-graded-muon-from-scratch").glob("2026*")]
    for home in sorted(homes):
        settings = json.loads((home / "config.json").read_text())["settings"]
        recipe = RECIPES[(settings.get("optimizer") or "adamw", settings["lr"])]
        runs.append({"recipe": recipe, "seed": settings["seed"], "home": home.name,
                     "evaluations": pooled(home)})
    read = json.loads((lake / "N598-graded-two-half-read" / "n598-read" / "results.json").read_text())
    two_half = [
        {"seed": entry["seed"], "checkpoints": [
            {"update": c["step"], **{form: {"stock_lead": round(v["L"], 3),
                                            "forced_lead": round(v["F"], 3),
                                            "accuracy": round(v["stored_accuracy"], 4)}
                                     for form, v in c["forms"].items()}}
            for c in entry["checkpoints"]]}
        for entry in read["seeds"]]
    out = {"task": "graded language, 4 x 128, 16 properties, 144 carriers, batch 64",
           "warm_updates": 20_000, "runs": sorted(runs, key=lambda r: (r["recipe"], r["seed"])),
           "two_half": sorted(two_half, key=lambda s: s["seed"])}
    Path(__file__).with_name("plateau_graded.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
