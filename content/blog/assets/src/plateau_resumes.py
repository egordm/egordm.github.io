"""Blog data: a paired resume screen on MQAR (nightjar N582).

Run from a nightjar checkout: uv run python <this file> <nightjar-data root>
Writes plateau_resumes.json beside this file. Every resume starts from one checkpoint, N581's
baseline seed 2 at update 5,500 (inside the loss plateau, a third of the way to that run's takeoff
at 16,500), and continues on a fresh data order (seeds 1 to 3). Arm `off` trains on stock rows;
arm `rep2` on rows where each key appears about twice as often. Evaluation every 250 updates on
N581's 512 stock rows. Only the technique phase is kept (the switch-off phase follows it).
"""

import json
import sys
from pathlib import Path

import pandas as pd

HOME = "N582-repetition-resume-screen"
ARMS = ("off", "rep2")
SEEDS = (1, 2, 3)


def main() -> None:
    lake = Path(sys.argv[1])
    out = {"task": "MQAR, 32 tokens, 256-word vocabulary", "source": "nightjar N582",
           "resume_from": 5_500, "copy_level": 0.1003, "runs": []}
    for arm in ARMS:
        for seed in SEEDS:
            home = lake / HOME / f"{arm}-d{seed}"
            df = pd.read_parquet(home / f"{arm}-seed{seed}" / "evaluations.parquet")
            df = df[df.phase == "technique"].sort_values("resume_update")
            (run,) = json.loads((home / "measurements.json").read_text())["runs"]
            out["runs"].append({
                "arm": arm,
                "data_seed": seed,
                "takeoff": int(run["takeoff_update"]),
                "update": df.resume_update.astype(int).tolist(),
                "accuracy": [round(float(x), 4) for x in df.accuracy],
            })
    target = Path(__file__).with_name("plateau_resumes.json")
    target.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
