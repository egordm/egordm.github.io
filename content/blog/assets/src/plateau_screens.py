"""Blog data: every MQAR fix screened from one plateau checkpoint (nightjar N582 to N590).

Run: python3 <this file> <nightjar-data root>
Writes plateau_screens.json beside this file. Every resume starts from N581's baseline seed 2 at
update 5,500 and runs on data orders 1 to 3, censored at 27,500 updates. Each screen ran its own
stock reference beside its arms (N583 reused N582's; the Muon screens reset the optimizer in the
reference too, because their arms start a fresh optimizer). A fix passes when all three of its
resumes take off before every resume of its reference and end at 0.95 or more on stock rows after
the switch-off. N585's out-of-memory homes were relaunched as `<id>-r`; the rerun is read.
"""

import json
import sys
from pathlib import Path

CAP = 27_500
SEEDS = (1, 2, 3)

# (node, lake folder, home pattern, reference arm, [(arm, label, verdict)])
SCREENS = [
    ("N582", "N582-repetition-resume-screen", "{arm}-d{seed}", "off",
     [("rep2", "repeated keys, 2x", "pass")]),
    ("N583", "N583-repetition-dose-family", "n583-{arm}-d{seed}", None,
     [("k14", "repeated keys, 1.5x", "pass"), ("k5", "repeated keys, 3x", "pass")]),
    ("N587", "N587-muon-resume-screen", "n587-{arm}-d{seed}", "offreset",
     [("muon3e-4", "Muon 3e-4", "pass"), ("muon1e-4", "Muon 1e-4", "pass"),
      ("muon1e-3", "Muon 1e-3", "fail")]),
    ("N589", "N589-muon-rate-control", "n589-{arm}-d{seed}", "reset1e-4",
     [("reset3e-4", "AdamW 3e-4 (rate control)", "fail")]),
    ("N588", "N588-long-curriculum-resume-screen", "n588-{arm}-d{seed}", "off",
     [("ramp750", "length ramp, a pair per 750", "pass"),
      ("ramp1500", "length ramp, a pair per 1,500", "fail")]),
    ("N585", "N585-technique-resume-screen", "n585-{arm}-d{seed}", "off",
     [("curr", "short three-stage curriculum", "fail"), ("mix10", "10% easy batches", "fail"),
      ("mix90", "90% easy batches", "fail"), ("sharp4", "sharper attention, 4x", "fail"),
      ("sharp12", "sharper attention, 12x", "fail"), ("sharp24", "sharper attention, 24x", "fail")]),
    ("N590", "N590-fixed-map-mixing-resume-screen", "n590-{arm}-d{seed}", "off",
     [("map10", "10% fixed-map batches", "fail"), ("map90", "90% fixed-map batches", "fail")]),
]


def read(folder: Path, pattern: str, arm: str) -> dict:
    takeoffs, finals = [], []
    for seed in SEEDS:
        home = folder / pattern.format(arm=arm, seed=seed)
        rerun = home.with_name(home.name + "-r")
        if rerun.exists():
            home = rerun
        (run,) = json.loads((home / "measurements.json").read_text())["runs"]
        assert run["state"] == "completed" and run["cap_updates"] == CAP, home
        takeoffs.append(None if run["censored"] else int(run["takeoff_update"]))
        finals.append(round(float(run["final_stock_accuracy"]), 3))
    return {"takeoff": takeoffs, "final_accuracy": finals}


def main() -> None:
    lake = Path(sys.argv[1])
    out = {"task": "MQAR, 32 tokens, 256-word vocabulary, 4 x 128",
           "resume_from": 5_500, "cap": CAP,
           "note": "N582's final_accuracy is its registered read, 3,000 updates after the crossing "
                   "(0.855 still rising); its Amendment 1 read after 11,000 more stock updates is "
                   "0.992, 1.000, 1.000. Later screens read after 11,000 stock updates.",
           "rows": []}
    n582 = read(lake / SCREENS[0][1], SCREENS[0][2], "off")
    for node, folder, pattern, ref, arms in SCREENS:
        reference = n582 if ref is None else read(lake / folder, pattern, ref)
        for arm, label, verdict in arms:
            row = read(lake / folder, pattern, arm)
            out["rows"].append({"node": node, "label": label, "verdict": verdict, **row,
                                "reference": reference["takeoff"]})
    out["rows"].sort(key=lambda row: row["verdict"] != "pass")  # passes first, screen order kept
    target = Path(__file__).with_name("plateau_screens.json")
    target.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
