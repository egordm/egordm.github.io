"""Blog data: what builds inside MQAR's plateau (nightjar N594 and N595, N581's eight baselines).

Run: python3 <this file> <nightjar-data root>
Writes plateau_inside.json beside this file. Per seed: the plateau's start P (first evaluation at
0.9 of the copy level or more) and N581's takeoff T; every evaluation's accuracy and attention to
the correct value (N594's V: log of the mean attention weight on the value positions over the mean
on other visible positions, maximum over heads and layers); and every checkpoint's forced lead F
(N595: the answer's logit lead over the row's other values with attention forced onto the correct
value, maximum over layers) and stock lead L (the same lead without forcing).
"""

import json
import sys
from pathlib import Path


def main() -> None:
    lake = Path(sys.argv[1])
    source = lake / "N595-plateau-copy-path-read" / "n595-read-cpu" / "results.json"
    data = json.loads(source.read_text())
    assert data["verdict"] == "BUILDS", data["verdict"]
    seeds = []
    for entry in data["seeds"]:
        assert all(c["status"] == "completed" for c in entry["checkpoints"])
        seeds.append({
            "seed": entry["seed"],
            "P": entry["stats"]["P"],
            "T": entry["stats"]["T"],
            "copy_level": round(entry["stats"]["copy_level"], 4),
            "evaluations": [
                {"update": e["step"], "accuracy": round(e["accuracy"], 4),
                 "attention_to_value": round(max(e["value"]), 4)}
                for e in entry["evaluations"]
            ],
            "checkpoints": [
                {"update": c["step"], "forced_lead": round(c["F"], 4),
                 "stock_lead": round(c["L"], 4)}
                for c in entry["checkpoints"]
            ],
        })
    out = {"task": "MQAR, 32 tokens, 256-word vocabulary, 4 x 128, AdamW 1e-4",
           "units": "attention_to_value: log ratio; forced_lead and stock_lead: logits",
           "seeds": sorted(seeds, key=lambda s: s["seed"])}
    path = Path(__file__).with_name("plateau_inside.json")
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
