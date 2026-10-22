"""Blog data: two MQAR training runs with the same seed, data and initialisation (nightjar N581).

Run from a nightjar checkout: uv run --with wandb python <this file> <nightjar-data root>
Writes plateau_curves.json beside this file. The training loss (mean next-token loss over every
position of a training batch, logged every 50 updates) is read from each run's local W&B record. The runs are N581's calibration cell s32-v256 seed 1
and its baseline seed 1: 4 layers, hidden 128, AdamW 1e-4, batch 64, evaluation every 250 updates
on 512 fixed rows. MadStream seeds every step from the run seed, so both runs saw identical rows;
GPU and compile nondeterminism is the only difference between them.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

RUNS = {
    "first": ("N581-plateau-screen-setup/cal-32-256/evaluations.parquet", 1),
    "second": ("N581-plateau-screen-setup/base-s1/evaluations.parquet", 1),
}
WANDB = {"first": "N581-plateau-screen-setup/cal-32-256", "second": "N581-plateau-screen-setup/base-s1"}
TAKEOFF = {"first": 14_000, "second": 21_750}  # N581's registered crossings (node Outcome, P3)


def train_loss(home: Path) -> tuple[list[int], list[float]]:
    """Return (update, loss) from the one seed-1 W&B record under a run home."""
    (record,) = home.glob("s32-v256-seed1/wandb/run-*/run-*.wandb")
    store = datastore.DataStore()
    store.open_for_scan(str(record))
    steps, losses = [], []
    while (raw := store.scan_data()) is not None:
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in rec.history.item}
        if "train/loss" in row:
            steps.append(int(row["trainer/global_step"]))
            losses.append(round(float(row["train/loss"]), 4))
    return steps, losses


def main() -> None:
    lake = Path(sys.argv[1])
    out = {"task": "MQAR, 32 tokens, 256-word vocabulary", "source": "nightjar N581", "runs": {}}
    for name, (path, seed) in RUNS.items():
        df = pd.read_parquet(lake / path)
        df = df[df.seed == seed].sort_values("step")
        out["copy_level"] = round(float(df.copy_level.iloc[0]), 4)
        loss_step, loss = train_loss(lake / WANDB[name])
        out["runs"][name] = {
            "takeoff": TAKEOFF[name],
            "step": df.step.astype(int).tolist(),
            "accuracy": [round(float(x), 4) for x in df.accuracy],
            "loss_step": loss_step,
            "train_loss": loss,
        }
    target = Path(__file__).with_name("plateau_curves.json")
    target.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
