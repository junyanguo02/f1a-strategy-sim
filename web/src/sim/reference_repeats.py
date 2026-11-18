"""Regenerate Python repeat means for the independent-draws Vitest check.

Run with the project .venv. Each observation is one mean of 5000 paired
differences (exported second plan minus first) using shared draws within a run.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from f1a.sim import RaceConfig, Strategy, TyreParams, common_draws, simulate

def compute(race_id):
    folder = ROOT / "web/public/data/races" / race_id
    doc = json.loads((folder / "sim_inputs_prerace.json").read_text())
    candidates = json.loads((folder / "strategies.json").read_text())["candidates"][:2]
    fcy = doc["fcy"]
    loss = doc["pit_loss"]
    race = RaceConfig(race_id=race_id, total_laps=doc["total_laps"],
                      base_lap_s=doc["base_lap_time_s"], pit_normal=np.array(loss["green"]),
                      pit_fcy={(event, int(phase)): np.array(pool)
                               for event, phases in loss["fcy"].items() for phase, pool in phases.items()},
                      **{k: {event: values[k] for event, values in fcy.items()}
                         for k in ("hazard", "lap1_prob", "durations", "lap_mult")})
    params = doc["tyre_posterior"]
    tyre = TyreParams(rem_mean=params["rem_mean"], **{
        k: np.array(params[k]) for k in ("k0","beta","kappa","gamma","sd_stint","nu","sigma")})
    plans = [Strategy(c["label"], c["compounds"][0], list(zip(c["pit_laps"], c["compounds"][1:])))
             for c in candidates]
    means = []
    for seed in range(doc["seed"], doc["seed"] + 20):
        draws = common_draws(race, tyre, 5000, seed, max_stints=4)
        first, second = [simulate(race, tyre, plan, draws=draws)["total"] for plan in plans]
        means.append(float((second-first).mean()))
    return {"id":race_id, "seed":doc["seed"], "labels":[p.name for p in plans], "means":means}

if __name__ == "__main__":
    output = {"nSims":5000, "repeats":20, "t975":[float(t.ppf(.975, df)) for df in range(19,39)],
              "races":[compute(r) for r in ("2024-1","2024-3","2024-10")]}
    (HERE / "python_repeats.json").write_text(json.dumps(output, indent=2)+"\n")
