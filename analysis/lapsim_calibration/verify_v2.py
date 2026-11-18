"""Recompute T052 outputs from the saved parameters and compare independent runs."""
import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import CarParams, HERE, POST, ROOT, load_tracks, solve_lap


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-9)


def verify(out):
    tracks = {t["race_id"]: t for t in load_tracks()}
    metrics = pd.read_csv(out / "metrics.csv")
    parameters = pd.read_csv(out / "parameters.csv")
    curves = pd.read_csv(out / "profiles.csv").groupby(["power", "mode", "race"])
    old = pd.read_csv(HERE / "outputs/profiles.csv").query("mode == 'shared'")
    offsets = {}
    sweep = pd.read_csv(out / "alignment.csv")
    for race, t in tracks.items():
        sim = old.loc[old.race == race, "sim"].to_numpy()
        losses = [np.mean((sim - np.roll(t["ref"], -i)) ** 2) ** .5 for i in range(-12, 13)]
        close(sweep.loc[sweep.race == race].sort_values("delta_m").v1_rmse, losses)
        offsets[race] = np.argmin(losses) - 12
    assert len(parameters) == 6 and len(metrics) == 30
    laps = pd.read_parquet(ROOT / "data/processed/model_laps.parquet")
    race_order = laps.loc[laps.year == 2024, "race_id"].unique().tolist()
    with np.load(POST) as samples:
        gamma = samples["gamma"]
    assert gamma.shape == (4000, len(race_order))
    for q in parameters.itertuples():
        group = metrics[(metrics.power == q.power) & (metrics["mode"] == q.mode)]
        assert set(group.race) == set(tracks)
        objective = 0.
        for m in group.itertuples():
            t = tracks[m.race]
            curve = curves.get_group((q.power, q.mode, m.race))
            shift = offsets[m.race]
            close(m.delta_m, shift * 5)
            close(curve.ref, np.roll(t["ref"], -shift))
            car = CarParams(mass_kg=t["mass"], power_kW=q.power, cda_m2=q.cda,
                            cla_m2=q.cla, brake_max_mps2=q.brake, mu=m.mu)
            def lap(delta):
                return solve_lap(t["s"], t["k"], replace(car, mass_kg=t["mass"] + delta))
            fitted = lap(0)
            close(fitted.v * 3.6, curve.sim)
            close(fitted.total - t["reference_lap_time_s"], m.dt)
            error = curve.sim - curve.ref
            close(np.mean(error ** 2) ** .5, m.rmse)
            objective += np.dot(np.roll(t["weight"], -shift), (error / 10) ** 2) + (m.dt / 2) ** 2
            slope = (lap(5).total - lap(-5).total) / 10
            close(slope, m.dt_dm)
            close(slope * np.array([105, 107.5, 110]) / t["race_laps"], [m.fuel_low, m.fuel, m.fuel_high])
            index = race_order.index(m.race)
            draws = gamma[:, index]
            close([index, draws.mean(), *np.quantile(draws, [.05, .95])], [m.gamma_index, m.gamma, m.q05, m.q95])
        close(q.objective, objective)
    pairs = pd.read_csv(out / "matched_pairs.csv")
    original = laps.set_index(["race_id", "driver", "lap"])
    for column in ["early", "late"]:
        indices = pd.MultiIndex.from_frame(pairs[["race", "driver", column]])
        source = original.loc[indices].reset_index(drop=True)
        assert (source.compound == pairs.compound).all()
        close(source.tyre_life, pairs.age)
        pairs[column + "_s"] = source.lap_time_s
        pairs[column + "_stint"] = source.stint
    close(pairs.early_s - pairs.late_s, pairs.gain)
    close(pairs.late - pairs.early, pairs.span)
    assert pairs.span.min() >= 10 and (pairs.early_stint != pairs.late_stint).all()
    main = metrics.query("power == 750 and mode == 'track_mu'").set_index("race")
    for row in pd.read_csv(out / "evolution.csv").itertuples():
        g = pairs[(pairs["sample"] == row.sample) & (pairs.race == row.race)]
        fuel = main.loc[row.race, "fuel"]
        residual = np.dot(g.span, g.gain) / np.dot(g.span, g.span) - fuel
        close([len(g), g.driver.nunique(), residual, main.loc[row.race,"gamma"] - fuel],
              [row.n, row.drivers, row.residual, row.posterior_gap])
    expected = {f"figures/{name}.png" for name in [*tracks, "fuel"]}
    assert expected == {str(f.relative_to(out)) for f in (out/"figures").glob("*.png")}
    print("Verified 5 offsets, 6 objectives, 30 speed curves, mass derivatives, fuel effects and posterior comparisons,",
          len(pairs), "matched-pair records and 15 trend estimates.")


def manifest(path):
    return {p.relative_to(path): sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "outputs_v2")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    verify(args.out)
    if args.compare:
        first = manifest(args.out)
        assert first == manifest(args.compare), "Run manifests differ"
        print(len(first), "output files identical by SHA-256.")
