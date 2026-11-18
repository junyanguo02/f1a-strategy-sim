from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1a.sim import RaceConfig, Strategy, TyreParams, common_draws, simulate
from f1a.sim.race import pit_tables

ROOT = Path(__file__).resolve().parents[1]


def test_unchanged_default_matches_original_inputs_and_simulation(race_tables):
    if not (ROOT / "models/tyre_v3/outputs31_2024/posterior_core.npz").is_file():
        pytest.skip("Missing the 2024 tyre posterior")
    race_id = "2024-10"
    processed = ROOT / "data/processed"
    model = pd.read_parquet(processed / "model_laps.parquet").query("race_id == @race_id")
    original = RaceConfig.from_samples(
        race_id, model.circuit_id.iloc[0], int(model.total_laps.iloc[0]), float(model.lap_time_s.median()),
        pd.read_parquet(ROOT / "analysis/fcy_inputs/pit_loss_relative.parquet"),
        pd.read_csv(ROOT / "analysis/sc_pit_timing/outputs/pit_match_detail.csv"),
        pd.read_parquet(processed / "fcy_episodes.parquet"),
        pd.read_csv(processed / "fcy_race_summary.csv"), pd.read_parquet(processed / "laps_pit_filled.parquet"))
    tyre = TyreParams.from_posterior(ROOT / "models/tyre_v3/outputs31_2024/posterior_core.npz",
                                    race_id, [2024], processed / "model_laps.parquet", n_draws=50)
    plan = Strategy("MH", "MEDIUM", [(22, "HARD")])

    def totals(cfg):
        draws = common_draws(cfg, tyre, 100, 20261118)
        return simulate(cfg, tyre, plan, draws=draws)["total"].tobytes()

    for current in (RaceConfig.from_project(ROOT, race_id),
                    RaceConfig.from_project(ROOT, race_id, pit_loss_filter=None)):
        assert current.notes == original.notes
        assert current.pit_normal.tobytes() == original.pit_normal.tobytes()
        assert current.pit_fcy.keys() == original.pit_fcy.keys()
        for key in current.pit_fcy:
            assert current.pit_fcy[key].tobytes() == original.pit_fcy[key].tobytes()
        assert totals(current) == totals(original)


def test_ranges_are_inclusive_and_clean_uses_stop_identity(tmp_path):
    raw = pd.DataFrame({"year": 2024, "round": 1, "driver": "BOT", "lap": range(1, 6),
                        "relative_loss_s": [9., 10., 60., 61., np.nan]})
    csv = tmp_path / "analysis/sc_pit_timing/outputs/pit_match_detail.csv"
    file = tmp_path / "analysis/fcy_inputs/pit_loss_relative.parquet"
    for path in (csv, file):
        path.parent.mkdir(parents=True)
    raw.iloc[[3, 1, 2, 0]].to_csv(csv, index=False)
    raw.to_parquet(file)
    pl, det = pit_tables(tmp_path, "range")
    assert pl.lap.tolist() == [2, 3] and det.lap.tolist() == [2, 3]
    assert pit_tables(tmp_path, "range", (9., 10.))[1].lap.tolist() == [2, 1]
    with pytest.raises(FileNotFoundError, match="T042"):
        pit_tables(tmp_path, "clean")
    path = tmp_path / "analysis/pitloss_audit/outputs/pit_loss_relative_clean.parquet"
    path.parent.mkdir(parents=True)
    raw.assign(clean_keep=[False, True, False, True, False]).to_parquet(path)
    pl, det = pit_tables(tmp_path, "clean")
    assert pl.lap.tolist() == [2, 4] and det.lap.tolist() == [4, 2]

    # A future unmatched detail row must not enter the pre-race join.
    future = raw.iloc[[0]].assign(year=2025, lap=99)
    pd.concat([raw.iloc[[3, 1]], future]).to_csv(csv, index=False)
    pl, det = pit_tables(tmp_path, "clean", history=lambda df: df[df.year < 2025])
    assert pl.lap.tolist() == [2, 4] and det.lap.tolist() == [4, 2]
