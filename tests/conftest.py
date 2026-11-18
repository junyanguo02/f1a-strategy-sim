"""Local inputs for the historical integration tests."""
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cached_race_only(monkeypatch):
    def missing(url, **kwargs):
        pytest.skip("Missing local OpenF1 or Jolpica response: " + url)
    monkeypatch.setattr(requests, "get", missing)


@pytest.fixture
def race_tables():
    for name in ("data/processed/model_laps.parquet", "data/processed/laps_pit_filled.parquet",
                 "data/processed/fcy_race_summary.csv", "data/processed/fcy_episodes.parquet",
                 "analysis/fcy_inputs/pit_loss_relative.parquet", "analysis/sc_pit_timing/outputs/pit_match_detail.csv"):
        if not (ROOT / name).is_file():
            pytest.skip("Missing " + name)


@pytest.fixture
def app_inputs(race_tables):
    for name in ("models/tyre_v3/outputs31_2024/posterior_core.npz", "analysis/actual_strategies/strategies.parquet"):
        if not (ROOT / name).is_file():
            pytest.skip("Missing " + name)


@pytest.fixture
def prior_inputs(race_tables):
    for name in ("models/tyre_v3/outputs31_2023/posterior_core.npz", "data/raw/jolpica/2024_races_limit=100.json"):
        if not (ROOT / name).is_file():
            pytest.skip("Missing " + name)
