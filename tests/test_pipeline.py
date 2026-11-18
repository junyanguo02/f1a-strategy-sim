from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
import pytest
from f1a import data
from scripts import build_cache as pipeline


def test_real_race(cached_race_only):
    laps = data.load_race(2024, 1)
    assert 1000 < len(laps) <= 20 * 57
    assert set(data.REQUIRED) <= set(laps.columns)
    assert not laps[data.REQUIRED].isna().all().any()
    assert laps.groupby("driver")["stint"].apply(lambda s: s.dropna().is_monotonic_increasing).all()
    assert laps["first_lap"].sum() == 20
    assert laps["pit_in"].any() and laps["pit_out"].any()


def test_flags_and_retention(monkeypatch):
    laps = pd.DataFrame({
        "Driver": ["AAA"] * 4, "Team": ["Test"] * 4, "LapNumber": [1, 2, 3, 4],
        "Compound": ["SOFT"] * 4, "TyreLife": [1, 2, 3, 4], "Stint": [1, 1, 2, 2],
        "TrackStatus": ["1", "14", "16", "17"],
        "LapTime": pd.to_timedelta([90, None, 100, 101], unit="s"),
        "PitInTime": pd.to_timedelta([None, 180, None, None], unit="s"),
        "PitOutTime": pd.to_timedelta([None, None, 190, None], unit="s"),
    })
    monkeypatch.setattr(pd.DataFrame, "get_weather_data", lambda self: pd.DataFrame({"AirTemp": [25.] * 4, "TrackTemp": [35.] * 4}), raising=False)
    monkeypatch.setattr(data, "enable_cache", lambda: None)
    session = SimpleNamespace(laps=laps, load=Mock())
    monkeypatch.setattr(data.fastf1, "get_session", lambda *args: session)
    result = data.load_race(2024, 1, source="fastf1")
    assert len(result) == 4
    assert result.lap_time_s.isna().sum() == 1
    assert result.safety_car.tolist() == [False, True, False, False]
    assert result.virtual_safety_car.tolist() == [False, False, True, True]
    assert result.first_lap.tolist() == [True, False, False, False]
    assert result.pit_in.tolist() == [False, True, False, False]
    assert result.pit_out.tolist() == [False, False, True, False]
    session.load.assert_called_once_with(laps=True, telemetry=False, weather=True, messages=False)
    session.laps = laps.iloc[:0]
    with pytest.raises(ValueError, match="No lap data"):
        data.load_race(2024, 1, source="fastf1")


def test_resume_and_failure_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "enable_cache", lambda: None)
    schedule = pd.DataFrame({"RoundNumber": [1, 2, 3], "EventName": ["A", "B", "C"]})
    monkeypatch.setattr(pipeline.fastf1, "get_event_schedule", lambda *args, **kwargs: schedule)
    calls = []
    def loader(year, round, source):
        calls.append(round)
        if round == 2:
            raise RuntimeError("test failure")
        return pd.DataFrame({"year": [year], "round": [round], "lap": [1]})
    monkeypatch.setattr(pipeline, "load_race", loader)
    assert pipeline.build_cache([2024], source="fastf1") == 1
    assert calls == [1, 2, 3]
    assert pd.read_parquet("data/processed/laps_2024_fastf1.parquet")["round"].tolist() == [1, 3]
    calls.clear()
    assert pipeline.build_cache([2024], source="fastf1") == 1
    assert calls == [2]
    assert len(pd.read_csv("data/processed/failed.csv")) == 2
    monkeypatch.setattr(pipeline, "load_race", lambda year, round, source: pd.DataFrame({"year": [year], "round": [round], "lap": [1]}))
    assert pipeline.build_cache([2024], source="fastf1") == 0
    assert len(pd.read_parquet("data/processed/laps_2024_fastf1.parquet")) == 3
    assert len(pd.read_csv("data/processed/failed.csv")) == 2


def test_schedule_failure_and_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "enable_cache", lambda: None)
    monkeypatch.setattr(pipeline.fastf1, "get_event_schedule", Mock(side_effect=RuntimeError("schedule unavailable")))
    monkeypatch.setattr("sys.argv", ["build_cache.py", "--source", "fastf1", "--years", "2024", "2025"])
    assert pipeline.main() == 1
    assert pd.read_csv("data/processed/failed.csv")["year"].tolist() == [2024, 2025]


def test_cache_stays_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = Mock()
    monkeypatch.setattr(data.fastf1.Cache, "enable_cache", cache)
    try:
        from fastf1.internals.f1account import token_manager
    except ImportError:
        token_manager = None
    if token_manager is not None:
        monkeypatch.setattr(token_manager, "_token_path", token_manager._token_path)
    data.enable_cache()
    assert (tmp_path / "data/raw/fastf1-cache").is_dir()
    cache.assert_called_once_with("data/raw/fastf1-cache")
    if token_manager is not None:
        assert token_manager._token_path == pipeline.Path("data/raw/fastf1-cache/f1auth_token.json")
