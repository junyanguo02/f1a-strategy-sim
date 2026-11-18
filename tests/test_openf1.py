import re
from datetime import timedelta

import pandas as pd

from f1a import data, openf1


def test_public_results_agree_with_openf1(cached_race_only):
    laps = data.load_race(2024, 1)
    counts = laps.groupby("driver_number").size()
    for _, group in laps.groupby("driver_number"):
        assert group["lap"].tolist() == list(range(1, len(group) + 1))
    finishers = [r for r in openf1.results(2024, 1)
                 if r["status"] in {"Finished", "Lapped"} or re.fullmatch(r"\+\d+ Laps?", r["status"])]
    assert len(finishers) == 20
    for result in finishers:
        assert counts[int(result["number"])] == int(result["laps"])


def test_control_intervals_and_missing_data():
    start = pd.Timestamp("2024-03-02T15:00Z")
    session = {"date_start": start}
    def at(seconds):
        return start + timedelta(seconds=seconds)
    inputs = dict(
        laps=[dict(driver_number=1, lap_number=i, date_start=at((i-1)*100),
                   lap_duration=100, is_pit_out_lap=i == 2) for i in (1, 2, 3)],
        drivers=[dict(driver_number=1, name_acronym="AAA", team_name="Team")],
        stints=[dict(driver_number=1, lap_start=1, lap_end=3, stint_number=1,
                     compound="MEDIUM", tyre_age_at_start=2)],
        pit=[dict(driver_number=1, lap_number=1)],
        weather=[dict(date=at(10), air_temperature=25, track_temperature=40)],
        race_control=[
            dict(date=at(25), message="SAFETY CAR DEPLOYED"),
            dict(date=at(100), message="TRACK CLEAR", flag="CLEAR", scope="Track"),
            dict(date=at(110), message="VIRTUAL SAFETY CAR DEPLOYED"),
            dict(date=at(195), message="VIRTUAL SAFETY CAR ENDING"),
            dict(date=at(200), message="GREEN FLAG", flag="GREEN", scope="Track"),
        ],
    )
    laps = openf1.assemble(inputs, session, 2024, 1)
    assert laps.track_status.tolist() == ["14", "167", "1"]
    assert laps.safety_car.tolist() == [True, False, False]
    assert laps.virtual_safety_car.tolist() == [False, True, False]
    assert laps.pit_in.tolist() == [True, False, False]
    assert laps.pit_out.tolist() == [False, True, False]
    assert laps.tyre_life.tolist() == [3, 4, 5]
    assert laps.air_temp.tolist() == [25, 25, 25]


def test_retry_after_then_raw_cache(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(openf1, "_last_request", {})
    sleep = Mock()
    monkeypatch.setattr(openf1.time, "sleep", sleep)
    response = Mock(status_code=200, content=b'[{"session_key":1}]')
    response.json.return_value = [{"session_key": 1}]
    request = Mock(side_effect=[Mock(status_code=429, headers={"Retry-After": "5"}), response])
    monkeypatch.setattr(openf1.requests, "get", request)
    for _ in range(2):
        assert openf1.fetch(openf1.OPEN, "laps", session_key=1) == [{"session_key": 1}]
    assert request.call_count == 2
    sleep.assert_any_call(5)


def test_future_round_does_not_request_openf1(monkeypatch):
    from unittest.mock import Mock
    import pytest

    monkeypatch.setattr(openf1, "calendar", lambda _: [{"round": "1", "date": "2099-01-01"}])
    fetch = Mock()
    monkeypatch.setattr(openf1, "fetch", fetch)
    with pytest.raises(ValueError, match="Race not yet held"):
        openf1.race_session(2099, 1)
    fetch.assert_not_called()


def test_only_expected_auxiliary_no_results_is_optional(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(openf1, "_last_request", {})
    monkeypatch.setattr(openf1.time, "sleep", Mock())
    response = Mock(status_code=404)
    response.json.return_value = {"detail": "No results found."}
    monkeypatch.setattr(openf1.requests, "get", Mock(return_value=response))
    assert openf1.fetch(openf1.OPEN, "pit", allow_no_results=True, session_key=1) == []
    with pytest.raises(RuntimeError, match="HTTP 404"):
        openf1.fetch(openf1.OPEN, "laps", session_key=1)
    response.json.return_value = {"detail": "Not Found"}
    with pytest.raises(RuntimeError, match="HTTP 404"):
        openf1.fetch(openf1.OPEN, "pit", allow_no_results=True, session_key=1)
