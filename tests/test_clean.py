import pandas as pd
import pytest

from f1a.clean import clean_laps


def lap(**changes):
    return dict({"year": 2024, "round": 1, "driver": "VER", "lap": 3,
                 "lap_time_s": 100., "compound": "MEDIUM", "pit_in": False,
                 "pit_out": False, "safety_car": False, "virtual_safety_car": False,
                 "track_status": "1"}, **changes)


@pytest.mark.parametrize(("change", "reason"), [
    ({"lap": 1}, "first_two_laps"), ({"lap": 2}, "first_two_laps"),
    ({"pit_in": True}, "pit_in"), ({"pit_out": True}, "pit_out"),
    ({"safety_car": True}, "safety_car"),
    ({"virtual_safety_car": True}, "virtual_safety_car"),
    ({"safety_car": pd.NA}, "unknown_safety_status"),
    ({"virtual_safety_car": pd.NA}, "unknown_safety_status"),
    ({"track_status": "154"}, "red_flag"),
    ({"compound": "INTERMEDIATE"}, "wet_compound"),
    ({"compound": "WET"}, "wet_compound"),
    ({"compound": "UNKNOWN"}, "unknown_compound"),
    ({"compound": ""}, "unknown_compound"),
    ({"compound": pd.NA}, "unknown_compound"),
    ({"lap_time_s": pd.NA}, "missing_lap_time"),
])
def test_preliminary_filter(change, reason):
    frame = pd.DataFrame([lap(**change)])
    result = clean_laps(frame)
    assert result["is_clean"].tolist() == [False]
    assert result["drop_reason"].tolist() == [reason]
    pd.testing.assert_frame_equal(result[frame.columns], frame)


def test_threshold_grouping_and_preservation():
    rows = [lap(), lap(), lap(lap_time_s=107.), lap(lap_time_s=108.),
            lap(lap_time_s=1., pit_in=True, pit_out=True),
            lap(round=2, lap_time_s=200.), lap(driver="NOR", lap_time_s=200.),
            lap(year=2025, lap_time_s=200.)]
    # Keep three eligible laps at 100 seconds, giving a 107-second threshold.
    rows.insert(0, lap())
    frame = pd.DataFrame(rows, index=[5] * len(rows))
    result = clean_laps(frame)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result["is_clean"].tolist() == [True, True, True, True, False, False, True, True, True]
    assert result["drop_reason"].iloc[4:6].tolist() == ["slow_lap", "pit_in|pit_out"]
