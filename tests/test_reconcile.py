import pandas as pd

from scripts.reconcile import compare


def test_reconcile_keeps_dnf_and_unmatched_drivers():
    laps = pd.DataFrame({"driver_number": [1, 1, 4], "driver": ["VER", "VER", "NOR"]})
    results = [
        {"number": "1", "laps": "1", "status": "Retired", "Driver": {"code": "VER"}},
        {"number": "16", "laps": "0", "status": "Did not start", "Driver": {"code": "LEC"}},
        {"number": "44", "laps": "7", "Driver": {"code": "HAM"}},
    ]
    assert compare(2024, 1, laps, results) == [
        (2024, 1, "VER", 2, 1, 1),
        (2024, 1, "NOR", 1, None, None),
        (2024, 1, "LEC", 0, 0, 0),
        (2024, 1, "HAM", 0, 7, -7),
    ]
    assert compare(2024, 1, laps.iloc[:0], results)[0] == (2024, 1, "VER", None, 1, None)
