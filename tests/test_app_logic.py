import pandas as pd
import pytest

from app.logic import COLUMNS, default_strategies, edited_strategies, strategy_row
from f1a.sim import Strategy


def test_strategy_round_trip():
    s = Strategy("两停", "SOFT", [(10, "HARD"), (32, "MEDIUM")])
    assert edited_strategies(pd.DataFrame([strategy_row(s)]), 50) == ([s], [])


@pytest.mark.parametrize(("seq", "laps"), [
    ("H-H", "12"), ("M-X", "10"), ("S-M", "10.5"), ("M-H-S", "15"),
    ("M-H-S", "19,15"), ("M-H", "50"), ("S-M", "0"),
])
def test_invalid_edit(seq, laps):
    table = pd.DataFrame([["a", seq, laps]], columns=COLUMNS)
    result, errors = edited_strategies(table, 50)
    assert not result and errors


def test_empty_table_and_duplicate_names():
    empty = pd.DataFrame(columns=COLUMNS)
    assert edited_strategies(empty, 50)[1]
    table = pd.DataFrame([["a","M-H","10"], [" a ","H-S","22"]], columns=COLUMNS)
    assert "重复" in edited_strategies(table, 50)[1][0]


def test_defaults_separate_stop_counts_and_use_observed_laps():
    opt = pd.DataFrame([("H-S",(18,),4700), ("M-H",(24,),4699), ("S-M-H",(10,30),4650)],
                       columns=["sequence","stops","mean"])
    cars = pd.DataFrame([(True,11,"XXX","MH","20"), (True,3,"C","SH","20"),
                         (True,2,"B","MH","25"), (True,1,"A","MH","24")],
                        columns=["finisher","finish_position","driver","strategy","pit_laps"])
    table, notes = default_strategies(opt, cars, 50)
    assert not notes
    assert list(table.itertuples(index=False, name=None)) == [
        ("最优一停","M-H","24"), ("最优两停","S-M-H","10,30"), ("前十主流 · A","M-H","24")]
    cars["strategy"] = "W"
    table, notes = default_strategies(opt, cars, 50)
    assert len(table) == 2 and notes


@pytest.mark.parametrize(("sequence", "expected"), [
    ("H-M","M-H"), ("S-S-M","S-M-S"), ("M-S-H","M-H-S"), ("S-H-M","S-H-M"),
])
def test_default_display_does_not_rewrite_execution(sequence, expected):
    from app.logic import display_sequence
    assert display_sequence(sequence) == expected
    stops = ",".join(str(i * 10) for i in range(1, len(sequence.split("-"))))
    table = pd.DataFrame([["我的策略", sequence, stops]], columns=COLUMNS)
    results, errors = edited_strategies(table, 60)
    assert not errors and strategy_row(results[0])["配方顺序"] == sequence


def test_paired_differences_change_sign_when_rebased():
    from app.logic import paired_summary
    totals = pd.DataFrame({"A":[10, 100, 45], "B":[12, 97, 46]})
    b = paired_summary(totals, "A").set_index("策略").loc["B"]
    a = paired_summary(totals, "B").set_index("策略").loc["A"]
    assert b["P50"] == 1 and b["配对均差"] == 0
    assert b["P2.5"] == pytest.approx(-a["P97.5"])
    assert a["P50"] == -1
    assert paired_summary(totals, "A").set_index("策略").loc["A"].eq(0).all()


@pytest.mark.parametrize(("mean","lo","hi","expected"), [
    (-2,-4,-1,"显著更快"), (2,1,4,"显著更慢"), (-1,-2,1,"包含 0"),
    (1,0,2,"包含 0"), (0,0,0,"平均用时相同")])
def test_conclusion(mean, lo, hi, expected):
    from app.logic import conclusion
    assert expected in conclusion("A","B",mean,lo,hi)


def test_rank_gain_uses_actual_minus_candidate():
    import numpy as np
    from app.logic import position_gain
    actual = np.array([1,4,5,2])
    assert position_gain(actual,actual) == 0
    assert position_gain(actual,np.array([2,2,3,1])) == 1


def test_race_scenarios(race_tables):
    from pathlib import Path
    import numpy as np
    from app.logic import SCENARIOS, scenario_race
    root = Path(__file__).resolve().parents[1]
    baseline, circuit, off = [scenario_race(root, "2024-10", s) for s in SCENARIOS]
    assert baseline.lap1_prob != circuit.lap1_prob
    assert any(value > 0 for value in baseline.lap1_prob.values())
    assert all(value == 0 for value in off.lap1_prob.values())
    assert all(np.all(values == 0) for values in off.hazard.values())
    with pytest.raises(ValueError):
        scenario_race(root, "2024-10", "unknown")
