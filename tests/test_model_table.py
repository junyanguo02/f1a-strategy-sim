import pandas as pd
import pytest
from f1a.pit import fill_pit_in, infer_pit_in, validate_pit_in
from f1a.clean import clean_laps
from scripts.race_quality import quality_row, model_rows, MODEL_COLUMNS


def fixture():
    return pd.DataFrame([dict(year=2024, round=1, driver='VER', driver_number=1,
        team='Red Bull', lap=n, stint=1 if n <= 3 else 2, pit_in=pd.NA,
        pit_out=False, compound='SOFT', tyre_life=n, lap_time_s=100.,
        safety_car=False, virtual_safety_car=False, track_status='1') for n in range(1, 7)])


def race():
    return dict(season='2024', round='1', raceName='Example', Circuit={'circuitId': 'example'},
                Results=[{'number': '1', 'laps': '6', 'position': '1'}])


def test_fill_boundaries_and_clean_preserves_original():
    df = fixture().sample(frac=1, random_state=7)
    original = df.copy(deep=True)
    out = clean_laps(df)
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_series_equal(out.pit_in, df.pit_in)
    assert out.loc[out.pit_in_inferred, 'lap'].tolist() == [3]
    assert out.loc[out.lap == 3, 'drop_reason'].iloc[0] == 'pit_in'
    assert set(out.loc[out.is_clean, 'lap']) == {4, 5, 6}
    assert not infer_pit_in(df).loc[df.lap == 6].iloc[0]


def test_no_fill_for_partial_missing_or_known_negative_race():
    df = fixture()
    df.loc[0, 'pit_in'] = False
    out = fill_pit_in(df)
    assert not out.pit_in_inferred.any()
    assert out.pit_in_filled.isna().sum() == 5
    df['pit_in'] = False
    assert not fill_pit_in(df).pit_in_filled.any()


def test_group_isolation_unknown_stint_and_duplicate_index():
    a = fixture()
    b = fixture().assign(round=2, pit_in=False)
    c = fixture().assign(driver_number=2, driver='NOR')
    c.loc[5, 'stint'] = pd.NA
    out = fill_pit_in(pd.concat([a, b, c]))
    assert set(map(tuple, out.loc[out.pit_in_inferred, ['round', 'driver_number', 'lap']].values)) == {(1, 1, 3), (1, 2, 3), (1, 2, 5)}
    assert pd.isna(out.loc[(out.driver_number == 2) & (out.lap == 6), 'pit_in_filled'].iloc[0])


def test_validation_counts_false_positive_false_negative_unknown():
    df = fixture()
    df['pit_in'] = [False, True, False, False, pd.NA, True]
    stats = validate_pit_in(df)
    assert stats == dict(tp=0, fp=1, fn=2, evaluated=5, unknown=1, precision=0., recall=0.)
    df['pit_in'] = df.lap.eq(3)
    assert validate_pit_in(df)['precision'] == 1.
    assert validate_pit_in(df)['recall'] == 1.


def test_quality_missing_driver_zero_laps_and_strategy_sequence():
    df = fill_pit_in(fixture())
    r = race()
    r['Results'] += [{'number': '2', 'laps': '0', 'position': '2'}]
    q = quality_row(df, r)
    assert q['median_coverage'] == 1 and q['is_complete']
    r['Results'] += [{'number': '3', 'laps': '6', 'position': '3'}]
    assert quality_row(df, r)['median_coverage'] == .5
    second = df.assign(driver_number=2, driver='NOR', stint=1, pit_in_filled=False)
    assert quality_row(pd.concat([df, second]), race())['n_distinct_strategies'] == 2


@pytest.mark.parametrize('wet_count, expected', [(1, False), (2, True)])
def test_wet_threshold(wet_count, expected):
    df = pd.concat([fixture(), fixture().iloc[:4]], ignore_index=True)
    df.loc[:wet_count - 1, 'compound'] = 'INTERMEDIATE'
    assert quality_row(fill_pit_in(df), race())['is_wet'] == expected


def test_coverage_threshold_and_missing_time():
    df = pd.concat([fixture()] * 4, ignore_index=True).iloc[:20].copy()
    df.loc[0, 'lap_time_s'] = None
    r = race()
    r['Results'][0]['laps'] = '20'
    assert quality_row(fill_pit_in(df), r)['is_complete']
    df.loc[1, 'lap_time_s'] = None
    assert not quality_row(fill_pit_in(df), r)['is_complete']


def test_model_schema_winner_laps_and_stable_ids():
    df = clean_laps(fixture())
    r = race()
    r['Results'].insert(0, {'number': '2', 'laps': '5', 'position': '2'})
    out = model_rows(df, r)
    assert out.columns.tolist() == MODEL_COLUMNS
    assert out.lap.tolist() == [4, 5, 6]
    assert out.laps_remaining.tolist() == [2, 1, 0]
    assert out.stint_uid.unique().tolist() == ['2024-1-1-2']
    assert out.race_id.unique().tolist() == ['2024-1']
