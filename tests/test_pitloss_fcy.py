import pandas as pd
import pytest
from f1a.fcy import split_episodes, start_bin
from f1a.pitloss import pit_losses, summarize_losses, pair_flag
from scripts.build_pitloss_fcy import cached_session
import json


def message(lap, state, second=None):
    flags = {'GREEN': 'CLEAR', 'RED': 'RED', 'FINISH': 'CHEQUERED'}
    return dict(lap_number=lap, date=f'2024-01-01T00:00:{second if second is not None else lap:02d}Z',
                message={'SC': 'SAFETY CAR DEPLOYED', 'VSC': 'VIRTUAL SAFETY CAR DEPLOYED',
                         'ENDING': 'VIRTUAL SAFETY CAR ENDING'}.get(state, state),
                scope='Track' if state in flags else None, flag=flags.get(state))


def test_merge_continuous_repeated_deployment_and_ending():
    result = split_episodes([message(2, 'VSC'), message(3, 'VSC'),
                             message(4, 'ENDING'), message(5, 'GREEN')], 10)
    assert result[['type', 'start_lap', 'end_lap', 'n_laps']].values.tolist() == [['VSC', 2, 5, 4]]
    assert result.start_frac.iloc[0] == .2


def test_direct_vsc_sc_and_red_break():
    feed = [message(2, 'VSC'), message(3, 'SC'), message(4, 'RED'),
            message(4, 'SC', 5), message(6, 'FINISH')]
    e = split_episodes(feed, 6)
    assert e.type.tolist() == ['VSC', 'SC', 'SC']
    assert e.followed_by.tolist() == [True, False, False]
    assert e.has_red_flag.tolist() == [False, True, False]
    assert e.n_laps.tolist() == [2, 2, 3]


def test_clear_breaks_vsc_sc_even_same_lap_and_ignores_local_yellow():
    local = message(2, 'GREEN', 3); local['scope'] = 'Sector'
    e = split_episodes([message(1, 'VSC'), local, message(3, 'GREEN', 4),
                        message(3, 'SC', 5), message(4, 'FINISH', 6)], 4)
    assert e.followed_by.tolist() == [False, False]
    assert e.end_lap.tolist() == [3, 4]


def test_zero_events_and_unclosed_feed():
    assert split_episodes([message(1, 'GREEN'), message(5, 'FINISH')], 5).empty
    e = split_episodes([message(2, 'SC')], 5).iloc[0]
    assert (e.end_lap, e.end_reason, e.end_time) == (5, 'feed_end', None)
    with pytest.raises(ValueError): split_episodes([], 5)
    bad = message(1, 'SC'); bad['lap_number'] = None
    with pytest.raises(ValueError): split_episodes([bad], 5)


@pytest.mark.parametrize('lap,label', [(1,'第1圈'), (2,'≤20%'), (3,'≤40%'), (4,'≤40%'),
    (6,'≤60%'), (8,'≤80%'), (10,'≤100%')])
def test_start_bins(lap, label):
    assert start_bin(lap, 10) == label


def fixture(inferred=False):
    f = pd.DataFrame([dict(year=2024, round=1, driver_number=1, driver='VER', lap=n,
        lap_time_s=100., pit_in_filled=n==5, pit_in_inferred=inferred and n==5,
        pit_out=n==6, is_clean=n not in [5,6], safety_car=False, virtual_safety_car=False)
        for n in range(1, 11)])
    f.loc[f.lap.eq(5),'lap_time_s'] = 125.
    f.loc[f.lap.eq(6),'lap_time_s'] = 135.
    return f


def test_loss_reference_and_no_input_mutation():
    f = fixture(); original = f.copy(deep=True)
    r = pit_losses(f, 'test').iloc[0]
    assert r.loss_s == 60 and r.reference_n == 6 and r.reference_time_s == 100
    assert r.reference_laps == '2,3,4,7,8,9'
    assert r.usable and r.condition == '正常'
    pd.testing.assert_frame_equal(f, original)


def test_nearest_clean_not_only_three_adjacent_and_pooled_median():
    f = fixture()
    f.loc[f.lap.eq(4), 'is_clean'] = False
    f.loc[f.lap.isin([1,2,3]), 'lap_time_s'] = [90., 95., 100.]
    f.loc[f.lap.isin([7,8,9]), 'lap_time_s'] = [110.,120.,130.]
    r = pit_losses(f, 'test').iloc[0]
    assert r.reference_laps == '1,2,3,7,8,9'
    assert r.reference_time_s == 105 and r.loss_s == 50


def test_fewer_reference_laps_one_side_and_none():
    f = fixture();f['is_clean'] = f.lap.eq(1)
    assert pit_losses(f, 'x').iloc[0].reference_n == 1
    f['is_clean'] = False
    r = pit_losses(f, 'x').iloc[0]
    assert not r.usable and pd.isna(r.loss_s) and r.unusable_reason == 'no_clean_reference'


def test_source_out_confirmation_and_inferred_override():
    f = fixture(); f['pit_out'] = False
    r = pit_losses(f, 'x').iloc[0]
    assert not r.usable and r.unusable_reason == 'unconfirmed_source_out_lap'
    f.loc[f.lap.eq(5), 'pit_in_inferred'] = True
    r = pit_losses(f, 'x').iloc[0]
    assert r.usable and r.out_lap_inferred and r.loss_s == 60
    f = f.loc[f.lap.ne(6)]
    r = pit_losses(f, 'x').iloc[0]
    assert not r.usable and 'missing_next_lap' in r.unusable_reason


def test_no_cross_lap_search_missing_time_negative_kept():
    f = fixture(); f.loc[f.lap.eq(6),'lap_time_s'] = None
    assert not pit_losses(f, 'x').iloc[0].usable
    f.loc[f.lap.isin([5,6]),'lap_time_s'] = 90
    r = pit_losses(f, 'x').iloc[0]
    assert r.usable and r.loss_s == -20


def test_mixed_priority_and_unknown_not_normal():
    f = fixture(); f.loc[f.lap.eq(5),'safety_car'] = True
    f.loc[f.lap.eq(6),'virtual_safety_car'] = True
    r = pit_losses(f, 'x').iloc[0]
    assert r.mixed_sc_vsc and r.condition == 'SC'
    f['safety_car'] = f.safety_car.astype('boolean')
    f.loc[f.lap.eq(6),'safety_car'] = pd.NA
    r = pit_losses(f, 'x').iloc[0]
    assert r.under_sc and r.mixed_sc_vsc and r.condition == 'unknown' and r.usable
    assert pair_flag(False, False) is False
    assert pd.isna(pair_flag(False, pd.NA))


def test_summary_iqr_low_n_empty_and_exclude_inferred():
    p = pd.concat([pit_losses(fixture(), 'x')]*5, ignore_index=True)
    p['loss_s'] = [10.,20.,30.,40.,50.]
    p.loc[0,'out_lap_inferred'] = True
    s = summarize_losses(p, ['x','y']).set_index(['circuit_id','condition'])
    assert (s.loc[('x','正常'), ['n','median_s','iqr_s','low_n']].tolist()) == [5,30,20,False]
    assert s.loc[('y','正常'), 'n'] == 0 and pd.isna(s.loc[('y','正常'), 'median_s'])
    strict = summarize_losses(p,['x'],True).set_index('condition')
    assert strict.loc['正常','n'] == 4 and strict.loc['正常','low_n']


def test_cached_session_local_date_and_duplicate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path/'data/raw/openf1';folder.mkdir(parents=True)
    path = folder/'sessions_session_name=Race&year=2024.json'
    session = dict(date_start='2024-11-24T06:00:00Z', gmt_offset='-08:00:00',session_key=1)
    path.write_text(json.dumps([session]))
    race = dict(season='2024', round='22', date='2024-11-23')
    assert cached_session(race)['session_key'] == 1
    path.write_text(json.dumps([session,session]))
    with pytest.raises(ValueError):cached_session(race)
