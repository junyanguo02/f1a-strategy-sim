"""Read-back verification for T007 outputs; run after race_quality.py."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def main():
    root = Path('data/processed')
    q = pd.read_csv(root / 'race_quality.csv')
    m = pd.read_parquet(root / 'model_laps.parquet')
    f = pd.read_parquet(root / 'laps_pit_filled.parquet')
    original = pd.concat([pd.read_parquet(root / f'laps_{y}.parquet')
                          for y in (2023, 2024, 2025, 2026)], ignore_index=True)
    pd.testing.assert_frame_equal(f[original.columns], original)
    ids = ['year', 'round', 'driver_number', 'lap']
    assert not m.duplicated(ids).any()
    assert m.notna().all().all()
    assert (m.laps_remaining == m.total_laps - m.lap).all()
    assert (m.laps_remaining >= 0).all()
    admitted = q.loc[q.is_complete & ~q.is_wet, ['year', 'round']]
    expected = f.loc[f.is_clean].merge(admitted, on=['year', 'round'], validate='many_to_one')
    assert set(map(tuple, expected[ids].values)) == set(map(tuple, m[ids].values))
    assert not m.merge(f[ids + ['pit_in_filled']], on=ids).pit_in_filled.fillna(False).any()
    for (year, number), group in f.groupby(['year', 'round']):
        raw = json.loads(Path(f'data/raw/jolpica/{year}_{number}_results_limit=100.json').read_text())
        race = raw['MRData']['RaceTable']['Races'][0]
        rates = []
        for r in race['Results']:
            if int(r['laps']) > 0:
                count = group.loc[group.driver_number.eq(int(r['number'])), 'lap_time_s'].notna().sum()
                rates.append(count / int(r['laps']))
        actual = q.loc[(q.year == year) & (q['round'] == number)].iloc[0]
        assert abs(actual.median_coverage - pd.Series(rates).median()) < 1e-12
        if group.pit_in.isna().all():
            expected_stops = set()
            for driver, d in group.groupby('driver_number'):
                for _, stint in d.groupby('stint'):
                    if stint.lap.max() != d.lap.max():
                        expected_stops.add((driver, stint.lap.max()))
            assert expected_stops == set(map(tuple, group.loc[group.pit_in_inferred, ['driver_number', 'lap']].values))
        else:
            pd.testing.assert_series_equal(group.pit_in.astype('boolean'), group.pit_in_filled, check_names=False)
            assert not group.pit_in_inferred.any()
    manifest = Path('work/t007/protected_hashes.json')
    if manifest.exists():
        baseline = json.loads(manifest.read_text())
        changed = [p for p, digest in baseline.items()
                   if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
        assert not changed, changed
        print(f'Protected baseline: {len(baseline)} files unchanged')
    print(f'Read-back PASS: {len(q)} quality rows; {len(m)} model rows; {m.race_id.nunique()} races; {m.stint_uid.nunique()} stints')
    print('Original columns, clean membership, pit boundaries, all-race coverage, keys and null checks: PASS')


if __name__ == '__main__':
    main()
