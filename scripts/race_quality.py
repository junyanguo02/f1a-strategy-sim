"""Build T007 outputs offline from the accepted T005 cache; run at project root."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import pandas as pd
from f1a.clean import clean_laps
from f1a.pit import validate_pit_in
from scripts.build_cache import atomic_write
from scripts.data_quality import table
from scripts.reconcile import YEARS, PROCESSED

MODEL_COLUMNS = ['year', 'round', 'event', 'circuit_id', 'driver', 'driver_number',
                 'team', 'lap', 'total_laps', 'laps_remaining', 'compound',
                 'tyre_life', 'stint', 'lap_time_s', 'race_id', 'stint_uid']


def cached_race(year, number):
    """Read frozen Jolpica race metadata and results, with no network fallback."""
    path = Path('data/raw/jolpica') / f'{year}_{number}_results_limit=100.json'
    race = json.loads(path.read_text())['MRData']['RaceTable']['Races'][0]
    if (int(race['season']), int(race['round'])) != (year, number):
        raise ValueError(f'Wrong identity in {path}')
    return race


def quality_row(laps, race):
    """Compute coverage over positive-lap result drivers and observed strategies."""
    timed = laps.groupby('driver_number')['lap_time_s'].count()
    coverage = [timed.get(int(r['number']), 0) / int(r['laps'])
                for r in race['Results'] if int(r['laps']) > 0]
    median = float(pd.Series(coverage, dtype=float).median())
    strategies = set()
    for _, driver in laps.groupby('driver_number'):
        driver = driver.sort_values('lap')
        stops = tuple(driver.loc[driver.pit_in_filled.fillna(False), 'lap'].astype(int))
        # One compound per observed stint, retaining consecutive same-compound stints.
        compounds = tuple(driver.dropna(subset=['stint']).drop_duplicates('stint')
                          ['compound'].astype('string').fillna('UNKNOWN'))
        strategies.add((stops, compounds))
    wet = laps.compound.astype('string').str.strip().str.upper().isin(['WET', 'INTERMEDIATE'])
    return dict(year=int(race['season']), round=int(race['round']), event=race['raceName'],
                n_drivers=laps.driver_number.nunique(), median_coverage=median,
                is_complete=median >= .95, is_wet=bool(wet.mean() > .10),
                n_distinct_strategies=len(strategies))


def model_rows(laps, race):
    """Select clean rows and attach race and stint identities without reindexing laps."""
    out = laps.loc[laps.is_clean].copy()
    winner = [r for r in race['Results'] if str(r['position']) == '1']
    if len(winner) != 1:
        raise ValueError('Expected exactly one Jolpica winner')
    out['event'] = race['raceName']
    out['circuit_id'] = race['Circuit']['circuitId']
    out['total_laps'] = int(winner[0]['laps'])
    out['laps_remaining'] = out.total_laps - out.lap
    out['race_id'] = out.year.astype(str) + '-' + out['round'].astype(str)
    if out.stint.isna().any():
        raise ValueError('Clean row has no stint; task owner must decide')
    out['stint_uid'] = (out.race_id + '-' + out.driver_number.astype(str)
                        + '-' + out.stint.astype('Int64').astype(str))
    return out[MODEL_COLUMNS]


def tyre_example(laps):
    """Verify Bahrain VER stint starts directly against cached OpenF1 stint JSON."""
    example = laps.loc[(laps.year == 2024) & (laps['round'] == 1) & (laps.driver == 'VER')]
    sessions = json.loads(Path('data/raw/openf1/sessions_session_name=Race&year=2024.json').read_text())
    session = [s for s in sessions if s['country_name'] == 'Bahrain']
    if len(session) != 1:
        raise ValueError('Bahrain race session is not unique')
    key = session[0]['session_key']
    raw = json.loads(Path(f'data/raw/openf1/stints_session_key={key}.json').read_text())
    rows = []
    for stint in raw:
        if stint['driver_number'] != int(example.driver_number.iloc[0]):
            continue
        first = example.loc[example.lap == stint['lap_start']].iloc[0]
        assert first.tyre_life == stint['tyre_age_at_start'] + 1
        rows.append((stint['stint_number'], stint['compound'], stint['lap_start'],
                     stint['tyre_age_at_start'], int(first.tyre_life)))
    return sorted(rows)


def build_report(laps, quality, model, validation, example):
    """Generate every reported numerical result from input data or derived tables."""
    summary = []
    for year in YEARS:
        q = quality.loc[quality.year == year]
        m = model.loc[model.year == year]
        summary.append((year, len(q), int(q.is_complete.sum()), int(q.is_wet.sum()),
                        m.race_id.nunique(), len(m)))
    excluded = []
    for r in quality.itertuples(index=False):
        if not r.is_complete or r.is_wet:
            reasons = (['median_coverage < 0.95'] if not r.is_complete else []) + (['雨胎圈 > 10%'] if r.is_wet else [])
            excluded.append((r.year, r.round, r.event, f'{r.median_coverage:.6f}', '；'.join(reasons)))
    comparison = pd.read_csv(PROCESSED / 'reconcile.csv')
    plus = comparison.loc[comparison['diff'].eq(1)]
    empty_final = 0
    for r in plus.itertuples(index=False):
        d = laps.loc[(laps.year == r.year) & (laps['round'] == r.round) & (laps.driver == r.driver)]
        if not d.empty and pd.isna(d.sort_values('lap').iloc[-1].lap_time_s):
            empty_final += 1
    coverage = pd.read_csv(PROCESSED / 'reconcile_coverage.csv')
    unavailable = coverage.loc[coverage.status.ne('compared')]
    inferred = laps.loc[laps.pit_in_inferred].groupby(['year', 'round']).size()
    text = '# T007 建模表\n\n由 `.venv/bin/python scripts/race_quality.py` 离线生成。输入为 T005 四季 parquet、Jolpica/OpenF1 原始 JSON 缓存和对账 CSV；不修改输入。\n\n'
    text += '## 定义与复用\n\n复用本项目 `clean_laps`、`atomic_write`、Markdown 表格工具及既有 pandas/pyarrow；没有引入外部代码或新增依赖。查阅 [OpenF1 官方资料](https://openf1.org/) 和 [Jolpica results 官方文档](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/results.md)，保留现有 API 与缓存格式。\n\n'
    text += '覆盖率按每名 Jolpica 正圈数车手计算；该车无 OpenF1 有时圈记 0，零圈车手不入分母。雨胎为 INTERMEDIATE/WET，比例分母为该站全部观测圈。质量表覆盖已缓存比赛；未举行比赛另列，不视为干燥或完整。n_drivers 为观测到的车手数。策略为按圈号排序的 pit_in_filled 真值圈序列与每段 stint 的配方序列；同配方连续 stint 不合并，无 stint 的未完成圈不添加配方。该字段是观测策略数量，不保证完整真实策略。\n\n'
    text += '仅整站 pit_in 全空时补推。每个已知 stint 的最后观测圈且非该车手最后观测圈标真；未知 stint 保留空值。pit_in_inferred 仅在补推为进站的圈标真。原 pit_in 不变。清洗先补推再执行 T005 规则并重算慢圈中位数。补推不是实测真值，退赛尾圈及红旗换胎可能造成误报；未增加任务单外规则。\n\n'
    text += '建模表仅保留完整、非雨战比赛中的 is_clean 圈。total_laps 取 Jolpica position=1 的圈数；laps_remaining = total_laps - lap（本圈结束后的剩余圈数）；race_id 使用 year-round，stint_uid 使用 year-round-driver_number-stint。原圈号和胎龄不重排。\n\n'
    text += '## 赛季汇总\n\n' + table(['year', '缓存分站', '完整分站', '雨战分站', '建模分站', '建模圈数'], summary)
    text += f'\n\n建模总计：{len(model)} 圈，{model.race_id.nunique()} 站，{model.stint_uid.nunique()} 个 stint。\n\n## 被剔除分站\n\n'
    text += table(['year', 'round', 'event', 'median_coverage', '原因'], excluded)
    text += '\n\n未进入缓存的赛历分站（沿用 T005 对账状态）：\n\n' + table(coverage.columns, unavailable.itertuples(index=False, name=None))
    text += '\n\n## 补推数量与验证\n\n' + table(['year', 'round', '补推进站圈'], [(y, r, n) for (y, r), n in inferred.items()])
    text += '\n\n2024 全季按 (year, round, driver_number, lap) 比较，所有已知 pit_in 圈进入验证，不预先清洗、不排除雨战；缺失标签不视为负例。\n\n'
    text += table(validation.keys(), [validation.values()])
    text += '\n\n## tyre_life 约定\n\n现有 assemble 公式：`tyre_age_at_start + lap - lap_start + 1`；每段第一圈等于起始胎龄加 1，表示完成该圈后的胎龄。2024 巴林 VER 原始 stint 缓存与 parquet 对照：\n\n'
    text += table(['stint', 'compound', 'lap_start', 'tyre_age_at_start', '首圈 tyre_life'], example)
    text += f'\n\n## 保留的对账差异\n\n{len(plus)} 处 +1 中 {empty_final} 处最后观测圈的圈时为空；Jolpica 圈数为 0 的记录总计 {int(comparison.laps_jolpica.eq(0).sum())} 处，其中差值大于 +1 的有 {int((comparison.laps_jolpica.eq(0) & comparison["diff"].gt(1)).sum())} 处（对应任务单所述 5 处）；其余包括 +1 和双方均为 0 的记录。原数据和 reconcile.csv 不改；无圈时圈仍由 clean_laps 排除，零圈车手不参与覆盖率中位数。\n'
    return text, summary


def main():
    seasons = [pd.read_parquet(PROCESSED / f'laps_{year}.parquet') for year in YEARS]
    laps = clean_laps(pd.concat(seasons, ignore_index=True))
    if laps.duplicated(['year', 'round', 'driver_number', 'lap']).any():
        raise ValueError('Duplicate lap identity')
    quality, models = [], []
    for (year, number), race_laps in laps.groupby(['year', 'round'], sort=True):
        race = cached_race(int(year), int(number))
        q = quality_row(race_laps, race)
        quality.append(q)
        if q['is_complete'] and not q['is_wet']:
            models.append(model_rows(race_laps, race))
    quality = pd.DataFrame(quality)
    model = pd.concat(models, ignore_index=True).sort_values(['year', 'round', 'driver_number', 'lap']).reset_index(drop=True)
    validation = validate_pit_in(laps.loc[laps.year == 2024])
    example = tyre_example(laps)
    report, summary = build_report(laps, quality, model, validation, example)
    atomic_write(quality, PROCESSED / 'race_quality.csv')
    atomic_write(laps, PROCESSED / 'laps_pit_filled.parquet')
    atomic_write(model, PROCESSED / 'model_laps.parquet')
    Path('notes/T007-model-table.md').write_text(report)
    print(table(['year', 'cached', 'complete', 'wet', 'model_races', 'model_laps'], summary))
    print(f'model_laps: {len(model)} rows; {model.race_id.nunique()} races')
    print('2024 pit inference:', json.dumps(validation))
    print('excluded:', quality.loc[~quality.is_complete | quality.is_wet, ['year', 'round']].to_dict('records'))


if __name__ == '__main__':
    main()
