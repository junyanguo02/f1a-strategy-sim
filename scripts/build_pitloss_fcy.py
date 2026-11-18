"""Offline T009 tables and generated report. Run from the project root."""
from pathlib import Path
from datetime import timedelta
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import pandas as pd
from f1a.fcy import split_episodes, start_bin
from f1a.pitloss import pit_losses, summarize_losses, CONDITIONS
from scripts.race_quality import cached_race
from scripts.data_quality import table
from scripts.build_cache import atomic_write

OUT = Path('data/processed')
BINS = ['第1圈', '≤20%', '≤40%', '≤60%', '≤80%', '≤100%']
# Transcribed from the required reading, section 6.2, Heilmeier 2020b Table 6 p.14.
PAPER_TABLE6 = {'catalunya': ('Catalunya', 19.04, 10.03, 7.88),
                'albert_park': ('Melbourne', 17.85, 12.95, 12.61),
                'monza': ('Monza', 20.60, 15.42, 10.16),
                'suzuka': ('Suzuka', 19.48, 15.05, 13.60)}


def cached_session(race):
    """Use the T003 unique UTC/local-date matching convention without fetching."""
    year = int(race['season'])
    sessions = json.loads(Path(f'data/raw/openf1/sessions_session_name=Race&year={year}.json').read_text())
    date = pd.Timestamp(race['date']).date()
    matches = []
    for s in sessions:
        utc = pd.Timestamp(s['date_start'])
        offset = s.get('gmt_offset') or '00:00:00'
        hours, minutes, seconds = map(int, offset.lstrip('+-').split(':'))
        delta = timedelta(seconds=hours*3600+minutes*60+seconds)
        local = utc - delta if offset.startswith('-') else utc + delta
        if not s.get('is_cancelled') and date in {utc.date(), local.date()}:
            matches.append(s)
    if len(matches) != 1:
        raise ValueError(f'Non-unique cached session for {year} R{race["round"]}')
    return matches[0]


def season_summary(races, episodes, pits):
    """Denominators include every cached race, including zero-event races."""
    rows = []
    for year, r in races.groupby('year'):
        p = pits.loc[pits.year == year]
        e = episodes.loc[episodes.year == year]
        rows.append(dict(year=int(year), races=len(r), n_sc=int(e.type.eq('SC').sum()),
            n_vsc=int(e.type.eq('VSC').sum()), episodes=len(e),
            sc_per_race=float(r.n_sc.mean()), vsc_per_race=float(r.n_vsc.mean()),
            pit_candidates=len(p), usable_pits=int(p.usable.sum()),
            usable_excluding_inferred=int((p.usable & ~p.out_lap_inferred).sum())))
    return pd.DataFrame(rows)


def condition_summary(pits):
    """Pooled usable medians, in both requested cohorts, including unknown."""
    rows = []
    for cohort, remove in [('全部可用', False), ('排除补推出站', True)]:
        frame = pits.loc[pits.usable & (~pits.out_lap_inferred if remove else True)]
        for condition in CONDITIONS:
            values = frame.loc[frame.condition == condition, 'loss_s']
            rows.append((cohort, condition, len(values), float(values.median()) if len(values) else None))
    return pd.DataFrame(rows, columns=['cohort', 'condition', 'n', 'median_s'])


def report_text(races, episodes, pits, summary, strict, seasons, conditions):
    """All observations calculated from produced tables; paper values explicit."""
    text = '''# T009 进站损失与安全车事件表

由 `.venv/bin/python scripts/build_pitloss_fcy.py` 离线生成；仅数据描述，不拟合策略或概率模型。

## 输入与复用

使用 T007 `laps_pit_filled.parquet` 的 is_clean 与补推标记，以及冻结的 OpenF1 race_control/sessions、Jolpica results 缓存。沿用 T003 的唯一 UTC/当地日期匹配、全场旗号含义；复用现有原子写出、Jolpica 读取和 Markdown 表格工具。没有下载数据、复制外部代码或新增依赖。
资料核对：[OpenF1 官方文档](https://openf1.org/docs/)；[官方 race_control 实现](https://github.com/br-g/openf1/blob/main/src/openf1/services/ingestor_livetiming/core/processing/collections/race_control.py)。无需引入 SDK，故无新增第三方代码许可证或维护成本。

## 事件判定规则

以全场 race_control 消息的 lap_number 作为比赛领先圈计数，不取最终冠军的个人圈序列，也不把所有车手受影响圈合并。按 UTC 时间排序；SC DEPLOYED 和 VSC DEPLOYED 开始对应阶段；同类型重复部署不切段，连续受影响圈合并。VSC ENDING 和 SC IN THIS LAP 不立即结束，等全场 GREEN/CLEAR、另一类部署、RED 或 CHEQUERED。

直接 VSC 转 SC 标 followed_by=True；中间有 GREEN/CLEAR 或红旗则 False。其他事件 followed_by=False。红旗立即截断活动事件，给这段 has_red_flag=True；红旗之后重新部署形成新段，红旗暂停时段不延长旧段。没有活动 SC/VSC 时的红旗不凭空创建事件；has_red_flag 是该事件被红旗截断，并非整场是否有红旗。

start_lap 为部署消息的领先圈，end_lap 为结束消息的领先圈（不超过 Jolpica 冠军实际圈数）。两端均计入，n_laps=end_lap-start_lap+1，因此表示触及的领先圈数，不是完整覆盖圈或秒数。转换发生在圈中时，两段可以触及同一圈，时长不可直接相加当作互不重叠的全场受影响圈数。start_frac=start_lap/total_laps；六段互斥，首圈优先，其他依次按 20%、40%、60%、80%、100% 上界分箱。

数据流结束仍未闭合的阶段延伸到 total_laps，标 end_reason=feed_end、end_time 空值供核查。空 race_control 流或关键状态消息缺圈号会报错，不当成无事件。零部署比赛在比赛级汇总中保留 0。只识别明确部署消息；列数为缓存可识别事件，并非通过录像验证的事件真值。消息缺失或非标准的安全车发车/重启文字可能导致漏记。

所有已缓存比赛均纳入，不套用 T007 的完整/干燥建模筛选。比赛级表附上 is_complete/is_wet 供使用者识别；未举行或缺缓存比赛不进入场均次数分母。2025 迈阿密的不完整圈表不补齐。全场旗号按 T003 约定处理，进站状态沿用已验收圈表，不在本任务重写历史状态。

## 进站与参考圈规则

每个 pit_in_filled=True 圈保留一行。只配对同车手 lap+1，不向后跨圈查找。pit_in_inferred=True 的候选按决定允许配对下一圈，并标 out_lap_inferred=True；数据源进站要求下一圈 pit_out=True，否则 usable=False。缺下一圈、两圈任一无圈时、全无参考圈也不可用，loss_s 保留空值；所有原因用竖线拼接。

参考圈取同车手同站在进站圈之前最近至多三个 is_clean 圈，以及出站圈之后最近至多三个 is_clean 圈；“各三个”指最近三个干净圈，而非只搜索相邻三个圈号。两侧合并后取中位数，不先分侧求平均；不足三个照用现有圈。排除本次进/出站两圈，不跨比赛；附 reference_laps、两侧数量、参考中位数便于读回。干净圈仍按 T007 既有标签，不另改规则。

loss_s=进站圈圈时+出站圈圈时−2×参考中位数。零值、负值、红旗影响的长圈及其他极端值不截断、不删除；usable 仅表示上述配对和算术输入可用，不代表结果是纯粹维修区损失。

under_sc/under_vsc 为两圈状态的三值逻辑或：任一真则真，全假才假，否则空。mixed_sc_vsc 为两列逻辑与。分类时两圈四个 SC/VSC 输入任一缺失则 unknown；完整已知时 SC 优先于 VSC，均假才正常。混合标记独立保留。unknown 不并入正常，usable 与状态分类分开。

汇总只计 usable=True；同时交付全部可用以及排除 out_lap_inferred 两版。每个赛道保留正常/VSC/SC/unknown 四格，包括 0 次格；n<5 标 low_n，0 次的中位数、四分位距为空。四分位数使用 pandas 默认线性插值，IQR=Q3−Q1，单位秒。

## 赛季事件与进站汇总

'''
    text += table(seasons.columns, seasons.itertuples(index=False, name=None))
    text += '\n\n## 按条件损失中位数\n\n' + table(conditions.columns, conditions.itertuples(index=False, name=None))
    mixed = pits.loc[pits.mixed_sc_vsc.fillna(False)]
    mixed_rows = []
    for label, frame in [('全部混合候选', mixed), ('排除补推出站的混合候选', mixed.loc[~mixed.out_lap_inferred])]:
        values = frame.loc[frame.usable, 'loss_s']
        mixed_rows.append((label, len(frame), len(values), float(values.median()) if len(values) else None))
    text += '\n\n## SC 与 VSC 混合进站\n\n' + table(['范围', '候选次数', '可用次数', '可用损失中位数 秒'], mixed_rows)
    text += '\n\n## 时长分布\n\n时长为触及圈数，比例在对应赛季和类型内计算。\n\n'
    duration_rows, bin_rows = [], []
    for (year, kind), group in episodes.groupby(['year', 'type']):
        for duration, n in group.n_laps.value_counts().sort_index().items():
            duration_rows.append((year, kind, duration, n, float(n/len(group))))
    for year in sorted(races.year.unique()):
        for kind in ['SC', 'VSC']:
            group = episodes.loc[(episodes.year == year) & (episodes.type == kind)]
            counts = group.start_bin.value_counts()
            for label in BINS:
                n = int(counts.get(label, 0))
                bin_rows.append((year, kind, label, n, n/len(group) if len(group) else None))
    text += table(['year', 'type', 'n_laps', '次数', '比例'], duration_rows)
    text += '\n\n## 起点六段分布\n\n' + table(['year', 'type', '区间', '次数', '比例'], bin_rows)
    text += '\n\n## 每赛道进站损失 全部可用\n\n' + table(summary.columns, summary.itertuples(index=False, name=None))
    text += '\n\n## 每赛道进站损失 排除补推出站\n\n' + table(strict.columns, strict.itertuples(index=False, name=None))
    comparison = []
    for circuit, (name, *values) in PAPER_TABLE6.items():
        for i, condition in enumerate(['正常', 'VSC', 'SC']):
            a = summary.loc[(summary.circuit_id == circuit) & (summary.condition == condition)].iloc[0]
            b = strict.loc[(strict.circuit_id == circuit) & (strict.condition == condition)].iloc[0]
            comparison.append((name, circuit, condition, values[i], a.n, a.median_s, b.n, b.median_s))
    text += '\n\n## Heilmeier 2020b 表 6 并列对照\n\n文献值来自必读笔记第 6.2 节转录的原文表 6（p.14），单位秒；这些固定文献值不由当前圈表估计。名称对应 Catalunya→catalunya、Melbourne→albert_park、Monza→monza、Suzuka→suzuka。仅并列呈现。\n\n'
    text += table(['文献赛道', 'circuit_id', '条件', '文献值', '全部可用 n', '全部中位数', '排除补推 n', '排除补推中位数'], comparison)
    reasons = pits.loc[~pits.usable, 'unusable_reason'].str.split('|', regex=False).explode().value_counts()
    text += '\n\n## 可核查的限制与计数\n\n' + table(['不可用原因 可重叠', '次数'], reasons.items())
    text += f'\n\n事件 {len(episodes)} 段；零事件比赛 {int((races.n_sc+races.n_vsc).eq(0).sum())} 场；红旗截断 {int(episodes.has_red_flag.sum())} 段；直接 VSC→SC {int(episodes.followed_by.sum())} 段；流末补闭合 {int(episodes.end_reason.eq("feed_end").sum())} 段。'
    text += f'\n\n进站候选 {len(pits)} 次，补推进站 {int(pits.pit_in_inferred.sum())} 次，可用 {int(pits.usable.sum())} 次；其中负损失 {int((pits.usable & pits.loss_s.lt(0)).sum())} 次，unknown 类可用 {int((pits.usable & pits.condition.eq("unknown")).sum())} 次，均保留。'
    coverage = pd.read_csv(OUT / 'reconcile_coverage.csv')
    text += '\n\n未进入本次缓存分析的分站：\n\n' + table(coverage.columns, coverage.loc[coverage.status.ne('compared')].itertuples(index=False, name=None))
    return text + '\n'


def main():
    laps = pd.read_parquet(OUT / 'laps_pit_filled.parquet')
    quality = pd.read_csv(OUT / 'race_quality.csv').set_index(['year', 'round'])
    episodes, race_rows, stops = [], [], []
    for (year, number), group in laps.groupby(['year', 'round'], sort=True):
        race = cached_race(int(year), int(number))
        session = cached_session(race)
        winner = [r for r in race['Results'] if str(r['position']) == '1']
        if len(winner) != 1:
            raise ValueError('Expected unique winner')
        total = int(winner[0]['laps'])
        key = session['session_key']
        feed = json.loads(Path(f'data/raw/openf1/race_control_session_key={key}.json').read_text())
        ep = split_episodes(feed, total)
        meta = dict(year=int(year), round=int(number), event=race['raceName'],
                    circuit_id=race['Circuit']['circuitId'])
        for name, value in meta.items():
            ep[name] = value
        ep['total_laps'] = total
        ep['session_key'] = key
        ep['start_bin'] = [start_bin(int(v), total) for v in ep.start_lap]
        episodes.append(ep)
        q = quality.loc[(year, number)]
        race_rows.append(dict(**meta, total_laps=total, n_sc=int(ep.type.eq('SC').sum()),
                              n_vsc=int(ep.type.eq('VSC').sum()), session_key=key,
                              is_complete=bool(q.is_complete), is_wet=bool(q.is_wet)))
        stops.append(pit_losses(group, meta['circuit_id']))
    eps = pd.concat([e for e in episodes if not e.empty], ignore_index=True)
    eps = eps.astype({c: 'int64' for c in ['year', 'round', 'start_lap', 'end_lap', 'n_laps', 'total_laps', 'session_key']})
    eps = eps.astype({'start_frac': 'float64', 'followed_by': 'bool', 'has_red_flag': 'bool'})
    first = ['year', 'round', 'event', 'circuit_id', 'type', 'start_lap', 'end_lap', 'n_laps', 'start_frac', 'followed_by', 'has_red_flag']
    eps = eps[first + [c for c in eps if c not in first]]
    races = pd.DataFrame(race_rows)
    pits = pd.concat(stops, ignore_index=True)
    summary = summarize_losses(pits, races.circuit_id.unique())
    strict = summarize_losses(pits, races.circuit_id.unique(), exclude_inferred=True)
    seasons = season_summary(races, eps, pits)
    conditions = condition_summary(pits)
    report = report_text(races, eps, pits, summary, strict, seasons, conditions)
    for frame, filename in [(eps, 'fcy_episodes.parquet'), (races, 'fcy_race_summary.csv'),
            (pits, 'pit_loss.parquet'), (summary, 'pit_loss_by_circuit.csv'),
            (strict, 'pit_loss_by_circuit_no_inferred.csv')]:
        atomic_write(frame, OUT / filename)
    Path('notes/T009-pitloss-fcy.md').write_text(report)
    print(seasons.to_string(index=False))
    print(conditions.to_string(index=False))
    print(f'Total: {len(eps)} episodes; {len(pits)} pit candidates; {int(pits.usable.sum())} usable')
    print(f'Mixed SC/VSC: {int(pits.mixed_sc_vsc.fillna(False).sum())}; usable median: {pits.loc[pits.usable & pits.mixed_sc_vsc.fillna(False), "loss_s"].median():.6f} s')
    print('Episode ends:', eps.end_reason.value_counts().to_dict())


if __name__ == '__main__':
    main()
