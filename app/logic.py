"""Editable dry-race strategies. The page and tests use the same validation."""
import re

import pandas as pd

from f1a.sim import COMPOUNDS, Strategy

TYRES = {name[0]: name for name in COMPOUNDS}
COLUMNS = ["策略名称", "配方顺序", "进站圈"]


def strategy_row(strategy):
    compounds = [strategy.start_compound] + [c for _, c in strategy.stops]
    return dict(zip(COLUMNS, [strategy.name, "-".join(c[0] for c in compounds),
                              ",".join(str(lap) for lap, _ in strategy.stops)]))


def parse_strategy(name, sequence, stops, laps):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("请填写策略名称")
    tokens = str(sequence).upper().strip().split("-")
    if any(token.strip() not in TYRES for token in tokens):
        raise ValueError("配方请使用 H、M、S，以连字符分隔")
    tyres = [TYRES[token.strip()] for token in tokens]
    fields = [] if pd.isna(stops) or not str(stops).strip() else str(stops).replace("，", ",").split(",")
    if len(fields) != len(tyres) - 1:
        raise ValueError("进站次数须等于配方段数减一")
    if any(not re.fullmatch(r"[0-9]+", value.strip()) for value in fields):
        raise ValueError("进站圈须为以逗号分隔的整数")
    strategy = Strategy(name.strip(), tyres[0], list(zip(map(int, fields), tyres[1:])))
    strategy.validate(laps)
    return strategy


def edited_strategies(table, laps):
    results, errors, names = [], [], set()
    if table.empty:
        errors.append("请至少保留一个策略")
    for index, values in enumerate(table[COLUMNS].itertuples(index=False, name=None), 1):
        try:
            strategy = parse_strategy(*values, laps)
            if strategy.name in names:
                raise ValueError("策略名称重复")
            names.add(strategy.name)
            results.append(strategy)
        except ValueError as exc:
            errors.append(f"第 {index} 行 · {exc}")
    return results, errors


def default_strategies(optimum, finishers, laps):
    """Use minimum simulated mean by stop count, then an observed modal strategy."""
    rows, notes = [], []
    valid = optimum.dropna(subset=["mean"]).sort_values("mean", kind="stable")
    for count, name in ((1, "最优一停"), (2, "最优两停")):
        candidates = valid[valid.sequence.str.count("-") == count]
        if candidates.empty:
            notes.append(f"该站无可行的{name}方案")
            continue
        best = candidates.iloc[0]
        strategy = parse_strategy(name, best.sequence, ",".join(map(str, best.stops)), laps)
        rows.append(strategy_row(strategy))
    if finishers is None:
        return pd.DataFrame(rows, columns=COLUMNS), notes
    top = finishers[finishers.finisher & finishers.finish_position.between(1, 10)].sort_values("finish_position")
    counts = top.strategy.dropna().value_counts()
    if not counts.empty:
        modal = set(counts[counts == counts.max()].index)
        for car in top[top.strategy.isin(modal)].itertuples():
            try:
                strategy = parse_strategy(f"前十主流 · {car.driver}", "-".join(car.strategy),
                                          str(car.pit_laps).replace(";", ","), laps)
            except ValueError:
                continue
            rows.append(strategy_row(strategy))
            break
    if not any(row["策略名称"].startswith("前十主流") for row in rows):
        notes.append("该站前十主流实际策略缺少合规干胎方案，已省略该行")
    return pd.DataFrame(rows, columns=COLUMNS), notes


SCENARIOS = ("全赛道基线", "本站出现率", "无安全车")


def scenario_race(root, rid, scenario):
    from f1a.sim import RaceConfig

    if scenario not in SCENARIOS:
        raise ValueError("未知安全车情景")
    race = RaceConfig.from_project(root, rid, fcy_by_circuit=scenario == "本站出现率")
    return race.fcy_rate_scale(0) if scenario == "无安全车" else race


def display_sequence(sequence):
    """Conventional notation for the default tyre multiset, independent of pit laps."""
    from itertools import permutations

    tyres = sequence.split("-")
    def preference(order):
        return (order[0] == "H", order[0] == order[1],
                "H" in order and "H" not in order[1:-1],
                sum(a == b for a, b in zip(order, order[1:])),
                sum(a != b for a, b in zip(order, tyres)), order)
    return "-".join(min(permutations(tyres), key=preference))


def paired_summary(totals, baseline):
    """Rows are common draws, columns are strategy names; subtract before quantiles."""
    diff = totals.sub(totals[baseline], axis="index")
    result = diff.quantile([.025, .25, .5, .75, .975]).T
    result.columns = ["P2.5", "Q1", "P50", "Q3", "P97.5"]
    result["配对均差"] = diff.mean()
    return result.rename_axis("策略").reset_index()


def conclusion(name, baseline, mean, lo, hi):
    if mean == 0:
        comparison = f"{name}与{baseline}平均用时相同"
    else:
        comparison = f"{name}平均比{baseline}{'快' if mean < 0 else '慢'} {abs(mean):.1f} 秒"
    interval = f"95% 配对差区间为 {lo:+.1f}～{hi:+.1f} 秒"
    if hi < 0 or lo > 0:
        verdict = "该情景下显著更快" if hi < 0 else "该情景下显著更慢"
    elif lo == hi == 0:
        verdict = "中央 95% 配对差均为 0" if mean == 0 else "区间包含 0"
    else:
        verdict = "区间包含 0，差别在模拟波动范围内"
    return f"{comparison}，{interval}，{verdict}。"


def position_gain(actual_positions, candidate_positions):
    """Positive values mean a better finish than the same driver's observed strategy."""
    return (actual_positions - candidate_positions).mean()
