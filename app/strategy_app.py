from dataclasses import replace
from pathlib import Path
import sys
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from app.logic import (COLUMNS, SCENARIOS, conclusion, default_strategies, display_sequence,
                       edited_strategies, paired_summary, position_gain, scenario_race)
from f1a.sim import COMPOUNDS, TyreParams, common_draws, evaluate, optimize_stops, simulate
from f1a.sim import prerace
from f1a.sim.backtest import candidate_sequences
from f1a.sim.field import FieldParams, cars_from_project, evaluate_ego, field_draws, simulate_field
from f1a.sim.params import estimated_compounds

ML = ROOT / "data/processed/model_laps.parquet"
POST = ROOT / "models/tyre_v3"
BASE_SEED = 20261118
st.set_page_config(page_title="F1 策略模拟器", layout="wide")


@st.cache_resource
def model_laps():
    return pd.read_parquet(ML)


@st.cache_resource
def finishers():
    return pd.read_parquet(ROOT / "analysis/actual_strategies/strategies.parquet")


@st.cache_resource(max_entries=16)
def tyre_inputs(rid):
    year = int(rid.split("-")[0])
    posterior = POST / f"outputs31_{year}/posterior_core.npz"
    tyre = TyreParams.from_posterior(posterior, rid, [year], ML, n_draws=200, seed=BASE_SEED)
    with np.load(posterior) as data:
        fp = FieldParams(d_gap_lt1=float(data["d_gap_lt1"].mean()), d_gap_1to2=float(data["d_gap_1to2"].mean()))
    return tyre, fp


@st.cache_resource(max_entries=48)
def prior_inputs(rid, circuit=False):
    year, rnd = map(int, rid.split("-"))
    return prerace.build_inputs(ROOT, year, rnd, 200, BASE_SEED, circuit)


@st.cache_data(max_entries=48)
def prior_optimum(rid):
    return prerace.optimization(prior_inputs(rid), 1000, BASE_SEED, 2)


@st.cache_resource(max_entries=48)
def race_inputs(rid, scenario, mode):
    if mode == "赛前":
        inputs = prior_inputs(rid, scenario == "本站出现率")
        race = inputs.race.fcy_rate_scale(0) if scenario == "无安全车" else inputs.race
        return race, inputs.tyre, inputs.field_params
    return scenario_race(ROOT, rid, scenario), *tyre_inputs(rid)


@st.cache_data(max_entries=96)
def defaults(rid, mode):
    race, tyre, _ = race_inputs(rid, SCENARIOS[0], mode)
    if mode == "赛前":
        best, actual, notes = prior_optimum(rid), None, []
    else:
        year, rnd = map(int, rid.split("-"))
        actual = finishers()
        actual = actual[(actual.year == year) & (actual["round"] == rnd)]
        allowed = estimated_compounds(model_laps(), rid)
        notes = []
        if len(allowed) < 2:
            allowed = COMPOUNDS
            notes.append("本站独立估计的配方不足两种，默认优化包含赛季均值回退的配方。")
        best = optimize_stops(race, tyre, candidate_sequences(allowed), n_sims=1000, seed=BASE_SEED, step=2)
    table, warnings = default_strategies(best, actual, race.total_laps)
    table["常见顺序"] = [display_sequence(row[1]) if row[0] in ("最优一停", "最优两停") else row[1]
                        for row in table.itertuples(index=False, name=None)]
    return table, notes + warnings


@st.cache_resource(max_entries=16)
def field(rid, mode):
    if mode == "赛前":
        key = rid.replace("-", "_")
        path = ROOT / f"data/raw/jolpica/{key}_results_limit=100.json"
        if not path.is_file():
            raise prerace.Unavailable("missing_starting_grid")
        races = json.loads(path.read_text())["MRData"]["RaceTable"]["Races"]
        if not races or not races[0].get("Results"):
            raise prerace.Unavailable("missing_starting_grid")
        cars, _ = prerace.cars_from_prerace(races[0]["Results"], prior_inputs(rid).paces,
                                            prerace.fastest_plan(prior_optimum(rid)))
        if len(cars) < 8:
            raise prerace.Unavailable("insufficient_field")
        return cars
    return cars_from_project(ROOT, rid, finishers())


@st.cache_data(max_entries=32)
def assessment(rid, records, n, seed, driver, scenario, mode):
    race, tyre, fp = race_inputs(rid, scenario, mode)
    table = pd.DataFrame.from_records(records, columns=COLUMNS)
    strategies, errors = edited_strategies(table, race.total_laps)
    if errors:
        raise ValueError("；".join(errors))
    max_stints = max(6, 1 + max(len(s.stops) for s in strategies))
    totals = {}
    distribution = []
    if driver is None:
        summary = evaluate(race, tyre, strategies, n_sims=n, seed=seed)
        draws = common_draws(race, tyre, n, seed, max_stints=max_stints)
        for s in strategies:
            times = simulate(race, tyre, s, draws=draws)["total"]
            totals[s.name] = times
            q = np.quantile(times, [.05, .25, .5, .75, .95])
            distribution.append(dict(策略=s.name, **dict(zip(["P5", "Q1", "P50", "Q3", "P95"], q))))
    else:
        cars = field(rid, mode)
        ego = [c.name for c in cars].index(driver)
        summary = evaluate_ego(race, tyre, cars, ego, strategies, fp, n_sims=n, seed=seed)
        max_stints = max(max_stints, 1 + max(len(c.strategy.stops) for c in cars))
        draws = field_draws(race, tyre, len(cars), n, seed, max_stints=max_stints)
        actual_positions = simulate_field(race, tyre, cars, fp, draws=draws)["position"][:, ego]
        gains = {}
        for s in strategies:
            lineup = list(cars)
            lineup[ego] = replace(cars[ego], strategy=s)
            positions = simulate_field(race, tyre, lineup, fp, draws=draws)["position"][:, ego]
            gains[s.name] = position_gain(actual_positions, positions)
            ranks, counts = np.unique(positions, return_counts=True)
            distribution.extend(dict(策略=s.name, 名次=int(rank), 概率=count/n) for rank, count in zip(ranks, counts))
        summary["gain"] = summary.strategy.map(gains)
    return summary, pd.DataFrame(distribution), pd.DataFrame(totals)


def box_chart(dist, lo, hi, title, zero=False):
    bounds = dist.rename(columns={lo: "lower", hi: "upper"})
    base = alt.Chart(bounds).encode(
        y=alt.Y("策略:N", title=None, sort=None),
        color=alt.Color("策略:N", scale=alt.Scale(scheme="tableau10"), legend=None),
        tooltip=["策略:N"] + [alt.Tooltip(field=k, title=label, type="quantitative", format=".2f")
                             for k, label in (("lower", lo), ("Q1", "Q1"), ("P50", "中位数"),
                                              ("Q3", "Q3"), ("upper", hi))])
    def axis(field):
        return alt.X(field=field, type="quantitative", title=title, scale=alt.Scale(zero=zero))
    chart = (base.mark_rule(size=2).encode(x=axis("lower"), x2="upper:Q") +
             base.mark_bar(size=20).encode(x=axis("Q1"), x2="Q3:Q") +
             base.mark_tick(thickness=3, size=20).encode(x=axis("P50"), color=alt.value("#20252a")))
    if zero:
        line = alt.Chart(pd.DataFrame({"origin":[0]})).mark_rule(strokeDash=[4, 3], color="#555").encode(x="origin:Q")
        chart = line + chart
    return chart.properties(height=max(115, 44*len(dist)))


def show_results(summary, dist, totals, multi, mode):
    if multi:
        st.subheader("名次分布")
        chart = alt.Chart(dist).mark_bar().encode(
            x=alt.X("名次:O"), xOffset="策略:N", y=alt.Y("概率:Q", axis=alt.Axis(format="%")),
            color=alt.Color("策略:N"), tooltip=["策略", "名次", alt.Tooltip("概率:Q", format=".1%")])
        left, right = st.columns([2, 1])
        left.altair_chart(chart.properties(height=230), width="stretch")
        baseline = "赛前单车最优策略" if mode == "赛前" else "实际策略"
        right.markdown(f"**相对{baseline}的期望名次变化**")
        right.caption("正数表示名次提升，负数表示退后。")
        right.dataframe(summary[["strategy", "gain"]].rename(columns={"strategy":"策略", "gain":"提升（位）"}).round(2),
                        hide_index=True)
        columns = {"strategy":"策略", "exp_pos":"期望名次", "pos_p10":"名次 P10", "pos_p90":"名次 P90",
                   "mean_time":"平均用时（秒）", "d_pos_vs_best":"相对最优名次差"}
    else:
        names = summary.strategy.tolist()
        st.subheader("策略配对差")
        baseline = st.selectbox("基准策略", names)
        paired = paired_summary(totals, baseline).set_index("策略")
        others = [name for name in names if name != baseline]
        if others:
            name = others[0]
            row = paired.loc[name]
            st.info(conclusion(name, baseline, row["配对均差"], row["P2.5"], row["P97.5"]))
            st.altair_chart(box_chart(paired.loc[others].reset_index(), "P2.5", "P97.5", "比基准慢多少秒", True),
                            width="stretch")
            st.caption("虚线为基准（0 秒）；横线为中央 95% 的模拟配对差，箱体为中间 50%，箱内刻线为中位数。")
        else:
            st.info("当前只有基准策略；在表格中添加策略即可比较。")
        with st.expander("总用时分布"):
            st.altair_chart(box_chart(dist, "P5", "P95", "总用时（秒）"), width="stretch")
            st.caption("横线为 P5–P95，箱体为中间 50%，箱内刻线为中位数。")
        for key, col in (("delta_mean", "配对均差"), ("delta_lo", "P2.5"), ("delta_hi", "P97.5")):
            summary[key] = summary.strategy.map(paired[col])
        columns = {"strategy":"策略","mean":"平均秒数","p05":"P5","p50":"P50","p95":"P95","cvar95":"CVaR95",
                   "delta_mean":"配对均差","delta_lo":"差值 P2.5","delta_hi":"差值 P97.5","p_fastest":"最快占比"}
    display = summary[list(columns)].rename(columns=columns).round(2)
    if not multi:
        display["最快占比"] = summary.p_fastest.map("{:.1%}".format)
    st.dataframe(display, hide_index=True, width="stretch")


def main():
    st.html('''<style>
    [data-testid="stMainBlockContainer"] {max-width: 76rem; padding-top: 2.7rem;}
    h1 {border-left: 6px solid #e53935; padding-left: .65rem !important;}
    [data-testid="stSidebar"] {border-right: 1px solid #85899538;}
    </style>''')
    st.title("F1 策略模拟器")
    with st.sidebar:
        st.header("比赛设置")
        mode = st.radio("模式", ["事后", "赛前"], horizontal=True)
        year = st.selectbox("赛季", [2023, 2024, 2025, 2026], index=1)
        if mode == "赛前":
            races = prerace.calendar(ROOT).query("year == @year").sort_values("round").copy()
            races["race_id"] = [f"{year}-{rnd}" for rnd in races["round"]]
            races["event"] = races.circuit_id
        elif ML.is_file():
            ml = model_laps()
            races = ml.loc[ml.year == year, ["race_id", "round", "event"]].drop_duplicates().sort_values("round")
        else:
            races = pd.DataFrame()
        if races.empty:
            st.info("本地缺少该赛季数据。请按 scripts/download_data.md 构建输入。")
            return
        ids = races.race_id.tolist()
        labels = {r.race_id: f"R{r.round:02d} · {r.event}" for r in races.itertuples()}
        rid = st.selectbox("分站", ids, index=ids.index("2024-10") if "2024-10" in ids else 0, format_func=labels.get)
        n = st.slider("模拟次数", 500, 5000, 1000, 100)
        seed = st.number_input("随机种子", 0, 2**32-1, BASE_SEED)
        scenario = st.selectbox("情景", SCENARIOS)
        multi = st.toggle("多车模式")
        names = [c.name for c in field(rid, mode)] if multi else []
        driver = st.selectbox("本车车手", names) if names else None
    race, _, _ = race_inputs(rid, scenario, mode)
    st.subheader(labels[rid])
    st.caption(f"{year} 赛季　/　{race.total_laps} 圈　/　基础圈时 {race.base_lap_s:.3f} 秒")
    table, notes = defaults(rid, mode)
    for note in notes:
        st.info(note)
    st.markdown("#### 进站策略")
    st.caption("H 硬胎 · M 中性胎 · S 软胎。配方用连字符连接，进站圈用逗号分隔。可在表中增删策略。")
    edited = st.data_editor(table, disabled=["常见顺序"], column_order=["策略名称", "常见顺序", "配方顺序", "进站圈"],
                            column_config={"配方顺序": "执行顺序", "常见顺序": "默认组合写法"}, num_rows="dynamic", hide_index=True, width="stretch", key=f"editor_{mode}_{rid}")
    st.caption("单车模型的轮胎项在各配方段长不变时与顺序无关。默认组合按常见写法显示，模拟按执行顺序和进站圈计算；安全车和多车交通可使换序后的总用时不同。")
    _, errors = edited_strategies(edited, race.total_laps)
    if multi and not names:
        errors.append("本站缺少可装配的车手。")
    for message in errors:
        st.error(message)
    records = tuple(edited[COLUMNS].itertuples(index=False, name=None))
    request = (rid, records, n, seed, driver, scenario, mode)
    if st.button("运行模拟", type="primary", disabled=bool(errors)):
        st.session_state["submitted"] = request
    if not errors and st.session_state.get("submitted") == request:
        with st.spinner("正在评估策略…"):
            show_results(*assessment(*request), multi, mode)
    st.divider()
    with st.expander("数据来源与模型局限"):
        if mode == "赛前":
            st.write("轮胎后验取自上一赛季，圈速与安全车输入按本站日期截断。对手采用赛前单车最优策略。超车、起步与重启常数使用全样本校准。")
        else:
            st.write("轮胎参数来自各赛季 v3.1 后验。默认优化使用固定种子、200 个后验抽样、1000 次模拟及 2 圈网格。前十主流采用并列众数中名次最高的合规车手的真实进站圈。多车模式使用本站事后配速、实际对手策略及可装配车手；默认优化使用全赛道基线，评估按侧栏情景生成 SC/VSC。")
        source = (ROOT / "src/f1a/sim/README.md").read_text()
        st.markdown("**单车模型 README · 已知局限（原文）**")
        st.markdown(source.split("## 已知局限（v0.1）", 1)[1].split("\n## ", 1)[0].strip())


try:
    main()
except prerace.Unavailable as exc:
    reasons = {"no_prior_season": "缺少上一赛季的轮胎后验", "no_base_lap": "上一赛季在该赛道缺少基础圈时",
               "no_scheduled_laps": "缺少计划圈数", "unresolved_starting_grid": "发车位无法唯一确定",
               "missing_starting_grid": "缺少发车名单", "insufficient_field": "可用车手少于八名"}
    st.warning(f"本站赛前模拟不可用 · {reasons.get(str(exc), '缺少符合赛前规则的输入')}（{exc}）")
except FileNotFoundError as exc:
    name = Path(exc.filename).name if exc.filename else "数据文件"
    st.info(f"本地缺少 {name}。请按 scripts/download_data.md 构建输入。")
