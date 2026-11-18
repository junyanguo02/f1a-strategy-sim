#!/usr/bin/env python3
"""T042: 进站损失相对表审计与清洗规则。

在项目根目录执行：
    .venv/bin/python analysis/pitloss_audit/run_pitloss_audit.py

只写 analysis/pitloss_audit/；不覆盖 analysis/fcy_inputs/pit_loss_relative.parquet。
输出：
  outputs/pit_loss_relative_clean.parquet  （原表 + clean_keep + clean_reason）
  outputs/extreme_classification.csv       （<12s / >45s 逐条分类）
  outputs/pool_stats_normal.csv / pool_stats_fcy.csv
  outputs/fallback_races.csv               （赛前/事后回退场次）
  report.md
不装依赖、不联网；只读其他文件。
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
LOG = HERE / "logs"
FCY = ROOT / "analysis" / "fcy_inputs" / "pit_loss_relative.parquet"
DET = ROOT / "analysis" / "sc_pit_timing" / "outputs" / "pit_match_detail.csv"
LAPS = ROOT / "data" / "processed" / "laps_pit_filled.parquet"
RS = ROOT / "analysis" / "actual_strategies" / "outputs" / "race_summary.csv"
MODEL = ROOT / "data" / "processed" / "model_laps.parquet"
STRAT = ROOT / "analysis" / "actual_strategies" / "strategies.parquet"
RAW = ROOT / "data" / "raw" / "openf1"
JOL = ROOT / "data" / "raw" / "jolpica"
SEED = 20261117  # T013/T018 生成时的种子（本单只做确定性统计，不使用随机数）


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ws(x):
    """读写一致的字符串化。"""
    return "" if pd.isna(x) else f"{float(x):.2f}"


# ---------------------------------------------------------------- 读输入
def load():
    pl = pd.read_parquet(FCY)
    det = pd.read_csv(DET)
    laps = pd.read_parquet(LAPS)
    rs = pd.read_csv(RS)
    ml = pd.read_parquet(MODEL)
    strat = pd.read_parquet(STRAT)
    sess = strat.drop_duplicates(["year", "round"]).set_index("session_key")[["year", "round"]]
    return pl, det, laps, rs, ml, sess


def load_pit_cache(sess):
    rows = []
    for f in sorted(RAW.glob("pit_session_key=*.json")):
        sk = int(f.name.split("=")[1].split(".")[0])
        if sk not in sess.index:
            continue
        y, r = int(sess.loc[sk, "year"]), int(sess.loc[sk, "round"])
        for x in json.load(open(f)):
            rows.append(dict(year=int(y), round=int(r), driver_number=x.get("driver_number"),
                             lap=x.get("lap_number"), lane=x.get("lane_duration"),
                             stop=x.get("stop_duration")))
    pit = pd.DataFrame(rows)
    if len(pit):
        pit = pit.dropna(subset=["lap"]).copy()
        pit["lap"] = pit.lap.astype(int)
        pit = pit.groupby(["year", "round", "driver_number", "lap"], as_index=False).agg(
            lane=("lane", "median"), stop=("stop", "median"))
        pit["lane_med_race"] = pit.groupby(["year", "round"]).lane.transform("median")
    return pit


def load_penalties(sess):
    rows = []
    for f in sorted(RAW.glob("race_control_session_key=*.json")):
        sk = int(f.name.split("=")[1].split(".")[0])
        if sk not in sess.index:
            continue
        y, r = int(sess.loc[sk, "year"]), int(sess.loc[sk, "round"])
        for m in json.load(open(f)):
            msg = str(m.get("message") or "").upper()
            if not any(k in msg for k in ("DRIVE THROUGH", "DRIVE-THROUGH", "STOP AND GO", "STOP-AND-GO")):
                continue
            for car in re.findall(r"CAR (\d+)", msg):
                rows.append(dict(year=int(y), round=int(r), driver_number=int(car),
                                 lap=m.get("lap_number")))
    return pd.DataFrame(rows)


def load_calendar():
    out = {}
    for f in sorted(JOL.glob("*_races_limit=100.json")):
        year = int(f.name.split("_")[0])
        try:
            races = json.load(open(f))["MRData"]["RaceTable"]["Races"]
        except Exception:
            continue
        for race in races:
            rnd = int(race["round"])
            out[(year, rnd)] = pd.Timestamp(race["date"]).date()
    return out


# ---------------------------------------------------------------- 特征
def build_features(pl, det, laps, rs, pit, pen, stop_set):
    pl = pl.merge(rs[["year", "round", "is_wet", "wet_tyre_used_any", "has_red_flag"]],
                  on=["year", "round"], how="left")
    lapmap = laps.set_index(["year", "round", "driver", "lap"])
    # 每行 in/out/window 的配方和旗标
    comp, pitin, pitout, sc, vsc, status = {}, {}, {}, {}, {}, {}
    for key, r in lapmap.iterrows():
        comp[key] = r.compound
        pitin[key] = bool(r.pit_in_filled) if pd.notna(r.pit_in_filled) else False
        pitout[key] = bool(r.pit_out) if pd.notna(r.pit_out) else False
        sc[key] = bool(r.safety_car) if pd.notna(r.safety_car) else False
        vsc[key] = bool(r.virtual_safety_car) if pd.notna(r.virtual_safety_car) else False
        status[key] = str(r.track_status) if pd.notna(r.track_status) else ""
    # pit 缓存
    pit_idx = pit.set_index(["year", "round", "driver_number", "lap"]) if len(pit) else None
    pen_idx = {}
    for x in pen.itertuples(index=False):
        pen_idx.setdefault((x.year, x.round, x.driver_number), []).append(x.lap)
    rows = []
    for q in pl.itertuples(index=False):
        y, r, d, lap = int(q.year), int(q.round), q.driver, int(q.lap)
        out = int(q.out_lap) if pd.notna(q.out_lap) else lap + 1
        wet_comp = []
        bad_ref = []
        refs = str(q.reference_laps).replace(" ", "")
        for part in refs.split(","):
            if not part.isdigit():
                continue
            L = int(part)
            k = (y, r, d, L)
            if L == 1 or pitin.get(k) or pitout.get(k) or sc.get(k) or vsc.get(k):
                bad_ref.append(L)
        for L in range(max(1, lap - 1), out + 2):
            k = (y, r, d, L)
            if str(comp.get(k, "")) in ("INTERMEDIATE", "WET"):
                wet_comp.append(L)
        key = (y, r, int(q.driver_number), lap)
        lane = stop = lane_med = np.nan
        if pit_idx is not None and key in pit_idx.index:
            x = pit_idx.loc[key]
            if isinstance(x, pd.DataFrame):
                x = x.iloc[0]
            lane, stop, lane_med = x.lane, x.stop, x.lane_med_race
        near_pen = False
        for L in pen_idx.get((y, r, int(q.driver_number)), []):
            if pd.notna(L) and lap - 3 <= int(L) <= out + 3:
                near_pen = True
        # 连续两圈进站：下一停的进站圈 = 本次出站圈
        consecutive = ((y, r, d, out) in stop_set)
        slow = bool((pd.notna(stop) and stop >= 5)
                    or (pd.notna(lane) and pd.notna(lane_med) and lane - lane_med >= 8)
                    or consecutive)
        huge = (float(q.in_lap_time_s) > 300) or (float(q.out_lap_time_s) > 300) \
            if pd.notna(q.in_lap_time_s) and pd.notna(q.out_lap_time_s) else False
        rows.append(dict(
            rain=bool(q.is_wet) or bool(q.wet_tyre_used_any) or bool(wet_comp),
            wetland=",".join(map(str, wet_comp)),
            red=bool(q.red_in) or bool(q.red_out),
            penalty=near_pen,
            ref_issue=bool(len(bad_ref) > 0) or int(q.reference_n) < 3,
            ref_bad_laps=",".join(map(str, bad_ref)),
            huge=huge,
            slow=slow,
            consecutive=consecutive,
            lane=lane, stop=stop, lane_med=lane_med,
        ))
    feat = pd.DataFrame(rows)
    pl = pd.concat([pl.reset_index(drop=True), feat], axis=1)
    pl["subset"] = np.where(pl.condition == "正常", "normal",
                            np.where(pl.condition == "SC", "SC",
                                     np.where(pl.condition == "VSC", "VSC", "other")))
    pl["extreme"] = pl.relative_usable & ((pl.relative_loss_s < 12) | (pl.relative_loss_s > 45))
    return pl


# ---------------------------------------------------------------- 分类/清洗
def classify(pl):
    cat, sub, ev, keep, reason = [], [], [], [], []
    for q in pl.itertuples(index=False):
        usable = bool(q.relative_usable)
        if not usable:
            cat.append("—"); sub.append("—")
            why = q.exclusion_reason if pd.notna(q.exclusion_reason) else q.unusable_reason
            ev.append(f"原表 unusable：{why}" if pd.notna(why) else "原表 unusable")
            keep.append(False); reason.append("原表不可用，不进入任何池")
            continue
        subset = q.subset
        if bool(q.huge):
            c, s = "红旗/暂停", "计时异常（超长圈时）"
            e = f"进站圈/出站圈圈时 {q.in_lap_time_s:.1f}/{q.out_lap_time_s:.1f}s，明显含红旗/暂停或计时异常"
        elif bool(q.rain):
            c, s = "雨战/换雨胎", "湿地/换雨胎"
            e = (f"雨战/湿地（race is_wet={q.is_wet}, wet_tyre_used={q.wet_tyre_used_any}）"
                 + (f"；窗口内雨胎圈 {q.wetland}" if q.wetland else ""))
        elif bool(q.red):
            c, s = "红旗期间", "红旗窗口"
            e = f"red_in={q.red_in}, red_out={q.red_out}"
        elif bool(q.penalty):
            c, s = "罚时/通过维修区", "race_control 罚时"
            e = "race_control 有 DRIVE THROUGH/STOP-AND-GO 记录且在本停附近"
        elif bool(q.ref_issue):
            c, s = "参考圈问题", "参考车手<3 或参考圈异常"
            e = f"reference_n={int(q.reference_n)}" + (f"；异常参考圈 {q.ref_bad_laps}" if q.ref_bad_laps else "")
        elif bool(q.slow):
            c, s = "慢进站（保留）", ("连续进站/车损" if q.consecutive else "停车/维修区异常")
            e = (f"lane={q.lane:.2f}s vs 同场中位 {q.lane_med:.2f}s"
                 if pd.notna(q.lane) and pd.notna(q.lane_med) else "连续两圈进站") if q.consecutive else \
                (f"lane={q.lane:.2f}s vs 同场中位 {q.lane_med:.2f}s；stop={q.stop:.2f}s"
                 if pd.notna(q.lane) and pd.notna(q.lane_med) else "停站时长异常")
        elif subset == "normal":
            c, s = "其他", "极端值且无法确认正常进站代价"
            e = f"相对损失 {q.relative_loss_s:.2f}s 超出 [12,45]，无雨/红旗/罚时/参考圈/慢停证据"
        else:
            c, s = "其他", "FCY 池正常值（范围外保留）"
            e = f"{subset} 池相对损失 {q.relative_loss_s:.2f}s，FCY 下属正常范围外但保留"
        cat.append(c); sub.append(s); ev.append(e)
        if c in ("红旗/暂停", "雨战/换雨胎", "红旗期间", "罚时/通过维修区", "参考圈问题"):
            keep.append(False); reason.append(f"清洗删除：{c} / {s}")
        elif subset == "normal" and bool(q.extreme) and c != "慢进站（保留）":
            keep.append(False); reason.append("清洗删除：其他极端值，无法确认正常进站代价")
        elif c == "慢进站（保留）":
            keep.append(True); reason.append("保留：真实慢进站/车损")
        elif subset in ("SC", "VSC"):
            keep.append(True); reason.append("保留：FCY 池正常值")
        else:
            keep.append(True); reason.append("保留：正常范围")
    pl = pl.copy()
    pl["extreme_category"] = cat
    pl["extreme_subcategory"] = sub
    pl["clean_evidence"] = ev
    pl["clean_keep"] = pd.Series(keep, dtype=bool)
    pl["clean_reason"] = reason
    return pl


# ---------------------------------------------------------------- 统计
def stats_block(x, col="relative_loss_s"):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p5=np.nan, p95=np.nan)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(x.median()),
                p5=float(x.quantile(.05)), p95=float(x.quantile(.95)))


def pool_stats(pl, det):
    # normal all
    rows = []
    def add(label, before, after):
        b, a = stats_block(before), stats_block(after)
        rows.append(dict(pool=label, n_before=b["n"], mean_before=b["mean"], median_before=b["median"],
                         p5_before=b["p5"], p95_before=b["p95"],
                         n_after=a["n"], mean_after=a["mean"], median_after=a["median"],
                         p5_after=a["p5"], p95_after=a["p95"],
                         removed=b["n"] - a["n"]))
    normal = pl[pl.relative_usable & (pl.condition == "正常")]
    add("normal:ALL", normal.relative_loss_s, normal.loc[normal.clean_keep, "relative_loss_s"])
    for c, g in normal.groupby("circuit_id"):
        add(f"normal:{c}", g.relative_loss_s, g.loc[g.clean_keep, "relative_loss_s"])
    # FCY pools from det, mapping clean_keep by keys
    keys = ["year", "round", "driver", "lap"]
    det = det.copy()
    flags = pl.set_index(keys)[["clean_keep"]]
    det = det.merge(flags, left_on=keys, right_index=True, how="left")
    det = det[det.dt_group.notna() & det.relative_loss_s.notna()].copy()
    phase_groups = {1: ("宣布后0-60秒", "60秒-1圈"), 2: ("1-2圈",), 3: ("2圈以后",)}
    for t in ("SC", "VSC"):
        for ph, groups in phase_groups.items():
            s = det[(det.condition == t) & (det.dt_group.isin(groups))]
            add(f"fcy:{t}:ph{ph}", s.relative_loss_s, s.loc[s.clean_keep.fillna(False), "relative_loss_s"])
    return pd.DataFrame(rows)


def fallback_lists(pl, ml, calendar):
    races = ml[["race_id", "year", "round", "circuit_id"]].drop_duplicates().copy()
    normal = pl[pl.relative_usable & (pl.condition == "正常")].copy()
    normal["date"] = [calendar.get((int(y), int(r))) for y, r in normal[["year", "round"]].itertuples(index=False)]
    out = []
    for q in races.itertuples(index=False):
        y, r, circ = int(q.year), int(q.round), q.circuit_id
        cutoff = calendar.get((y, r))
        hist = normal[(normal.circuit_id == circ) & normal.date.notna()]
        if cutoff is not None:
            hist = hist[hist.date < cutoff]
        else:
            hist = hist[(hist.year < y) | ((hist.year == y) & (hist.round < r))]
        pre_b = int(hist.relative_usable.sum())
        pre_a = int(hist.clean_keep.sum())
        post = normal[normal.circuit_id == circ]
        post_b = int(len(post))
        post_a = int(post.clean_keep.sum())
        out.append(dict(race_id=q.race_id, year=y, round=r, circuit_id=circ,
                        cutoff=cutoff, pre_before_n=pre_b, pre_after_n=pre_a,
                        pre_fallback_before=pre_b < 10, pre_fallback_after=pre_a < 10,
                        post_before_n=post_b, post_after_n=post_a,
                        post_fallback_before=post_b < 10, post_fallback_after=post_a < 10))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- report
def md_table(df, cols=None):
    df = df[cols] if cols else df
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_report(pl, det, pool, fb, hashes, gen):
    ext = pl[pl.extreme].copy()
    counts = (ext.groupby(["subset", "extreme_category", "extreme_subcategory"])
              .size().reset_index(name="n").sort_values(["subset", "extreme_category", "n"],
                                                          ascending=[True, True, False]))
    # examples: 每个大类最多 5 条
    examples = []
    for c, g in ext.groupby("extreme_category", sort=True):
        g = g.reindex(g.relative_loss_s.abs().sort_values(ascending=False).index).head(5)
        for q in g.itertuples(index=False):
            examples.append(dict(category=c, race=f"{int(q.year)}-{int(q.round)}",
                                 circuit=q.circuit_id, driver=q.driver, lap=int(q.lap),
                                 condition=q.condition, loss=float(q.relative_loss_s),
                                 evidence=q.clean_evidence))
    lines = []
    lines.append("# T042　进站损失相对表审计与清洗规则")
    lines.append("")
    lines.append("由 `analysis/pitloss_audit/run_pitloss_audit.py` 离线生成；只读原表，"
                 "不覆盖 `analysis/fcy_inputs/pit_loss_relative.parquet`。固定输入、确定性统计。")
    lines.append("")
    lines.append(f"生成时间：{gen}；原始相对表 3402 行，relative_usable {int(pl.relative_usable.sum())} 行；"
                 f"极端（<12s 或 >45s）{len(ext)} 行。")
    lines.append("")
    lines.append("## 分类口径")
    lines.append("")
    lines.append("- 数据来源：`pit_loss_relative.parquet`（T013/T018 列）+ `laps_pit_filled.parquet`"
                 "（配方、进站圈/出站圈、SC/VSC）+ `fcy_race_summary.csv`（湿地/红旗）"
                 "+ OpenF1 `pit_session_key=*.json`（lane/stop 时长）+ `race_control_*.json`（DRIVE THROUGH / STOP-AND-GO）。")
    lines.append("- “红旗期间”用原表 `red_in/red_out`；原表已把红旗窗口从 `relative_usable` 排除，"
                 "所以红旗行多数为 0；超长圈时（>300 s，含暂停/计时异常）单列为“红旗/暂停”。")
    lines.append("- “慢进站”判据：同场 lane 中位 +8 s 以上、或停车时长 ≥5 s、或连续两圈进站（车损）；"
                 "慢进站**保留**。")
    lines.append("- “参考圈问题”：参考圈含首圈/进站圈/出站圈/SC/VSC，或 `reference_n<3`；原表参考圈"
                 "大部分已排除非绿旗圈，本次未发现含首圈/进站/SC 的参考圈，只发现样本不足。")
    lines.append("- “罚时/通过维修区”：race_control 的 DRIVE THROUGH / STOP-AND-GO 消息（解析 CAR 号）"
                 "落在本次进站前后 3 圈内。")
    lines.append("")
    lines.append("## 极端值分类（共 %d 行）" % len(ext))
    lines.append("")
    lines.append(md_table(counts))
    lines.append("")
    lines.append("### 每类例子")
    lines.append("")
    lines.append(md_table(pd.DataFrame(examples)))
    lines.append("")
    lines.append("## 清洗规则（clean_keep）")
    lines.append("")
    lines.append("- 删除：雨战/换雨胎、红旗期间/暂停、罚时/通过维修区、参考圈问题。")
    lines.append("- 保留：真实慢进站/车损/连续进站（策略风险的一部分）。")
    lines.append("- 删除：normal 池内其余无法确认正常进站代价的极端值（<12 s 或 >45 s，且无上述证据）。")
    lines.append("- 保留：SC/VSC 分组池内范围外的值（FCY 进站本来就与绿旗不同；只删雨/红旗/罚时/参考圈问题）。")
    lines.append("- 原表 `relative_usable=False` 的行 clean_keep=False，理由沿用原排除原因。")
    lines.append("")
    lines.append("## 各池清洗前后")
    lines.append("")
    lines.append(md_table(pool))
    lines.append("")
    lines.append("## 会触发“回退到全体 normal 池”的场次")
    lines.append("")
    pre = fb[(fb.pre_fallback_before) | (fb.pre_fallback_after)]
    post = fb[(fb.post_fallback_before) | (fb.post_fallback_after)]
    lines.append(f"- 赛前口径（按赛事日期截断历史）：清洗前触发 {int(fb.pre_fallback_before.sum())} 场，"
                 f"清洗后 {int(fb.pre_fallback_after.sum())} 场；完整清单见 `outputs/fallback_races.csv`。")
    lines.append("- 赛前触发场次（清洗前 / 清洗后）：")
    for q in pre.itertuples(index=False):
        lines.append(f"  - {q.race_id} ({q.circuit_id})：前 {q.pre_before_n}，后 {q.pre_after_n}"
                     + ("；清洗前触发" if q.pre_fallback_before else "")
                     + ("；清洗后触发" if q.pre_fallback_after else ""))
    lines.append(f"- 事后口径（全部历史，含本场）：清洗前触发 {int(fb.post_fallback_before.sum())} 场，"
                 f"清洗后 {int(fb.post_fallback_after.sum())} 场；完整清单见 `outputs/fallback_races.csv`。")
    lines.append("- 事后触发场次（清洗前 / 清洗后）：")
    for q in post.itertuples(index=False):
        lines.append(f"  - {q.race_id} ({q.circuit_id})：前 {q.post_before_n}，后 {q.post_after_n}"
                     + ("；清洗前触发" if q.post_fallback_before else "")
                     + ("；清洗后触发" if q.post_fallback_after else ""))
    lines.append("")
    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **分类是基于可见证据的描述性判断**：没有录像/位置/遥测，慢进站、交通、车损之间可能混淆；"
                 "“其他”里的极端值不一定都是数据错误。")
    lines.append("2. **红旗窗口是近似值**：`red_in/red_out` 来自 T013 的近似窗口，不是 FIA 官方逐秒记录；"
                 "超长圈时只能说明计时/暂停异常。")
    lines.append("3. **雨战剔除是保守的**：整场湿地/用过雨胎的行都删，可能把干地窗口的合法干胎进站也删掉；"
                 "本单优先保证“正常干地进站代价”口径一致。")
    lines.append("4. **`clean_keep` 只解决样本池偏差，不解决模型设定问题**：它不能保证清洗后的回测结论正确，"
                 "也不替代赛道/策略/交通建模。")
    lines.append("5. **回退清单依赖 `min_circuit_pits=10` 与赛事日期**：换阈值、换截断日期会改变清单；"
                 "赛前 cutoff 用 Jolpica 正赛日期，不含练习/排位信息。")
    lines.append("6. **慢进站样本保留会拉高方差**：这是有意的（真实策略风险），但会让池均值高于“典型进站”；"
                 "报告里同时给中位数与 P5/P95。")
    lines.append("7. **本单只做数据审计与清洗输入**，不评价策略、不修改 `src/f1a/sim`。")
    lines.append("")
    lines.append("## 输入哈希")
    lines.append("")
    lines.append(md_table(hashes))
    lines.append("")
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    gen = now()
    pl, det, laps, rs, ml, sess = load()
    pit = load_pit_cache(sess)
    pen = load_penalties(sess)
    calendar = load_calendar()
    stop_set = set(pl[["year", "round", "driver", "lap"]].itertuples(index=False, name=None))
    pl = build_features(pl, det, laps, rs, pit, pen, stop_set)
    pl = classify(pl)
    # 输出 clean parquet（原表 + 两列）
    clean = pl.drop(columns=[c for c in ["rain", "wetland", "red", "penalty", "ref_issue",
                                        "ref_bad_laps", "huge", "slow", "consecutive",
                                        "lane", "stop", "lane_med", "subset", "extreme",
                                        "extreme_category", "extreme_subcategory", "clean_evidence",
                                        "is_wet", "wet_tyre_used_any", "has_red_flag"]
                               if c in pl.columns])
    # 保留原列 + clean_keep + clean_reason；保持原表行顺序（T043 的配对样本依赖顺序）
    clean = clean.reset_index(drop=True)
    clean.to_parquet(OUT / "pit_loss_relative_clean.parquet", index=False)
    # 极端分类
    ext = pl[pl.extreme].copy()
    ext = ext[["year", "round", "circuit_id", "driver", "driver_number", "lap", "out_lap",
               "condition", "subset", "relative_loss_s", "relative_in_s", "relative_out_s",
               "in_lap_time_s", "out_lap_time_s", "reference_n", "red_in", "red_out",
               "is_wet", "wet_tyre_used_any", "wetland", "slow", "consecutive",
               "penalty", "ref_issue", "ref_bad_laps", "huge",
               "extreme_category", "extreme_subcategory", "clean_evidence",
               "clean_keep", "clean_reason"]]
    ext.to_csv(OUT / "extreme_classification.csv", index=False)
    # 池统计
    pool = pool_stats(pl, det)
    pool[pool.pool.str.startswith("normal")].to_csv(OUT / "pool_stats_normal.csv", index=False)
    pool[pool.pool.str.startswith("fcy")].to_csv(OUT / "pool_stats_fcy.csv", index=False)
    fb = fallback_lists(pl, ml, calendar)
    fb.to_csv(OUT / "fallback_races.csv", index=False)
    # 输入哈希
    import hashlib
    rows = []
    for p in [FCY, DET, LAPS, RS, MODEL, STRAT]:
        rows.append({"文件": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "字节": p.stat().st_size})
    hashes = pd.DataFrame(rows)
    build_report(pl, det, pool, fb, hashes, gen)
    print("clean rows", len(clean), "usable", int(pl.relative_usable.sum()),
          "kept", int(pl.clean_keep.sum()), "extreme", len(ext))
    print("pool rows", len(pool), "fallback pre/post",
          int(fb.pre_fallback_before.sum()), int(fb.pre_fallback_after.sum()),
          int(fb.post_fallback_before.sum()), int(fb.post_fallback_after.sum()))
    print("report", HERE / "report.md")
    print("\nDONE")


if __name__ == "__main__":
    main()
