"""Generate the T005 Markdown report from parquet and reconciliation CSVs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import pandas as pd

from f1a.clean import REASONS, clean_laps
from scripts.reconcile import PROCESSED, YEARS


def table(headers, rows):
    def render(values):
        return "| " + " | ".join(str(v).replace("|", r"\|") for v in values) + " |"
    return "\n".join([render(headers), render(["---"] * len(headers)),
                      *(render(row) for row in rows)])


def summarize(laps):
    total = len(laps)
    clean = int(laps["is_clean"].sum())
    wet = int(laps["compound"].astype("string").str.strip().str.upper()
              .isin(["INTERMEDIATE", "WET"]).sum())
    unknown_sc = int((laps["safety_car"].isna() | laps["virtual_safety_car"].isna()).sum())
    return (total, clean, f"{clean/total:.2%}", wet, f"{wet/total:.2%}",
            unknown_sc, int(laps["pit_in"].isna().sum()))


def reason_counts(laps):
    return tuple(int(laps["drop_reason"].str.split("|", regex=False).map(
        lambda values: reason in values).sum()) for reason in REASONS)


def main():
    coverage = pd.read_csv(PROCESSED / "reconcile_coverage.csv")
    comparison = pd.read_csv(PROCESSED / "reconcile.csv", dtype={
        name: "Int64" for name in ("laps_openf1", "laps_jolpica", "diff")
    })
    round_rows, season_rows, reason_rows, seasons = [], [], [], []
    for year in YEARS:
        path = PROCESSED / f"laps_{year}.parquet"
        if not path.exists():
            continue
        season = clean_laps(pd.read_parquet(path))
        seasons.append(season)
        for number, race in season.groupby("round"):
            values = summarize(race)
            round_rows.append((year, number, *values, "是" if values[3] * 10 > values[0] else "否"))
            reason_rows.append((year, number, *reason_counts(race)))
        season_rows.append((year, season["round"].nunique(), *summarize(season)))
        reason_rows.append((year, "合计", *reason_counts(season)))
    all_laps = pd.concat(seasons, ignore_index=True)
    season_rows.append(("全部", len(round_rows), *summarize(all_laps)))
    recorded = {(row[0], row[1]) for row in round_rows}
    for race in coverage.itertuples(index=False):
        if (race.year, race.round) not in recorded:
            round_rows.append((race.year, race.round, 0, 0, "—", 0, "—", 0, 0, "—"))
            reason_rows.append((race.year, race.round, *([0] * len(REASONS))))
    round_rows.sort(key=lambda row: (row[0], row[1]))
    reason_rows.sort(key=lambda row: (
        float("inf") if row[0] == "全部" else row[0],
        float("inf") if row[1] == "合计" else row[1],
    ))
    reason_rows.append(("全部", "合计", *reason_counts(all_laps)))
    summary = ["总圈数", "干净圈", "干净占比", "雨胎圈", "雨胎占比", "SC/VSC 未知", "pit_in 未知"]
    comparable = comparison["diff"].notna()
    different = comparison.loc[comparable & comparison["diff"].ne(0)]
    unavailable = comparison.loc[~comparable]
    print("""# T005 数据质量

由 `python scripts/data_quality.py` 从赛季 parquet 和对账 CSV 生成。干净圈规则参考 Heilmeier 等（2020，p.4）的前两圈、进站和 FCY 排除，其他规则依据 T005。每圈保留所有触发原因，以竖线分隔，所以原因计数可重叠。慢圈阈值按赛季、分站及车号，对通过前置条件的圈一次性计算中位数，严格大于 1.07 倍才标记。

两类雨胎除以该站全部圈，严格超过 10% 才标记雨战。未缓存分站的计数为零，比例及雨战结果以破折号表示。SC/VSC 任一值缺失即计入未知。进站缺失单独计数，T005 的进出站规则只排除已标真值，因此有未知进站的赛站可能高估干净圈数。干净圈标记由函数按需生成，原始圈表保留。

以分站和车号连接两来源，OpenF1 按记录行计数。已缓存的分站如有赛果车号未出现在 OpenF1，OpenF1 计 0。整站缺源、Jolpica 无该车记录或圈数缺失则用空值表示不可比较。差值为 OpenF1 减 Jolpica。差异原样保留。

## 赛季汇总
""")
    print(table(["赛季", "已缓存分站", *summary], season_rows))
    print("\n## 逐站质量\n")
    print(table(["赛季", "分站", *summary, "雨战"], round_rows))
    print("\n## 逐站及赛季排除原因\n")
    print(table(["赛季", "分站", *REASONS], reason_rows))
    print("\n## 未完成双方对账的分站\n")
    print(table(coverage.columns, coverage.loc[coverage["status"].ne("compared")].itertuples(index=False, name=None)))
    print(f"\n## 圈数不一致清单（{len(different)} 位次）\n")
    print(table(comparison.columns, different.itertuples(index=False, name=None)))
    print(f"\n## 圈数不可比较清单（{len(unavailable)} 位次）\n")
    print(table(comparison.columns, unavailable.fillna(pd.NA).astype("string").fillna("—")
                .itertuples(index=False, name=None)))


if __name__ == "__main__":
    main()
