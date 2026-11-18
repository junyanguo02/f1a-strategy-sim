# 从公开数据重建

从仓库根目录执行。使用 Python 3.12；模型拟合需要 model 依赖组。OpenF1 历史数据用于非商业研究，商业使用须先确认授权。Jolpica 提供赛历、发车位和赛果。MIT 许可适用于本仓库代码，数据权利归各提供方。参见 [OpenF1](https://openf1.org/) 与 [Jolpica F1](https://github.com/jolpica/jolpica-f1)。

## 下载、对账与生成中间表

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[app,model]'
mkdir -p notes work
python scripts/build_cache.py --years 2023 2024 2025 2026 --source openf1
python scripts/reconcile.py
python scripts/race_quality.py
python scripts/build_pitloss_fcy.py
python analysis/traffic_features/run_traffic_features.py
python analysis/fcy_inputs/run_fcy_inputs.py
python analysis/sc_pit_timing/run_sc_pit_timing.py
python analysis/actual_strategies/run_actual_strategies.py
```

先检查缓存构建日志中的缺场与请求失败，再继续拟合。下载脚本会缓存成功响应，逐站对账保留缺失与差异。仓库中的 scheduled_laps.csv 提供赛前计划圈数。

## 拟合、验证与本机运行

```sh
for y in 2023 2024 2025 2026; do
  CHAIN_METHOD=parallel XLA_FLAGS=--xla_force_host_platform_device_count=4 python models/tyre_v3/fit_tyre_v3_1.py \
    --data data/processed/model_laps.parquet --traffic analysis/traffic_features/traffic_features.parquet \
    --years "$y" --out "models/tyre_v3/outputs31_$y" --chains 4 --warmup 1000 --samples 1000 || exit 1
done
python -m pytest
python scripts/sim_demo.py --race 2024-10 --season 2024
for y in 2024 2025 2026; do
  python scripts/backtest.py --season "$y" --prerace --out "work/single_$y.csv"
  python scripts/backtest_field.py --season "$y" --prerace --sensitivity --out "work/field_$y.csv"
done
streamlit run app/strategy_app.py --server.address 127.0.0.1
```

事后回测使用相同脚本，省略 `--prerace` 与 `--sensitivity` 并指定另一个输出文件。首次测试也可仅安装 `.[app]` 并运行 pytest；真实分站集成测试会报告缺少的本地输入。

## Reproduce the seven analyses and verify their outputs

Run the earlier data-build and per-season fit steps first. The actual-strategy builder also writes
`analysis/actual_strategies/strategies.csv` and `stints.csv`. Stint heterogeneity produces the
stint slopes, teammate outcomes builds the matched pairs, and extra-stop classification supplies
the pair metrics for the robustness analysis.

From the repository root, with the virtual environment activated, run:

```sh
(
  set -eu
  for name in tyre_cliff tyre_management stint_heterogeneity overtake_start teammate_outcomes extra_stop extra_stop_robustness; do
    (
      cd "analysis/$name"
      mkdir -p logs
      for run in 1 2; do
        python "run_$name.py"
        python - <<'PY' > "logs/run$run.sha256"
from hashlib import sha256
from pathlib import Path
files = [*Path("outputs").glob("*.csv"), *Path("figures").glob("*.png"), Path("report.md")]
for path in sorted(files):
    print(sha256(path.read_bytes()).hexdigest(), path.as_posix(), sep="  ")
PY
      done
      cmp logs/run1.sha256 logs/run2.sha256
      echo 'TWO RUNS IDENTICAL' > logs/two_runs.txt
      python "verify_$name.py"
    )
  done
)
```

Run `python -m pytest` from the repository root to include the seven analyses' synthetic tests.
The hash comparison checks reports, figures and CSVs produced with the same inputs and software environment.
