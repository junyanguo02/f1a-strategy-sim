# T048　真实赛道几何与参考车速曲线（OpenF1 2024）

由 `analysis/track_geometry/run_track_geometry.py` 离线生成。联网仅使用 `https://api.openf1.org/v1/location` 与 `/v1/car_data`；原始响应缓存在 `data/raw/openf1_track/`，已缓存不重复下载；每次请求间隔 >=1.05 s。

## 方法

- 每场从 `laps_pit_filled.parquet` 选绿旗、非进站圈/出站圈、非第 1 圈的 5 个最快圈（不同车手优先；若数据不足则顺延候选）。
- 用缓存 laps 的 `date_start` 与 `lap_duration` 截取 OpenF1 `/location` 和 `/car_data`。OpenF1 的 `x,y` 是水平坐标、单位为 **1/10 m**；`z` 是海拔，本单不使用。
- 对每圈路径按自身弧长重采样，再按平均圈长归一化对齐；每 5 m 一个中心线点。中心线坐标做 9 点（45 m）周期移动平均；平滑会缩短路径，所以把平滑后的中心线**均匀缩放到 5 m 网格总长**（N×5 m），官方长度只用于独立校验，不参与缩放。
- 曲率用周期中心差分：`k=(x'y''-y'x'')/(x'^2+y'^2)^(3/2)`，单位 1/m，符号表示左/右；再对曲率做 3 点（15 m）平滑。最小半径 = 1/max|k|。
- 参考车速：把每圈 car_data 的时间按 location 的累计弧长映射到本圈距离，再插值到公共中心线 s 网格；给出 5 圈均值/min/max、油门均值、刹车比例（brake>0 的圈占比）。
- 图为左：中心线颜色=带符号曲率；右：曲率–距离。

## 各赛道校验

| race_id | 赛道 | 官方长度 m | 中心线长度 m | 长度误差 % | 最小半径 m | 参考圈时 s | 圈数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024-1 | Bahrain International Circuit | 5412 | 5335 | -1.42 | 19.8 | 93.947 | 57 |
| 2024-3 | Albert Park Circuit | 5278 | 5240 | -0.72 | 27.8 | 80.048 | 58 |
| 2024-10 | Circuit de Barcelona-Catalunya | 4657 | 4610 | -1.01 | 30.8 | 77.672 | 66 |
| 2024-13 | Hungaroring | 4381 | 4325 | -1.28 | 27.2 | 80.862 | 70 |
| 2024-16 | Autodromo Nazionale Monza | 5793 | 5760 | -0.57 | 30.2 | 81.734 | 53 |

### 所用 5 圈

- **2024-1**：VER #1 lap 39 (92.608s)；LEC #16 lap 36 (94.090s)；ALO #14 lap 48 (94.199s)；PER #11 lap 40 (94.364s)；NOR #4 lap 35 (94.476s)
- **2024-3**：LEC #16 lap 56 (79.813s)；NOR #4 lap 49 (79.915s)；SAI #55 lap 48 (80.031s)；PIA #81 lap 54 (80.199s)；RUS #63 lap 53 (80.284s)
- **2024-10**：NOR #4 lap 51 (77.115s)；VER #1 lap 54 (77.776s)；PER #11 lap 51 (77.785s)；HAM #44 lap 46 (77.809s)；PIA #81 lap 56 (77.874s)
- **2024-13**：RUS #63 lap 55 (80.305s)；SAR #2 lap 65 (80.561s)；VER #1 lap 51 (80.908s)；PER #11 lap 49 (81.096s)；SAI #55 lap 53 (81.441s)
- **2024-16**：NOR #4 lap 53 (81.432s)；HAM #44 lap 53 (81.512s)；VER #1 lap 43 (81.745s)；PIA #81 lap 53 (81.943s)；RUS #63 lap 49 (82.036s)

## 曲率平滑敏感性

下表是中心线在平滑窗口（点数/米）下的**未缩放**闭合长度与最小半径；主结果用 9 点，并把中心线缩放到 N×5 m（所以主结果总长不随窗口改变，最小半径会变）。

| race_id | w=3 (15m) L/rmin | w=5 (25m) L/rmin | w=9 (45m) L/rmin | w=15 (75m) L/rmin |
|---|---:|---:|---:|---:|
| 2024-1 | 5326 m / 18.5 m | 5319 m / 18.8 m | 5297 m / 19.7 m | 5247 m / 20.8 m |
| 2024-3 | 5224 m / 25.8 m | 5220 m / 26.3 m | 5205 m / 27.6 m | 5173 m / 30.0 m |
| 2024-10 | 4593 m / 29.1 m | 4588 m / 30.3 m | 4570 m / 30.5 m | 4527 m / 30.5 m |
| 2024-13 | 4296 m / 26.5 m | 4290 m / 26.7 m | 4269 m / 26.9 m | 4220 m / 27.2 m |
| 2024-16 | 5749 m / 23.2 m | 5747 m / 24.3 m | 5738 m / 30.0 m | 5720 m / 42.9 m |

## 缓存与请求

- 缓存：`data/raw/openf1_track/` 共 **80** 个 JSON、**3,925,269 字节**。
- 首次填充缓存共 40 次请求（location 20 + car_data 20），约 1.80 MB 原始响应；本单运行只读缓存，请求数 0。每请求间隔 >=1.05 s，失败重试最多 3 次。
- 缓存文件名含 `session_key/driver/lap`，可复现；没有下载其它域名。

## 图

- [2024-1 Bahrain International Circuit](outputs/figures/2024-1.png)
- [2024-3 Albert Park Circuit](outputs/figures/2024-3.png)
- [2024-10 Circuit de Barcelona-Catalunya](outputs/figures/2024-10.png)
- [2024-13 Hungaroring](outputs/figures/2024-13.png)
- [2024-16 Autodromo Nazionale Monza](outputs/figures/2024-16.png)

## 这些数据不能说明什么

1. **中心线不是赛车线**：5 圈位置平均会抹掉走线差异、路肩使用和超车；不能用于判断最优走线。
2. **OpenF1 location 是 GPS/计时的低采样数据**（约 3–5 Hz），经过 5 m 重采样和平滑；曲率与最小半径对平滑窗口敏感，不是赛道几何设计值。
3. **没有用高程**：z 轴海拔未用于曲率/速度；坡度、路拱、banking 未建模。
4. **只有 5 个参考圈**：不是全场/全赛季分布；车速 min/max 只描述这 5 圈，油门/刹车比例也不代表所有车手或所有轮胎状态。
5. **赛道长度用了官方公开值做独立检验**：中心线长度由 OpenF1 路径与 5 m 网格决定，不是官方测量；误差 <=1.5% 只说明量级一致。
6. **只覆盖 2024 年 5 条赛道**，不能外推到其它赛道/年份；赛道可能改版。
7. **本单只做几何与参考速度曲线**，不评价策略、不修改 `src/f1a/sim`。

## 复现

```sh
.venv/bin/python analysis/track_geometry/run_track_geometry.py
.venv/bin/python analysis/track_geometry/verify_track_geometry.py
```
