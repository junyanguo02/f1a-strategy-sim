"""Render the T050 report and five figures from the saved calibration outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {"shared":"共用","track_mu":"逐站 μ"}
NAMES = {"2024-1":"巴林","2024-3":"澳大利亚","2024-10":"西班牙","2024-13":"匈牙利","2024-16":"蒙扎"}
ENG = ["Bahrain","Australia","Spain","Hungary","Monza"]


def table(header, rows):
    return "\n".join(["| "+" | ".join(row)+" |" for row in [header,["---"]*len(header),*rows]])


def make_report(out):
    fit=json.loads((out/"fit.json").read_text())
    m,f,p,r=[pd.read_csv(out/(name+".csv")) for name in ["metrics","fuel","profiles","regions"]]
    figdir=out/"figures"
    figdir.mkdir(exist_ok=True)
    for (rid,label),english in zip(NAMES.items(),ENG):
        fig,axes=plt.subplots(2,1,sharex=True,sharey=True,figsize=(11,5.8),layout="constrained")
        for ax,(mode,title),color in zip(axes,[("shared","Shared"),("track_mu","Circuit grip")],["#0072b2","#c45b19"]):
            a=p[(p.race==rid)&(p["mode"]==mode)]
            end=a.s.max()+5
            ax.fill_between(a.s,a.low,a.high,color="#bcc6cc",alpha=.65,label="Five-lap range")
            ax.plot(a.s,a.ref,color="#27343c",lw=1.3,label="Reference mean")
            ax.plot(a.s,a.sim,color=color,lw=1.3,label=title)
            for reg in r[(r.race==rid)&(r["mode"]==mode)].itertuples():
                spans=[(reg.start_m,reg.end_m)] if reg.end_m>reg.start_m else [(reg.start_m,end),(0,reg.end_m)]
                for left,right in spans:
                    ax.axvspan(left,right,color="#edb22e",alpha=.32)
                ax.text((reg.start_m+50)%end,.04,str(reg.rank),transform=ax.get_xaxis_transform(),ha="center",fontsize=9)
            ax.set_ylabel("Speed (km/h)")
            ax.set_xlim(0,end)
            ax.legend(loc="upper center",bbox_to_anchor=(.5,1.18),ncol=3,fontsize=8,frameon=False)
        axes[1].set_xlabel("Centreline distance (m)")
        fig.suptitle(f"2024 {english}  |  Speed profiles and three largest 100 m RMS errors")
        fig.savefig(figdir/(rid+".png"),dpi=130)
        plt.close(fig)
    report=["# 五站单圈标定与燃油效应（2024）",
            "逐站抓地模型的五站圈时均贴近参考圈时，但逐点车速仍有偏差。正式标定的两个模型都把有效轮上功率推到 450 kW 下界。以下为样本内描述和给定参数下的质量敏感性。",
            "## 方法",
            """使用 T048 五个分站、每站五个绿旗非进出站快圈。保留原始 5 m 网格、有符号曲率、相位和闭合末段。拟合对象为五圈车速均值，圈时目标为五个实测圈时的算术均值。每场质量固定为 798 kg 加估计平均余油：假定 107.5 kg 均匀耗尽，第 l 圈中点的余油为 107.5 × [1 − (l − 0.5)/正赛圈数] kg，再对五个样本圈取平均。其余固定量沿用 T049（空气密度 1.20 kg/m³，前轴静态与气动份额均为 0.46，独立限速 110 m/s）。""",
            """逐站目标 J = Σᵢ wᵢ[(模拟速度−参考速度)/10 km/h]² + [(模拟圈时−参考圈时)/2 s]²，各站等权相加。wᵢ 正比于 1/[10² + ((最高−最低参考速度)/2)²]，站内归一至 1，速度用 km/h。权重让五圈分散的区段影响较小，10 km/h 和 2 s 为选定的拟合尺度。优化为 scipy 有界非线性最小二乘，两个确定性起点取较低损失；逐站版只放开 μ，功率、两项面积和制动上限仍跨站共用。每个目标均用全圈 T049 求解。后验未参与标定。""",
            "## 最终参数与拟合误差",
            table(["模型","P (kW)","CdA (m²)","ClA (m²)","制动 (m/s²)","μ","J"],[
                [LABEL[mode],*(f"{fit[mode]['params'][key]:.3f}" for key in ["power_kW","cda_m2","cla_m2","brake_max_mps2"]),
                 (f"{fit[mode]['params']['mu']:.4f}" if mode=="shared" else "见逐站表"),f"{fit[mode]['objective']:.3f}"] for mode in LABEL]),
            "所有误差取模拟减参考；速度 RMSE 按距离等权计算，最高与最低为两条曲线各自全圈极值之差（km/h）。",
            table(["站","模型","质量 kg","μ","参考 s","模拟 s","Δ圈时 s","速度 RMSE","Δ最高速","Δ最低弯速"],[
                [NAMES[q.race],LABEL[q.mode],f"{q.mass:.1f}",f"{q.mu:.4f}",*(f"{v:.3f}" for v in [q.ref_s,q.sim_s,q.dt,q.rmse,q.dmax,q.dmin])] for q in m.itertuples()])]
    for mode,d in m.groupby("mode",sort=False):
        report.append(f"{LABEL[mode]}：五站等权平均 |圈时误差| {d.dt.abs().mean():.3f} s，平均速度 RMSE {d.rmse.mean():.2f} km/h。")
    report += ["## 参数合理性和可辨识性",
               """任务单给出的比较量级为功率 750–800 kW、CdA 0.9–1.3 m²、ClA 4–6 m²、μ 1.5–2。两模型面积和 μ 落入这些范围，有效功率明显偏低。2024 年无油质量基准为 798 kg；车队公开资料列出的 MGU-K 峰值 120 kW 只描述动力单元中的电机部分。这里的恒定轮上功率还会吸收收油、混动释放、传动损失与赛车线误差，不能作为发动机台架测量值。制动参数为机械减速度上限，另有抓地和气阻约束。[FIA 质量说明](https://www.fia.com/news/formula-1-commission-meeting-23072024-media-statement) · [Mercedes W15 技术规格](https://www.mercedesamgf1.com/f1-w15-2024-technical-specifications)""",
               "主拟合搜索范围为 P 450–1050 kW、CdA 0.5–2、ClA 1–9 m²、制动 10–65 m/s²、μ 0.8–2.8。局部线性化的参数相关矩阵保存在 outputs/correlation_shared.csv 和 outputs/correlation_track_mu.csv。按缩放雅可比的正规矩阵伪逆计算；边界处的数值仅说明参数补偿方向。"]
    for mode in LABEL:
        corr=pd.read_csv(out/f"correlation_{mode}.csv",index_col=0)
        mu_cols=[key for key in corr if key.startswith("mu")]
        report.append(f"{LABEL[mode]}：功率与 CdA 的局部相关 {corr.loc['power_kW','cda_m2']:.3f}；ClA 与 μ 的相关范围 {corr.loc['cla_m2',mu_cols].min():.3f} 至 {corr.loc['cla_m2',mu_cols].max():.3f}。两个起点的 J 分别为 "+'、'.join(f"{z:.5f}" for z in fit[mode]["start_objectives"])+"。")
    report+=["固定功率并重新优化其余参数。350 kW 探查主拟合下界之外，775 kW 对应任务单给的参考量级。表中 J 是加权损失，未作似然标度。",table(["模型","固定功率 kW","重新拟合 J","CdA m²","五站物理中点范围 s/圈"],[
        [LABEL[mode],f"{q['power']:.0f}",f"{q['objective']:.3f}",f"{q['params']['cda_m2']:.3f}",f"{min(q['fuel_mid'].values()):.4f}–{max(q['fuel_mid'].values()):.4f}"] for mode in LABEL for q in fit[mode]["power_profile"]])]
    report.append("把功率固定到 350 kW 时，两种拟合的损失都继续下降，但对应 CdA 下降到约 0.56–0.60 m²，低于任务参考量级。固定到 775 kW 时，两者 CdA 均触及搜索上界 2.0 m²，损失明显上升。速度与圈时数据在这套模型中无法同时支持参考功率和合理阻力；450 kW 是有界解，不能解释为测得的轮上功率。")
    report+=["## 燃油对照",
        """v3.1 的 gamma[race] 乘以站内中心化的剩余圈数，因变量是以秒计、减去分站中位数的圈时。故 γ 的单位为秒/多剩一圈，正值表示其他条件相同时，剩余圈数多一圈会更慢。以 model_laps 的 2024 行首次出现的 race_id 顺序重建 factorize 的映射，使用 outputs31_2024 的 gamma 全部 4000 个抽样，区间取 5% 和 95% 分位数。""",
        """物理系数 = [t(m+5)−t(m−5)]/10 × 总燃油/正赛圈数，单位为秒/多剩一圈。质量差分保持拟合参数固定。低/中/高取 105/107.5/110 kg；这个范围只传播总油量假设，不包含参数或赛道几何不确定性。中点落入、两区间相交分别按未舍入数值判断。""",
        table(["站","模型","∂t/∂m (s/kg)","物理中点 [低,高] s/圈","γ 均值 [5%,95%] s/圈","中点落入","区间相交"],[
            [NAMES[q.race],LABEL[q.mode],f"{q.dt_dm:.5f}",f"{q.mid:.4f} [{q.low:.4f},{q.high:.4f}]",f"{q.gamma:.4f} [{q.q05:.4f},{q.q95:.4f}]",'是' if q.mid_inside else '否','是' if q.intervals_overlap else '否'] for q in f.itertuples()])]
    for mode,d in f.groupby("mode",sort=False):
        report.append(f"{LABEL[mode]}：五站物理中点与后验均值的 Pearson r = {d.mid.corr(d.gamma):.3f}，Spearman ρ = {d.mid.rank().corr(d.gamma.rank()):.3f}；平均物理值减后验均值 {(d.mid-d.gamma).mean():+.4f} s/圈。")
    report+=["## 速度曲线与局部误差","灰带为五圈参考最小至最大值。两幅子图分别按各自模型的局部 RMSE 标出最差的三个互不重叠的 100 m 区间（黄色，1 为最大，可跨起终线）。区间起止与误差见 outputs/regions.csv。"]
    report.extend(f"![{NAMES[rid]}车速与误差区段](outputs/figures/{rid}.png)" for rid in NAMES)
    report+=["## 这些结果不能说明什么",
             """两种标定给出同方向、同数量级的燃油效应；逐站 μ 使圈时更接近实测，五站物理中点却都低于对应后验 90% 区间。仅在共用模型的澳大利亚站，油耗范围上端与后验区间相交。因此现有标定不足以独立验证各站 γ 的大小。五点相关只是描述，多出的四个抓地参数也会改善样本内圈时。""",
             """五圈可能来自不同车手、圈次、胎龄和油量，曲率经过平滑，中心线与真实赛车线、弯道相位都可能不同。质点模型没有坡度、混动部署、轮胎随载荷和温度的变化；车手节油、保胎、赛道演化等随比赛进程变化的因素也可能进入后验 γ。没有实测油量或受控对照，不能将两种估计的差额归因于其中一项。""",
             "## 复现","在项目根目录执行 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python analysis/lapsim_calibration/calibrate.py`；用 `--out /临时目录/outputs` 重跑，然后对比两份 outputs 和各自上一级的 report.md。运行 `.venv/bin/python analysis/lapsim_calibration/verify.py` 读回并交叉检查。"]
    (out.parent/"report.md").write_text("\n\n".join(report)+"\n")
