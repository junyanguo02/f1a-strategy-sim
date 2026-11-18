"""Produce the v2 report from independent numerical outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from report import NAMES, LABEL, ENG, table
from calibrate import HERE
import matplotlib.pyplot as plt


def render(out):
    def read(name):
        return pd.read_csv(out/(name+".csv"))
    a,m,p,c,e=map(read,["alignment","metrics","parameters","profiles","evolution"])
    stats,pred=[],[]
    for q in p.itertuples():
        d=m[(m.power==q.power)&(m["mode"]==q.mode)]
        mus=" / ".join(f"{v:.3f}" for v in d.mu.unique())
        stats.append([str(q.power),LABEL[q.mode],f"{q.cda:.3f}",f"{q.cla:.3f}",mus,f"{q.brake:.2f}",
                      f"{q.power_per_cda:.1f}",f"{q.objective:.3f}",f"{d.dt.abs().mean():.3f}",f"{d.rmse.mean():.2f}",
                      f"{d.dmax.abs().mean():.2f}",f"{d.dmin.abs().mean():.2f}"])
        pred.append([str(q.power),LABEL[q.mode],f"{d.fuel.corr(d.gamma):.3f}",f"{d.fuel.rank().corr(d.gamma.rank()):.3f}",
                     f"{(d.gamma-d.fuel).mean():+.4f}",str(((d.q05<=d.fuel)&(d.fuel<=d.q95)).sum()),
                     str(((d.fuel_low<=d.q95)&(d.fuel_high>=d.q05)).sum())])
    primary=m[(m.power==750)&(m["mode"]=="track_mu")].set_index("race")
    common=e[e["sample"]=="same_traffic"].set_index("race")
    align_rows,fuel_rows,evol_rows,v1_rows=[],[],[],[]
    v1=json.loads((HERE/"outputs/fit.json").read_text())
    for rid,name in NAMES.items():
        x=a[a.race==rid]
        best=x.loc[x.v1_rmse.idxmin()]
        main=primary.loc[rid]
        mean_speed=(5*len(c[(c.race==rid)&(c.power==750)&(c["mode"]=="track_mu")]))/(main.sim_s-main["dt"])
        align_rows.append([name,f"{int(best.delta_m):+d}",f"{x[x.delta_m==0].v1_rmse.iloc[0]:.2f}",f"{best.v1_rmse:.2f}",
                           f"{best.delta_m/mean_speed:.2f}",f"{main.rmse_unaligned:.2f}",f"{main.rmse:.2f}"])
        d=m[m.race==rid]
        fuel_rows.append([name,f"{d.dt_dm.min():.5f}–{d.dt_dm.max():.5f}",f"{main.dt_dm:.5f}",f"{main.fuel:.4f} [{main.fuel_low:.4f},{main.fuel_high:.4f}]",
                          f"{d.fuel.min():.4f}–{d.fuel.max():.4f}",f"{d.fuel_low.min():.4f}–{d.fuel_high.max():.4f}",
                          f"{main.gamma:.4f} [{main.q05:.4f},{main.q95:.4f}]",
                          "是" if main.q05<=main.fuel<=main.q95 else "否"])
        b=common.loc[rid]
        other=e[e.race==rid].set_index("sample")
        evol_rows.append([name,f"{b.n:.0f}/{b.drivers:.0f}",f"{b.total_trend:+.4f}",f"{b.posterior_gap:+.4f}",
                          f"{b.residual:+.4f} [{b.lo:+.4f},{b.hi:+.4f}]",f"{b.ratio:.2f}",
                          f"{other.loc['matched','residual']:+.4f}",f"{other.loc['clear_air','residual']:+.4f}"])
        row=[name]
        for mode in LABEL:
            for power in (450,775):
                scenario=next(q for q in v1[mode]["power_profile"] if q["power"]==power)
                row.append(f"{scenario['fuel_mid'][rid]*main.laps/107.5:.5f}")
        v1_rows.append(row)
    figdir=out/"figures"
    figdir.mkdir(exist_ok=True)
    for (rid,name),english in zip(NAMES.items(),ENG):
        g=c[(c.race==rid)&(c.power==750)]
        d=g[g["mode"]=="track_mu"]
        fig,(ax,err)=plt.subplots(2,1,figsize=(10,5),sharex=True,gridspec_kw={"height_ratios":[2.2,1]},layout="constrained")
        ax.fill_between(d.s,d.low,d.high,color="#CCD8DC",label="Aligned five-lap range")
        ax.plot(d.s,d.ref_raw,color="#959595",ls=":",lw=1,label="Original reference")
        ax.plot(d.s,d.ref,color="#26353B",lw=1.1,label="Aligned reference")
        for mode,color in [("shared","#2878A1"),("track_mu","#C15A17")]:
            z=g[g["mode"]==mode]
            label="Shared grip" if mode=="shared" else "Circuit grip"
            ax.plot(z.s,z.sim,color=color,lw=.95,label=label)
            err.plot(z.s,z.sim-z.ref,color=color,lw=.9)
        err.axhline(0,lw=.75,color="#363636")
        ax.set_ylabel("Speed (km/h)")
        err.set(xlabel="Centreline distance (m)",ylabel="Sim − ref (km/h)")
        ax.legend(ncol=3,fontsize=8,loc="lower right")
        fig.suptitle(f"2024 {english}   |   750 kW   |   Reference offset +{primary.loc[rid,'delta_m']:.0f} m")
        fig.savefig(figdir/(rid+".png"),dpi=135)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
    for i,(rid,name) in enumerate(NAMES.items()):
        d=m[m.race==rid]
        b=primary.loc[rid]
        ax.plot([d.fuel_low.min(),d.fuel_high.max()],[i+.12]*2,color="#2379A2",lw=4,label="Fuel & parameter envelope" if i==0 else None)
        ax.plot(b.fuel,i+.12,"o",color="#124858",label="750 kW, circuit grip" if i==0 else None)
        ax.plot([b.q05,b.q95],[i-.12]*2,color="#C5671C",lw=4,label="Posterior 90% interval" if i==0 else None)
        ax.plot(b.gamma,i-.12,"D",color="#8F3F00",label="Posterior mean" if i==0 else None)
    ax.set_yticks(range(5),ENG)
    ax.set_xlabel("Lap-time change per remaining lap (s/lap)")
    ax.invert_yaxis()
    ax.legend(fontsize=8,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.22))
    fig.savefig(figdir/"fuel.png",dpi=135)
    plt.close(fig)
    lines=["# T052｜五站速度对齐、固定功率标定及燃油对照",
           "主方案取预设功率中档 750 kW、逐站 μ，以反映路面与轮胎差异；共用 μ 及 700/800 kW 均作敏感性检查。主方案选择不使用后验 γ。全部比较在同一批五个快圈上，属样本内结果。",
           "## 1　参考曲线的距离偏移",
           "固定 T050 共用参数模拟曲线，对每站原参考均速在 ±60 m 内按 5 m 周期平移，取未加权全圈 RMSE 最小的 δ。等价于最大化循环互相关：平移后曲线的均值、方差不变。正号取 ref(s+δ)，将参考特征向较小弧长移动；没有移动曲率、模拟轨迹或实测圈时。各次拟合沿用同一 δ，参考带和精度权重一同平移。",
           table(["站","δ m","v1 原 RMSE","v1 对齐 RMSE","δ/均速 s","主方案原 RMSE","主方案对齐 RMSE"],align_rows),
           "五站方向一致，量级 10–25 m；换算时间仅按圈长/参考圈时作数量级诊断。常数距离不能表示全圈恒定时延，弯道曲率平滑、弧长映射、车手制动点和模型误差也可能造成偏移。建议 T048 维护者在原时间戳上先独立检验，再决定是否更改映射。",
           "## 2　固定功率与拟合误差",
           """三档都是有效轮上功率假设。两种模型分别拟合跨站共用的 CdA、ClA、机械制动上限；μ 共用或每站一项。T050 的各站固定质量（798 kg + 按五圈圈次估计的均余油）、5 m 闭环积分和损失权重不变：每站 J = 速度残差/10 km/h 的加权均方 + 圈时残差/2 s 的平方；点权重 ∝ 1/[100+((max−min)/2)²]，站内和为 1。主拟合上下界 CdA 0.5–3 m²（放宽 v1 的 2，以观察边界外趋势）、ClA 1–9 m²、制动 10–65 m/s²、μ 0.8–2.8。每组两个起点取低损失。""",
           "μ 顺序为巴林、澳大利亚、西班牙、匈牙利、蒙扎。下表四项误差为五站等权均值；极值误差取绝对值，速度单位 km/h。逐站有符号误差在 metrics.csv。",
           table(["kW","抓地","CdA","ClA","μ","制动 m/s²","P/CdA","J","圈时 MAE s","速度 RMSE","最高速 MAE","最低速 MAE"],stats),
           "终端直线速度的三次方正比于 P/CdA；仅凭极速只能识别这个比值。加速段理论上也含单独的功率信息，但恒定功率、未建模收油或 ERS 状态、中心线误差都会影响反推；不能据此测量真实轮上功率。correlations.csv 为每组自由参数基于局部雅可比正规矩阵伪逆的相关矩阵，固定功率不进入矩阵；边界处不作统计置信区间解释。"]
    cor=pd.read_csv(out/"correlations.csv")
    strongest=[]
    for (power,mode),group in cor[cor.param<cor.other].groupby(["power","mode"]):
        b=group.loc[group.r.abs().idxmax()]
        strongest.append([str(power),LABEL[mode],b.param,b.other,f"{b.r:+.3f}"])
    lines.extend([table(["kW","模型","局部相关最大的一对","另一参数","r"],strongest),
                  "## 3　燃油效应的稳健性",
                  "固定其余参数，中心差分 [t(m+5)−t(m−5)]/10 求秒/kg。乘总油量/正赛圈数，得到每多一圈剩余燃油增加的圈时。中点总油量 107.5 kg，单组油耗范围 105–110 kg，跨六组包络同时涵盖功率、抓地方案和油耗假设。包络不是置信区间。2024 v3.1 的 γ 为中心化剩余圈数的系数；按原输入首次出现的分站顺序取全部 4000 个抽样的均值和 5%、95% 分位数。",
                  table(["站","六组 ∂t/∂m s/kg","主 ∂t/∂m","主燃油中点 [油耗范围]","六组中点范围","六组含油耗包络","后验均值 [90%]","主中点落入"],fuel_rows),
                  table(["功率","模型","Pearson r","Spearman ρ","均值差 γ−物理 s/圈","中点落入/5","区间相交/5"],pred),
                  "相关系数比较五站物理中点和后验均值。五点相关及排序不能当作外推验证。各字段及每组 ∂t/∂m 见 metrics.csv。",
                  "v1 未对齐的固定功率 profile 作为额外对照。下表将 T050 保存的物理中点按正赛圈数/107.5 还原至 s/kg；与 v2 的差别还包含对齐与 CdA 搜索范围变化。",
                  table(["站","v1 共用450","共用775","逐站450","逐站775"],v1_rows),
                  "![六组物理包络与后验](figures/fuel.png)",
                  "## 4　赛道演化代理",
                  """用同站、同车手、同配方、同精确胎龄且不同 stint 的绿旗清洁圈作差。同一匹配组只取最早与最晚圈，要求相隔至少 10 圈。主代理还要求两圈处于相同前车间隔类别（<1、1–2、≥2 s；缺失排除）。拟合无截距差分 gain = b × 圈数间隔，gain 为早圈减晚圈的秒数，所以 b>0 表示后段更快。b 减去主方案物理中点后得到剩余比赛进程趋势。95% 区间按车手整簇重抽 4000 次，固定种子与物理值；没有传播模型参数或后验不确定性。另报不控制交通和两圈均 ≥2 s 的对照。""",
                  table(["站","匹配数/车手","总趋势 b","γ−物理","剩余趋势 [95%]","剩余/差距","不限交通剩余","净空剩余"],evol_rows),
                  "比例为量级对照，不截断到 0–1。负值代表与所需方向相反，大于 1 代表剩余趋势大于要解释的差额。后验与匹配使用同源圈时，燃油消耗与比赛进程同步，缺少独立的赛道抓地测量；即使剩余趋势恰好相等，也不能给出橡胶化的因果贡献比例。还可能混有胎组历史、对手、管理策略、赛道温度及只留下清洁圈的选择效应。"])
    positive=(common.residual>0).sum()
    covered=((common.lo<=common.posterior_gap)&(common.posterior_gap<=common.hi)).sum()
    lines.append(f"五站等权：待解释差距 {common.posterior_gap.mean():+.4f} s/圈，主代理剩余趋势 {common.residual.mean():+.4f} s/圈；{positive}/5 站点估计为正，{covered}/5 站的簇重抽区间包含所需差额。单站结果及样本支持应与总体均值一起看。")
    lines.append("按五站等权均值计算，剩余趋势相当于待解释差距的约 86%。匈牙利仅约 46%，且区间跨零；西班牙虽接近差额，主样本仅有五名车手。现有数据支持量级比较，无法给出橡胶化单独贡献的比例。")
    lines.append("六组标定和 105–110 kg 油耗包络下，五站的物理上沿均低于本站 γ 的后验第 5 百分位。站间排序对 μ 的设定敏感：逐站 μ 的三档 Spearman 为 0.9–1.0，共用 μ 为 −0.1–0.3。三档中 700 kW 的目标函数最低；750 kW 逐站 μ 的平均圈时绝对误差约 0.10 s，但速度曲线 RMSE 仍有 16.56 km/h，局部误差会在圈时积分中抵消。逐站 μ 的假设功率从 700 增至 800 kW 时，CdA 从 1.84 升至 2.19 m²，P/CdA 只从 380 降至 366 kW/m²。")
    lines+=["## 5　主功率档的速度残差","实线对比对齐后的参考和两种 750 kW 拟合；点线保留原始参考。下图残差均为模拟减对齐参考。"]
    lines.extend(f"![{name}](figures/{rid}.png)" for rid,name in NAMES.items())
    lines+=["## 复现","本次使用已有的 work/T034_env/validation-venv。在项目根目录运行 `OPENBLAS_NUM_THREADS=1 work/T034_env/validation-venv/bin/python analysis/lapsim_calibration/recalibrate.py`。用 `--out <另一空目录>` 全量重复，然后运行 `verify_v2.py --compare <另一目录>` 复核数值和文件。v1 输入与输出未覆写。"]
    (out/"report.md").write_text("\n\n".join(lines)+"\n")
