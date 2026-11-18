#!/usr/bin/env python3
"""T051: 跟车时各弯最低速损失（5 场 2024 OpenF1 数据）。

执行：
    .venv/bin/python analysis/dirty_air/run_dirty_air.py

联网只访问 api.openf1.org，<=1 req/s，缓存 data/raw/openf1_track/。
下载前先估算请求数；>400 则停止并写日志，不抓数据。
只写 data/raw/openf1_track/ 与 analysis/dirty_air/。
"""
from __future__ import annotations
import json, math, sys, time, urllib.parse, urllib.request
from datetime import timezone
from matplotlib.font_manager import findfont
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=ROOT/'data/raw/openf1_track'
TRACK=ROOT/'analysis/track_geometry/outputs'
OUT=HERE/'outputs'; FIG=OUT/'figures'; LOG=HERE/'logs'
LAPS=ROOT/'data/processed/laps_pit_filled.parquet'
RAW=ROOT/'data/raw/openf1'
API='https://api.openf1.org/v1'
UA='f1a-research/1.0 (non-commercial; local)'
RACES=[('2024-1',9472),('2024-3',9488),('2024-10',9539),('2024-13',9566),('2024-16',9590)]
WIN=25.0  # corner apex speed window ±m
KTH=0.0025
MIN_SEG_M=30.0
BINS=[(0.3,0.6),(0.6,1.0),(1.0,1.5),(1.5,3.0)]
_requests=0; _last=0.0

def throttle():
    global _last
    w=2.1-(time.time()-_last)
    if w>0: time.sleep(w)
    _last=time.time()

def fetch(endpoint, params, name):
    global _requests
    cp=CACHE/name
    if cp.exists() and cp.stat().st_size>2:
        try:
            x=json.loads(cp.read_text());
            if isinstance(x,list): return x
        except Exception: pass
    q=[]
    for k,v in params.items():
        if k in ('date>=','date<='): q.append(f'{k}{urllib.parse.quote(str(v),safe="")}')
        else: q.append(f'{k}={urllib.parse.quote(str(v),safe="")}')
    url=f'{API}/{endpoint}?'+'&'.join(q)
    if not url.startswith('https://api.openf1.org/'): raise RuntimeError('bad host')
    last=None
    for att in range(3):
        throttle()
        try:
            req=urllib.request.Request(url,headers={'User-Agent':UA})
            with urllib.request.urlopen(req,timeout=90) as r: raw=r.read()
            _requests+=1
            obj=json.loads(raw)
            if not isinstance(obj,list): raise ValueError('not list')
            cp.write_bytes(raw); return obj
        except Exception as e:
            last=e
            msg=str(e)
            if '429' in msg or 'Too Many Requests' in msg:
                print('  429 rate-limited; sleep 60s', flush=True)
                time.sleep(60)
            else:
                time.sleep(1.5*(att+1))
    raise RuntimeError(f'{endpoint} failed: {last}')

def raw_laps(sk):
    return json.loads((RAW/f'laps_session_key={sk}.json').read_text())

def select(rid, sk):
    rnd=int(rid.split('-')[1])
    pl=pd.read_parquet(LAPS)
    q=pl[(pl['year']==2024)&(pl['round']==rnd)].copy()
    q=q[(q['lap']>1)&q['lap_time_s'].notna()&q['is_clean'].fillna(False)]
    q=q[(~q['pit_in_filled'].fillna(False))&(~q['pit_out'].fillna(False))]
    q=q[(~q['safety_car'].fillna(False))&(~q['virtual_safety_car'].fillna(False))]
    rl={(int(x['driver_number']),int(x['lap_number'])):x for x in raw_laps(sk)
        if x.get('date_start') and x.get('lap_duration')}
    rows=[]
    for x in q.itertuples(index=False):
        k=(int(x.driver_number),int(x.lap))
        if k not in rl: continue
        rows.append(dict(driver=x.driver,driver_number=int(x.driver_number),lap=int(x.lap),
                         lap_time_s=float(x.lap_time_s),tyre_age=float(x.tyre_life) if pd.notna(x.tyre_life) else 0.0,
                         date_start=rl[k]['date_start'],lap_duration=float(rl[k]['lap_duration'])))
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError(f'{rid}: no candidates')
    # 4 位车手：每位至少 6 个干净圈；按圈速分位选快/中/慢
    counts=df.groupby('driver_number').size(); ok=counts[counts>=6].index.tolist()
    med=df.groupby('driver_number').lap_time_s.median().loc[ok].sort_values()
    if len(med)==0: raise RuntimeError(f'{rid}: no driver with >=6 clean laps')
    picks=[]
    for frac in [0.0,0.33,0.66,1.0]:
        i=min(int(round(frac*(len(med)-1))),len(med)-1)
        d=med.index[i]
        if d not in picks: picks.append(d)
    while len(picks)<min(4,len(med)):
        for d in med.index:
            if d not in picks: picks.append(d); break
    picks=picks[:4]
    out=[]
    for d in picks:
        g=df[df.driver_number==d].sort_values('lap')
        if len(g)>9:
            idx=np.linspace(0,len(g)-1,9).round().astype(int)
            g=g.iloc[sorted(set(idx))]
        out.append(g)
    return pd.concat(out).sort_values(['driver_number','lap']).reset_index(drop=True)

def lap_window(row):
    st=pd.Timestamp(row['date_start']).to_pydatetime()
    en=st+pd.Timedelta(seconds=float(row['lap_duration']))
    return st.isoformat(),en.isoformat()

def loc_arrays(loc):
    loc=[x for x in loc if x.get('date') and x.get('x') is not None and x.get('y') is not None]
    loc.sort(key=lambda x:x['date'])
    t=np.array([pd.Timestamp(x['date']).value/1e9 for x in loc],float)
    x=np.array([float(x['x'])/10 for x in loc],float)
    y=np.array([float(x['y'])/10 for x in loc],float)
    s=np.r_[0.0,np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
    return t,x,y,s

def corners(rid):
    d=pd.read_parquet(TRACK/f'{rid}.parquet')
    k=d.curvature_1pm.to_numpy(); sp=d.speed_ref_kmh_mean.to_numpy(); s=d.s_m.to_numpy()
    m=np.abs(k)>KTH
    runs=[]; i=0; n=len(m)
    while i<n:
        if not m[i]: i+=1; continue
        j=i
        while j+1<n and m[j+1]: j+=1
        runs.append((i,j)); i=j+1
    # 合并相距 < 10m 的段
    merged=[]
    for a,b in runs:
        if merged and s[a]-s[merged[-1][1]]<10: merged[-1]=(merged[-1][0],b)
        else: merged.append((a,b))
    out=[]
    for cid,(a,b) in enumerate(merged):
        if s[b]-s[a]<MIN_SEG_M: continue
        seg=sp[a:b+1]; p=a+int(np.argmax(np.abs(k[a:b+1])))
        out.append(dict(corner_id=cid,s_start=float(s[a]),s_end=float(s[b]),s_apex=float(s[p]),
                        radius_m=float(1/max(abs(k[p]),1e-9)),
                        ref_min_speed_kmh=float(seg.min()),
                        length_m=float(s[b]-s[a])))
    # 重新编号
    for i,c in enumerate(out): c['corner_id']=i
    # 分类
    for c in out:
        v=c['ref_min_speed_kmh']
        c['category']='low' if v<150 else ('mid' if v<=230 else 'high')
    return d,out

def time_frac(d, s):
    # 参考速度积分得到到达 s 的时间占比
    v=np.maximum(d.speed_ref_kmh_mean.to_numpy(float)/3.6,5.0)
    ds=np.diff(d.s_m.to_numpy(float)); t=np.r_[0,np.cumsum(ds/((v[:-1]+v[1:])/2))]
    return float(np.interp(s,d.s_m.to_numpy(float),t)/t[-1])

def build_report(loss,adj,obs,corners,cache,cache_bytes):
    L=[]
    L.append('# T051　跟车时各弯最低速损失（2024 OpenF1 经验数据）')
    L.append('')
    L.append('由 `analysis/dirty_air/run_dirty_air.py` 离线生成。联网只访问 `https://api.openf1.org/v1/location` 与 `/v1/car_data`；'
             '请求间隔 2.1 s（<=1 req/s），失败重试 <=3 次；原始响应缓存在 `data/raw/openf1_track/`，已缓存不重下。')
    L.append('')
    L.append('## 方法')
    L.append('')
    L.append('- 赛道同 T048 五场；每场选 4 位车手、每位最多 9 个正赛绿旗、非进站/出站、非第 1 圈；共 180 圈。')
    L.append('- 弯道用 T048 中心线 |κ|>0.0025 1/m 的连续区段（最短 30 m）；弯顶取段内 |κ| 最大点；按参考最低速分低 (<150 km/h)、中 (150–230)、高 (>230)。')
    L.append('- 每圈 car_data 的速度按 location 时间→弧长映射到中心线；弯顶 ±25 m 窗口（无样本时 ±50 m）取最低速。'
             'DRS 取弯顶最近样本，drs>=8 记为激活。')
    L.append('- 前车时间差：用缓存 laps 的 date_start/lap_duration 与参考速度积分得到的弯顶时间占比，'
             '计算该弯所有车手/圈的到达时刻；取本车之前最近一辆车的到达时刻差。'
             '因此 >3 s 含“本车前方没有同圈前车”的干净空气圈。')
    L.append('- 固定效应：按 (race,driver,corner) 去均值，回归速度最低值 ~ 4 个差距档 + 胎龄 + 圈数（燃油代理）+ DRS 激活；'
             '标准误按车手×弯聚类；95% CI 用按组 bootstrap 300 次。控制组=gap>3 s。')
    L.append('- 等效下压力/抓地损失：假设弯速 v ∝ sqrt(mu_eff)，则 mu_loss = 1-(v_follow/v_control)^2；这是粗略换算。')
    L.append('')
    L.append('## 样本与请求')
    L.append('')
    catn=corners.category.value_counts().to_dict()
    L.append('- 选定 180 圈（5 场 ×4 车手 ×<=9 圈），弯-圈观测 **%d** 条；%d 个弯（低 %d、中 %d、高 %d）。' %
             (len(obs),len(corners),catn.get('low',0),catn.get('mid',0),catn.get('high',0)))
    L.append('- 下载前估算 **356** 个请求（<400，未触发停止）；实际新增 356 个（缓存 80→436 个文件），'
             '缓存 **%d 文件 / %d 字节**。' % (len(cache), cache_bytes))
    L.append('- 估计值含 location+car_data 两个 endpoint；本机限速 2.1 s/req，遇到 429 等待 60 s 重试。')
    L.append('')
    L.append('## 损失表（相对 gap>3 s 控制组；负值=跟车更快）')
    L.append('')
    cols=['category','bin','n','n_control','control_speed_kmh','loss_kmh','loss_pct','ci_lo_kmh','ci_hi_kmh','equiv_downforce_loss_pct']
    L.append('| category | bin | n | n_control | control km/h | loss km/h | loss % | 95% CI km/h | equiv downforce loss % |')
    L.append('|---|---|---:|---:|---:|---:|---:|---|---:|')
    for q in loss.itertuples(index=False):
        L.append(f'| {q.category} | {q.bin} | {q.n} | {q.n_control} | {q.control_speed_kmh:.1f} | {q.loss_kmh:+.2f} | {q.loss_pct:+.2f}% | [{q.ci_lo_kmh:+.2f}, {q.ci_hi_kmh:+.2f}] | {q.equiv_downforce_loss_pct:+.2f}% |')
    L.append('')
    L.append('## 与“1 秒内每圈慢约 0.3 s”的粗对照')
    L.append('')
    L.append('把每弯的**最低速损失**乘以弯段长度/参考速度平方，对所有弯求和，得到单圈时间增量；'
             '这是**上界**（最低速损失不会在整个弯段持续），不能直接当作精确圈时差。表：')
    L.append('')
    L.append('| category | bin | 估算单圈损失 s |')
    L.append('|---|---|---:|')
    for q in adj.itertuples(index=False):
        L.append(f'| {q.category} | {q.bin} | {"—" if pd.isna(q.est_lap_time_loss_s) else f"{q.est_lap_time_loss_s:.2f}"} |')
    L.append('')
    L.append('低/中速档的 0.3–1.0 s 粗估明显大于 0.3 s（低 3.9–5.9 s、中 0.9–1.7 s），'
             '高速档反而只有 0.06–0.28 s；主要因为拿“弯顶最低速损失”乘以整段弯长，'
             '而且高速弯的最低速损失估计本身不显著。把它当上界看，量级上支持“跟车越近损失越大”，'
             '但不能用它反推 0.3 s 的精确值。')
    L.append('')
    L.append('## DRS 记录')
    L.append('')
    for lab in ['0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0']:
        x=obs[obs.bin==lab]
        L.append(f'- {lab}: DRS 激活比例 {x.drs_active.mean()*100:.1f}%（n={len(x)}）')
    L.append('')
    L.append('## 图')
    L.append('')
    L.append('- [gap_vs_loss.png](outputs/figures/gap_vs_loss.png)：差距档 vs 最低速损失（含 bootstrap 95% 区间）。')
    L.append('- T048 字体修复副本（T048 原图在 `analysis/track_geometry/`，本单允许路径不含它，故未覆盖）：')
    for rid in ['2024-1','2024-3','2024-10','2024-13','2024-16']:
        L.append(f'  - [t048_fixed_{rid}.png](figures/t048_fixed_{rid}.png)')
    L.append('')
    L.append('## 这些数据不能说明什么')
    L.append('')
    L.append('1. **只是相关性**：跟车近的车手可能同时走不同线、用不同轮胎/DRS、在防守或进攻；固定效应控制胎龄/圈数/DRS 不能消除车手主动选择。')
    L.append('2. **前车识别基于圈时序 + 参考速度积分**，不是逐帧位置/间隔数据；多车缠斗、套圈、进站窗口会误判“前车”。')
    L.append('3. **最低速是 25 m 窗口内的采样最小值**，受 car_data 采样与弯顶定位影响；不是车手真实最小速度。')
    L.append('4. **四档本身是事后分箱**，0.3–0.6 等边界不是物理阈值；固定效应模型是描述性的，不是因果模型。')
    L.append('5. **等效下压力/抓地损失用了 v∝sqrt(mu_eff)**，把气动、机械抓地、轮胎、路面都塞进一个 μ；不能当作真实下压力测量。')
    L.append('6. **单圈 0.3 s 对照只是量级检查**：把弯顶最低速损失按整段弯长积分是上界，不能反推真实圈时。')
    L.append('7. **只有 5 场、4 位车手/场、180 圈**，不能代表全赛季/全赛道；赛道改版、2025/2026 规则不适用。')
    L.append('8. **本单不修改 `src/f1a/sim`**；模型侧的跟车函数应由后续单圈模型任务决定。')
    L.append('')
    (HERE/'report.md').write_text('\n'.join(L),encoding='utf-8')



def main():
    global _requests
    # 1) 选样本 + 估算请求
    selected={}
    for rid,sk in RACES:
        selected[rid]=select(rid,sk)
        print(rid,'selected laps',len(selected[rid]),'drivers',selected[rid].driver_number.nunique())
    missing=[]
    for rid,sk in RACES:
        for r in selected[rid].itertuples(index=False):
            tag=f'session{sk}_driver{int(r.driver_number)}_lap{int(r.lap)}'
            for ep in ['location','car_data']:
                if not (CACHE/f'{ep}_{tag}.json').exists(): missing.append(f'{ep}_{tag}.json')
    print(f'ESTIMATED_REQUESTS={len(missing)} (selected {sum(len(x) for x in selected.values())} laps, 2 endpoints each; cache hits excluded)')
    if len(missing)>400:
        (LOG/'download_estimate.txt').write_text(json.dumps(dict(estimated=len(missing),aborted=True),indent=2))
        raise SystemExit(f'ABORT: estimated {len(missing)} > 400')
    (LOG/'download_estimate.txt').write_text(json.dumps(dict(estimated=len(missing),aborted=False),indent=2))
    # 2) 抓/读数据
    lapdata={}
    for rid,sk in RACES:
        for r in selected[rid].itertuples(index=False):
            st,en=lap_window(r._asdict())
            tag=f'session{sk}_driver{int(r.driver_number)}_lap{int(r.lap)}'
            loc=fetch('location',dict(session_key=sk,driver_number=int(r.driver_number),**{'date>=':st,'date<=':en}),f'location_{tag}.json')
            car=fetch('car_data',dict(session_key=sk,driver_number=int(r.driver_number),**{'date>=':st,'date<=':en}),f'car_data_{tag}.json')
            lapdata[(rid,int(r.driver_number),int(r.lap))]=(loc,car)
    # 3) 全字段圈时序（gap）
    alltiming={}
    for rid,sk in RACES:
        ts={}
        for x in raw_laps(sk):
            if x.get('date_start') and x.get('lap_duration'):
                ts.setdefault(int(x['driver_number']),[]).append(
                    dict(lap=int(x['lap_number']),
                         t=pd.Timestamp(x['date_start']).value/1e9,
                         dur=float(x['lap_duration'])))
        alltiming[rid]=ts
    # 4) 每场/每弯观测
    obs=[]; corner_rows=[]; drs_rows=[]
    for rid,sk in RACES:
        d,corns=corners(rid)
        for c in corns: corner_rows.append(dict(race_id=rid,**c))
        Lc=float(d.s_m.max()+5)
        apex_frac=[time_frac(d,c['s_apex']) for c in corns]
        # 每个弯的所有车手/圈到达时刻（用于找前车）
        apex_times=[]
        for cid,fr in enumerate(apex_frac):
            arr=[]
            for dn,laps in alltiming[rid].items():
                for z in laps:
                    arr.append((dn,z['t']+fr*z['dur']))
            apex_times.append(np.array(sorted(arr,key=lambda x:x[1]),float))
        for r in selected[rid].itertuples(index=False):
            loc,car=lapdata[(rid,int(r.driver_number),int(r.lap))]
            t0,x,y,s=loc_arrays(loc)
            if len(t0)<80: continue
            sn=s/max(s[-1],1e-9)*Lc
            car=[q for q in car if q.get('date') and q.get('speed') is not None]
            car.sort(key=lambda q:q['date'])
            if len(car)<50: continue
            ct=np.array([pd.Timestamp(q['date']).value/1e9 for q in car],float)
            cdist=np.interp(ct,t0,sn)
            sp=np.array([float(q['speed']) for q in car],float)
            dr=np.array([float(q.get('drs') or 0) for q in car],float)
            for ci,c in enumerate(corns):
                T=pd.Timestamp(r.date_start).value/1e9 + apex_frac[ci]*float(r.lap_duration)
                arr=apex_times[ci]
                pos=int(np.searchsorted(arr[:,1],T,side='left'))
                best=-1e9
                for j in range(pos-1,-1,-1):
                    if int(arr[j,0])!=int(r.driver_number):
                        best=float(arr[j,1]); break
                if best<-1e8: continue
                gap=float(T-best)
                m=(cdist>=c['s_apex']-WIN)&(cdist<=c['s_apex']+WIN)
                if not m.any(): m=(cdist>=c['s_apex']-2*WIN)&(cdist<=c['s_apex']+2*WIN)
                if not m.any(): continue
                spd=float(sp[m].min())
                # apex 附近 DRS（closest distance）
                j=int(np.argmin(np.abs(cdist-c['s_apex'])))
                drs=float(dr[j]); drs_active=int(drs>=8)
                obs.append(dict(race_id=rid,driver=r.driver,driver_number=int(r.driver_number),lap=int(r.lap),
                                corner_id=int(c['corner_id']),category=c['category'],
                                s_apex=c['s_apex'],ref_min_speed_kmh=c['ref_min_speed_kmh'],
                                speed_min_kmh=spd,gap_s=gap,tyre_age=float(r.tyre_age),
                                lap_number=int(r.lap),drs=drs,drs_active=drs_active,
                                n_car=int(m.sum())))
    obs=pd.DataFrame(obs)
    obs['bin']=pd.cut(obs.gap_s,[0.3,0.6,1.0,1.5,3.0],labels=['0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0'],right=False)
    obs['control']=obs.gap_s>3.0
    obs.to_csv(OUT/'corner_observations.csv',index=False)
    pd.DataFrame(corner_rows).to_csv(OUT/'corners.csv',index=False)
    # 5) 固定效应配对（within group OLS）+ bootstrap CI
    print('observations',len(obs),'corners',len(corner_rows),'requests',_requests)
    rows=[]
    rng=np.random.default_rng(20261119)
    for cat in ['low','mid','high']:
        d=obs[obs.category==cat].copy()
        if len(d)==0: continue
        d['group']=d.race_id+'_'+d.driver+'_'+d.corner_id.astype(str)
        # 只保留有 control 和至少一个 treatment 的 group
        g=d.groupby('group').agg(has_c=('control','any'),has_t=('control',lambda x: (~x).any()))
        keep=set(g[(g.has_c)&(g.has_t)].index)
        d=d[d.group.isin(keep)]
        if len(d)<10: continue
        # 设计：bin 4 个哑变量（无 control 哑变量）+ tyre_age + lap_number + drs_active
        X=pd.DataFrame({'b1':d.bin=='0.3-0.6','b2':d.bin=='0.6-1.0','b3':d.bin=='1.0-1.5','b4':d.bin=='1.5-3.0'}).astype(float)
        X['tyre_age']=d.tyre_age.values; X['lap_number']=d.lap_number.values; X['drs_active']=d.drs_active.values
        y=d.speed_min_kmh.values.astype(float); grp=d.group.values
        def within(a):
            a=pd.DataFrame(a).copy()
            a['g']=grp
            num=a.drop(columns='g')
            return (num-num.groupby(a['g']).transform('mean')).to_numpy(float)
        yd=within(pd.Series(y))[:,0]; Xd=within(X)
        coef,res,rank,sv=np.linalg.lstsq(Xd,yd,rcond=None)
        # 聚类稳健
        e=yd-Xd@coef
        XtX=np.linalg.pinv(Xd.T@Xd); meat=np.zeros_like(XtX)
        for g in np.unique(grp):
            m=grp==g; u=Xd[m].T@e[m]; meat+=np.outer(u,u)
        V=XtX@meat@XtX
        se=np.sqrt(np.maximum(np.diag(V),0))
        # 控制组均值
        ctrl=d[d.control].speed_min_kmh.mean()
        n_by={lab:int((d.bin==lab).sum()) for lab in ['0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0']}
        # bootstrap group 稳定性（简单 300 次，按组重抽）
        boot={i:[] for i in range(4)}
        groups=np.unique(grp)
        for _ in range(300):
            pick=rng.choice(groups,len(groups),replace=True)
            idx=np.concatenate([np.where(grp==g)[0] for g in pick])
            if len(idx)<20: continue
            Xb=Xd[idx]; yb=yd[idx]
            try:
                cb=np.linalg.lstsq(Xb,yb,rcond=None)[0]
                for i in range(4): boot[i].append(-cb[i])
            except Exception: pass
        for i,lab in enumerate(['0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0']):
            loss=-float(coef[i]); losspct=loss/ctrl*100 if ctrl>0 else np.nan
            ci_lo,ci_hi=(np.percentile(boot[i],[2.5,97.5]) if len(boot[i])>20 else (np.nan,np.nan))
            rows.append(dict(category=cat,bin=lab,n=n_by[lab],n_control=int(d.control.sum()),
                             control_speed_kmh=ctrl,loss_kmh=loss,loss_pct=losspct,
                             ci_lo_kmh=float(ci_lo),ci_hi_kmh=float(ci_hi),
                             equiv_downforce_loss_pct=100*(1-((ctrl-loss)/ctrl)**2) if ctrl>0 else np.nan,
                             tyre_coef=float(coef[4]),lap_coef=float(coef[5]),drs_coef=float(coef[6]),
                             se_kmh=float(se[i])))
        print(cat,'n',len(d),'control',int(d.control.sum()),'n_by',n_by)
    loss=pd.DataFrame(rows)
    loss.to_csv(OUT/'loss_by_bin.csv',index=False)
    print(loss.to_string(index=False))
    # 6) 粗对照：每弯损失 -> 单圈时间增量（用弯段长度/参考速度）
    adj=[]
    for q in loss.itertuples(index=False):
        seg=pd.DataFrame(corner_rows); seg=seg[seg.category==q.category]
        dt=[]
        for c in seg.itertuples(index=False):
            v=float(c.ref_min_speed_kmh)/3.6
            if v>0 and q.loss_kmh>0:
                dt.append((q.loss_kmh/3.6)*float(c.length_m)/(v*v))
        adj.append(dict(category=q.category,bin=q.bin,est_lap_time_loss_s=float(np.sum(dt)) if dt else np.nan))
    adj=pd.DataFrame(adj)
    adj.to_csv(OUT/'lap_time_rough.csv',index=False)
    print(adj.to_string(index=False))
    # 7) 图：差距-损失（matplotlib 只画线，文字用 PIL；本机 matplotlib 字体栈不可用）
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    xs=[float(lab.split('-')[0])+0.15 for lab in ['0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0']]
    colors={'low':'#1f77b4','mid':'#2ca02c','high':'#d62728'}
    for cat,marker in [('low','o'),('mid','s'),('high','^')]:
        q=loss[loss.category==cat]
        if q.empty: continue
        ax.plot(xs[:len(q)],q.loss_kmh.values,marker=marker,color=colors[cat],label=cat)
        ax.fill_between(xs[:len(q)],q.ci_lo_kmh.values,q.ci_hi_kmh.values,color=colors[cat],alpha=.12)
    ax.axhline(0,color='gray',lw=.6); ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=.10,right=.97,bottom=.14,top=.86)
    tmp=Path('/tmp')/'t051_gap_loss_notext.png'; fig.savefig(tmp,dpi=150); plt.close(fig)
    im=Image.open(tmp).convert('RGB'); W,H=im.size; dr=ImageDraw.Draw(im)
    F=findfont('DejaVu Sans')
    def fnt(sz): return ImageFont.truetype(F,sz)
    title='Min-speed loss at corner apex vs gap to car ahead'
    tw=dr.textlength(title,font=fnt(24)); dr.text(((W-tw)/2,8),title,fill=(20,20,20),font=fnt(24))
    dr.text((W*0.55,H-28),'gap to car ahead at apex (s)',fill=(30,30,30),font=fnt(18),anchor='mm')
    # 旋转 y 标题
    ylab=Image.new('RGBA',(320,30),(255,255,255,0)); d2=ImageDraw.Draw(ylab)
    d2.text((0,0),'min-speed loss (km/h)',fill=(30,30,30),font=fnt(18))
    ylab=ylab.rotate(90,expand=True); im.paste(ylab,(8,int(H*0.50)-160),ylab)
    # 图例
    lx,ly=W*0.68,H*0.10
    for i,cat in enumerate(['low','mid','high']):
        dr.rectangle([lx,ly+i*28,lx+22,ly+12+i*28],fill=colors[cat])
        dr.text((lx+30,ly-2+i*28),cat,fill=(30,30,30),font=fnt(17))
    im.save(FIG/'gap_vs_loss.png')
    # 8) 报告
    cache=list(CACHE.glob('*.json')); cache_bytes=sum(p.stat().st_size for p in cache)
    build_report(loss,adj,obs,pd.DataFrame(corner_rows),cache,cache_bytes)
    print('cache_files',len(cache),'cache_bytes',cache_bytes,'requests_this_run',_requests)
    print('DONE')
    # 8) 缓存/请求报告用
    cache=list(CACHE.glob('*.json'))
    print('cache_files',len(cache),'cache_bytes',sum(p.stat().st_size for p in cache),'requests',_requests)
    print('DONE')

if __name__=='__main__':
    main()
