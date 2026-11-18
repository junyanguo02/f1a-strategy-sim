#!/usr/bin/env python3
"""T053: 跟车弯速损失 v2（OpenF1 /intervals 差距 + 进攻/防守标记 + 8 车手/场）。"""
from __future__ import annotations
import json, math, sys, time, urllib.parse, urllib.request
from matplotlib.font_manager import findfont
from pathlib import Path
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import run_dirty_air as t51  # 复用缓存/几何/圈速工具

CACHE=ROOT/'data/raw/openf1_track'; TRACK=ROOT/'analysis/track_geometry/outputs'
OUT=HERE/'outputs_v2'; FIG=OUT/'figures'; LOG=HERE/'logs'
LAPS=ROOT/'data/processed/laps_pit_filled.parquet'
RAW=ROOT/'data/raw/openf1'
API='https://api.openf1.org/v1'; UA='f1a-research/1.0 (non-commercial; local)'
RACES=[('2024-1',9472),('2024-3',9488),('2024-10',9539),('2024-13',9566),('2024-16',9590)]
WIN=25.0; BINS=[(0.3,0.6),(0.6,1.0),(1.0,1.5),(1.5,3.0)]
_interval_requests=0; _last=0.0

def throttle():
    global _last
    w=2.1-(time.time()-_last)
    if w>0: time.sleep(w)
    _last=time.time()

def fetch_json(endpoint, params, name):
    global _interval_requests
    cp=CACHE/name
    if cp.exists() and cp.stat().st_size>2:
        try:
            x=json.loads(cp.read_text())
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
            _interval_requests+=1
            cp.write_bytes(raw); obj=json.loads(raw)
            if not isinstance(obj,list): raise ValueError('not list')
            return obj
        except Exception as e:
            last=e
            if '429' in str(e): print('  429 intervals; sleep 60s',flush=True); time.sleep(60)
            else: time.sleep(1.5*(att+1))
    raise RuntimeError(f'{endpoint} failed: {last}')

def add_drivers(rid, sk, base_drv):
    rnd=int(rid.split('-')[1]); pl=pd.read_parquet(LAPS)
    q=pl[(pl['year']==2024)&(pl['round']==rnd)].copy()
    q=q[(q['lap']>1)&q['lap_time_s'].notna()&q['is_clean'].fillna(False)]
    q=q[(~q['pit_in_filled'].fillna(False))&(~q['pit_out'].fillna(False))]
    q=q[(~q['safety_car'].fillna(False))&(~q['virtual_safety_car'].fillna(False))]
    raw=t51.raw_laps(sk); rl={(int(x['driver_number']),int(x['lap_number'])):x for x in raw if x.get('date_start') and x.get('lap_duration')}
    rows=[]
    for x in q.itertuples(index=False):
        k=(int(x.driver_number),int(x.lap))
        if k not in rl: continue
        rows.append(dict(driver=x.driver,driver_number=int(x.driver_number),lap=int(x.lap),
                         lap_time_s=float(x.lap_time_s),tyre_age=float(x.tyre_life) if pd.notna(x.tyre_life) else 0.0,
                         date_start=rl[k]['date_start'],lap_duration=float(rl[k]['lap_duration'])))
    df=pd.DataFrame(rows)
    counts=df.groupby('driver_number').size(); cand=[d for d in counts[counts>=6].index if d not in base_drv]
    med=df[df.driver_number.isin(cand)].groupby('driver_number').lap_time_s.median().sort_values()
    if len(med)==0: return df.iloc[0:0]
    # 从剩余车手按圈速分位选 4 位（中游优先：0.15/0.4/0.65/0.9）
    picks=[]
    for frac in [0.15,0.4,0.65,0.9]:
        i=min(int(round(frac*(len(med)-1))),len(med)-1); d=int(med.index[i])
        if d not in picks: picks.append(d)
    while len(picks)<min(4,len(med)):
        for d in med.index:
            if int(d) not in picks: picks.append(int(d)); break
    picks=picks[:4]
    out=[]
    for d in picks:
        g=df[df.driver_number==d].sort_values('lap')
        if len(g)>8:
            idx=np.linspace(0,len(g)-1,8).round().astype(int); g=g.iloc[sorted(set(idx))]
        out.append(g)
    return pd.concat(out).sort_values(['driver_number','lap']).reset_index(drop=True) if out else df.iloc[0:0]

def load_intervals(rid,sk):
    fn=f'intervals_session{sk}.json'
    obj=fetch_json('intervals',{'session_key':sk},fn)
    by={}
    for x in obj:
        d=x.get('date'); dn=x.get('driver_number')
        if d is None or dn is None: continue
        t=pd.Timestamp(d).value/1e9
        def num(v):
            try:
                if isinstance(v,str) and 'LAP' in v.upper(): return np.nan
                return float(v)
            except Exception: return np.nan
        by.setdefault(int(dn),[]).append(dict(t=t,interval=num(x.get('interval')),gap=num(x.get('gap_to_leader'))))
    out={}
    for dn,a in by.items():
        a=sorted(a,key=lambda z:z['t']); arr=np.array([[z['t'],z['interval'],z['gap']] for z in a],float)
        out[dn]=arr
    return out

def interp_col(arr,t,col):
    if arr is None or len(arr)<2: return np.nan
    idx=np.arange(len(arr)); vals=arr[:,col]; ok=np.isfinite(vals)
    if ok.sum()<2: return np.nan
    return float(np.interp(t,arr[ok,0],vals[ok]))

def position_at(intervals,t,current_dn):
    gaps=[]
    for dn,a in intervals.items():
        g=interp_col(a,t,2)
        if np.isfinite(g): gaps.append((g,dn))
    if not gaps: return None
    gaps.sort()
    for i,(g,dn) in enumerate(gaps,1):
        if dn==current_dn: return i
    return None

def model(obs, label, path):
    rows=[]; rng=np.random.default_rng(20261119)
    for cat in ['low','mid','high']:
        d=obs[obs.category==cat].copy()
        if len(d)==0: continue
        d['group']=d.race_id+'_'+d.driver+'_'+d.corner_id.astype(str)
        g=d.groupby('group').agg(hc=('control','any'),ht=('control',lambda x:(~x).any()))
        keep=set(g[(g.hc)&(g.ht)].index); d=d[d.group.isin(keep)]
        if len(d)<10: continue
        X=pd.DataFrame({'b1':d.bin=='0.3-0.6','b2':d.bin=='0.6-1.0','b3':d.bin=='1.0-1.5','b4':d.bin=='1.5-3.0'}).astype(float)
        X['tyre_age']=d.tyre_age.values; X['lap_number']=d.lap_number.values; X['drs_active']=d.drs_active.values
        y=d.speed_min_kmh.values; grp=d.group.values
        def within(a):
            a=pd.DataFrame(a).copy(); a['g']=grp; num=a.drop(columns='g')
            return (num-num.groupby(a['g']).transform('mean')).to_numpy(float)
        yd=within(pd.Series(y))[:,0]; Xd=within(X)
        coef=np.linalg.lstsq(Xd,yd,rcond=None)[0]
        e=yd-Xd@coef; XtX=np.linalg.pinv(Xd.T@Xd); meat=np.zeros_like(XtX)
        for gg in np.unique(grp):
            m=grp==gg; u=Xd[m].T@e[m]; meat+=np.outer(u,u)
        V=XtX@meat@XtX; se=np.sqrt(np.maximum(np.diag(V),0))
        ctrl=d[d.control].speed_min_kmh.mean()
        boot={i:[] for i in range(4)}; groups=np.unique(grp)
        for _ in range(300):
            pick=rng.choice(groups,len(groups),replace=True); idx=np.concatenate([np.where(grp==g)[0] for g in pick])
            if len(idx)<20: continue
            try:
                cb=np.linalg.lstsq(Xd[idx],yd[idx],rcond=None)[0]
                for i in range(4): boot[i].append(-cb[i])
            except Exception: pass
        for i,lab in enumerate(['0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0']):
            loss=-float(coef[i]); ci=(np.percentile(boot[i],[2.5,97.5]) if len(boot[i])>20 else (np.nan,np.nan))
            rows.append(dict(model=label,category=cat,bin=lab,n=int((d.bin==lab).sum()),n_control=int(d.control.sum()),
                             control_speed_kmh=ctrl,loss_kmh=loss,loss_pct=loss/ctrl*100,
                             ci_lo_kmh=float(ci[0]),ci_hi_kmh=float(ci[1]),
                             equiv_downforce_loss_pct=100*(1-((ctrl-loss)/ctrl)**2),
                             se_kmh=float(se[i])))
    df=pd.DataFrame(rows); df.to_csv(path,index=False); return df

def main():
    global _interval_requests
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    # base = T051 已选 180 圈
    oldobs=pd.read_csv(HERE/'outputs/corner_observations.csv')
    base=oldobs[['race_id','driver_number','lap']].drop_duplicates()
    print('base laps',len(base))
    # 每场补齐到 8 位车手
    selected={}
    for rid,sk in RACES:
        b=base[base.race_id==rid]
        bd=set(int(x) for x in b.driver_number)
        add=add_drivers(rid,sk,bd)
        # base 行从 T051 selected 还原
        raw=t51.raw_laps(sk); rl={(int(x['driver_number']),int(x['lap_number'])):x for x in raw if x.get('date_start') and x.get('lap_duration')}
        pl=pd.read_parquet(LAPS); pp=pl[(pl['year']==2024)&(pl['round']==int(rid.split('-')[1]))]
        rows=[]
        for x in b.itertuples(index=False):
            k=(int(x.driver_number),int(x.lap))
            if k not in rl: continue
            p=pp[(pp.driver_number==int(x.driver_number))&(pp.lap==int(x.lap))].iloc[0]
            rows.append(dict(driver=p.driver,driver_number=int(x.driver_number),lap=int(x.lap),
                             lap_time_s=float(p.lap_time_s),tyre_age=float(p.tyre_life) if pd.notna(p.tyre_life) else 0.0,
                             date_start=rl[k]['date_start'],lap_duration=float(rl[k]['lap_duration'])))
        selected[rid]=pd.concat([pd.DataFrame(rows),add],ignore_index=True).sort_values(['driver_number','lap']).reset_index(drop=True)
        print(rid,'selected laps',len(selected[rid]),'drivers',selected[rid].driver_number.nunique())
    # 请求估算
    missing=[]
    for rid,sk in RACES:
        for r in selected[rid].itertuples(index=False):
            tag=f'session{sk}_driver{int(r.driver_number)}_lap{int(r.lap)}'
            for ep in ['location','car_data']:
                if not (CACHE/f'{ep}_{tag}.json').exists(): missing.append(f'{ep}_{tag}.json')
        if not (CACHE/f'intervals_session{sk}.json').exists(): missing.append(f'intervals_session{sk}.json')
    print(f'ESTIMATED_REQUESTS={len(missing)} (new sample {sum(len(x) for x in selected.values())} laps; cache excluded)')
    if len(missing)>400:
        (LOG/'v2_download_estimate.txt').write_text(json.dumps(dict(estimated=len(missing),aborted=True),indent=2))
        raise SystemExit(f'ABORT: {len(missing)} > 400')
    (LOG/'v2_download_estimate.txt').write_text(json.dumps(dict(estimated=len(missing),aborted=False),indent=2))
    # 抓 intervals / 缺的 lap 数据
    intervals={rid:load_intervals(rid,sk) for rid,sk in RACES}
    lapdata={}
    for rid,sk in RACES:
        for r in selected[rid].itertuples(index=False):
            st,en=t51.lap_window(r._asdict()); tag=f'session{sk}_driver{int(r.driver_number)}_lap{int(r.lap)}'
            loc=t51.fetch('location',dict(session_key=sk,driver_number=int(r.driver_number),**{'date>=':st,'date<=':en}),f'location_{tag}.json')
            car=t51.fetch('car_data',dict(session_key=sk,driver_number=int(r.driver_number),**{'date>=':st,'date<=':en}),f'car_data_{tag}.json')
            lapdata[(rid,int(r.driver_number),int(r.lap))]=(loc,car)
    print('intervals requests',_interval_requests,'location/car requests',t51._requests)
    # 观测
    obs=[]; corner_rows=[]
    for rid,sk in RACES:
        d,corns=t51.corners(rid)
        for c in corns: corner_rows.append(dict(race_id=rid,**c))
        Lc=float(d.s_m.max()+5); frac=[t51.time_frac(d,c['s_apex']) for c in corns]
        for r in selected[rid].itertuples(index=False):
            loc,car=lapdata.get((rid,int(r.driver_number),int(r.lap)),(None,None))
            if not loc: continue
            t0,x,y,s=t51.loc_arrays(loc)
            if len(t0)<80: continue
            sn=s/max(s[-1],1e-9)*Lc
            car=[q for q in car if q.get('date') and q.get('speed') is not None]; car.sort(key=lambda q:q['date'])
            if len(car)<50: continue
            ct=np.array([pd.Timestamp(q['date']).value/1e9 for q in car],float)
            cdist=np.interp(ct,t0,sn); sp=np.array([float(q['speed']) for q in car],float)
            dr=np.array([float(q.get('drs') or 0) for q in car],float)
            T0=pd.Timestamp(r.date_start).value/1e9
            for ci,c in enumerate(corns):
                T=T0+frac[ci]*float(r.lap_duration)
                gap=interp_col(intervals[rid].get(int(r.driver_number)),T,1)
                if not np.isfinite(gap): continue
                m=(cdist>=c['s_apex']-WIN)&(cdist<=c['s_apex']+WIN)
                if not m.any(): m=(cdist>=c['s_apex']-2*WIN)&(cdist<=c['s_apex']+2*WIN)
                if not m.any(): continue
                spd=float(sp[m].min()); j=int(np.argmin(np.abs(cdist-c['s_apex']))); drs=float(dr[j])
                # 前后 2 个弯名次变化
                pos=[];
                for kk in range(max(0,ci-2),min(len(corns),ci+3)):
                    pos.append(position_at(intervals[rid],T0+frac[kk]*float(r.lap_duration),int(r.driver_number)))
                pos_change=int(len(set(p for p in pos if p is not None))>1)
                # 后 2 秒 DRS 激活
                fut=(ct>=T)&(ct<=T+2.0); drs_next=int(fut.any() and float(np.nanmax(dr[fut]))>=8)
                attack=int(bool(pos_change) or (gap<0.3 and bool(drs_next)))
                obs.append(dict(race_id=rid,driver=r.driver,driver_number=int(r.driver_number),lap=int(r.lap),
                                corner_id=int(c['corner_id']),category=c['category'],
                                ref_min_speed_kmh=c['ref_min_speed_kmh'],s_apex=c['s_apex'],
                                speed_min_kmh=spd,gap_s=float(gap),tyre_age=float(r.tyre_age),
                                lap_number=int(r.lap),drs=drs,drs_active=int(drs>=8),
                                position_change=pos_change,drs_next=drs_next,attack_defense=attack,
                                n_car=int(m.sum())))
    obs=pd.DataFrame(obs)
    obs['bin']=pd.cut(obs.gap_s,[0.3,0.6,1.0,1.5,3.0],labels=['0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0'],right=False)
    obs['control']=obs.gap_s>3.0; obs['base']=obs.set_index(['race_id','driver_number','lap']).index.isin(base.set_index(['race_id','driver_number','lap']).index)
    # 与 T051 旧 gap 对比
    old=oldobs[['race_id','driver','lap','corner_id','gap_s']].rename(columns={'gap_s':'old_gap_s'})
    cmp=obs.merge(old,on=['race_id','driver','lap','corner_id'],how='inner')
    cmp['diff_s']=cmp.gap_s-cmp.old_gap_s
    cmp.to_csv(OUT/'v2_gap_compare_observations.csv',index=False)
    oldsumm=dict(n=len(cmp),mean_diff=float(cmp.diff_s.mean()),median_diff=float(cmp.diff_s.median()),
                 median_abs_diff=float(cmp.diff_s.abs().median()),p5=float(cmp.diff_s.quantile(.05)),
                 p95=float(cmp.diff_s.quantile(.95)),corr=float(cmp.gap_s.corr(cmp.old_gap_s)))
    obs.to_csv(OUT/'corner_observations_v2.csv',index=False)
    pd.DataFrame(corner_rows).to_csv(OUT/'corners_v2.csv',index=False)
    print('observations',len(obs),'attack',int(obs.attack_defense.sum()),'old compare',len(cmp),oldsumm)
    # 模型：all / 剔除 attack
    all_loss=model(obs,'all',OUT/'loss_all.csv')
    clean_obs=obs[obs.attack_defense==0].copy()
    clean_loss=model(clean_obs,'excl_attack',OUT/'loss_excl_attack.csv')
    print(all_loss.to_string(index=False)); print(clean_loss.to_string(index=False))
    # 粗圈时
    def rough(lossdf,path):
        rows=[]
        for q in lossdf.itertuples(index=False):
            seg=pd.DataFrame(corner_rows); seg=seg[seg.category==q.category]; dt=[]
            for c in seg.itertuples(index=False):
                v=float(c.ref_min_speed_kmh)/3.6
                if v>0 and q.loss_kmh>0: dt.append((q.loss_kmh/3.6)*float(c.length_m)/(v*v))
            rows.append(dict(model=q.model,category=q.category,bin=q.bin,est_lap_time_loss_s=float(np.sum(dt)) if dt else np.nan))
        pd.DataFrame(rows).to_csv(path,index=False); return pd.DataFrame(rows)
    ra=rough(all_loss,OUT/'lap_time_rough_all.csv'); rc=rough(clean_loss,OUT/'lap_time_rough_excl.csv')
    # 图：all vs excluding
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw,ImageFont
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    xs=[float(l.split('-')[0])+0.15 for l in ['0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0']]
    colors={('low','all'):'#1f77b4',('mid','all'):'#2ca02c',('high','all'):'#d62728',
            ('low','excl_attack'):'#1f77b4',('mid','excl_attack'):'#2ca02c',('high','excl_attack'):'#d62728'}
    for df,ls,suf in [(all_loss,'-','all'),(clean_loss,'--','excl')]:
        for cat in ['low','mid','high']:
            q=df[df.category==cat]
            if q.empty: continue
            ax.plot(xs[:len(q)],q.loss_kmh.values,ls,marker='o',color=colors[(cat,'all')],label=f'{cat} {suf}')
    ax.axhline(0,color='gray',lw=.6); ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.86); tmp=Path('/tmp')/'t053_notext.png'; fig.savefig(tmp,dpi=150); plt.close(fig)
    im=Image.open(tmp).convert('RGB'); W,H=im.size; dr=ImageDraw.Draw(im); F=findfont('DejaVu Sans')
    def fnt(sz): return ImageFont.truetype(F,sz)
    title='Dirty-air loss: all vs excluding attack/defence'; tw=dr.textlength(title,font=fnt(23)); dr.text(((W-tw)/2,8),title,fill=(20,20,20),font=fnt(23))
    dr.text((W*.55,H-28),'gap to car ahead at apex (s)',fill=(30,30,30),font=fnt(17),anchor='mm')
    yl=Image.new('RGBA',(300,28),(255,255,255,0)); d2=ImageDraw.Draw(yl); d2.text((0,0),'min-speed loss (km/h)',fill=(30,30,30),font=fnt(17)); yl=yl.rotate(90,expand=True); im.paste(yl,(8,int(H*.5)-130),yl)
    lx,ly=W*.64,H*.09
    for i,lab in enumerate(['low all','mid all','high all','low excl','mid excl','high excl']):
        cat=lab.split()[0]; fill={'low':'#1f77b4','mid':'#2ca02c','high':'#d62728'}[cat]
        if 'excl' in lab: dr.line([lx,ly+8+i*23,lx+22,ly+8+i*23],fill=fill,width=3)
        else: dr.rectangle([lx,ly+i*23,lx+22,ly+12+i*23],fill=fill)
        dr.text((lx+30,ly-2+i*23),lab,fill=(30,30,30),font=fnt(14))
    im.save(FIG/'v2_gap_vs_loss.png')
    # 报告
    build_report(obs,cmp,oldsumm,all_loss,clean_loss,ra,rc,corner_rows)
    cache=list(CACHE.glob('*.json'))
    print('cache',len(cache),sum(p.stat().st_size for p in cache),'DONE')

def build_report(obs,cmp,oldsumm,all_loss,clean_loss,ra,rc,corners):
    L=['# T053　跟车弯速损失 v2（/intervals 差距 + 进攻/防守剔除）','',
       '由 `analysis/dirty_air/run_dirty_air_v2.py` 生成；联网仅 api.openf1.org，限速 2.1 s/req，缓存复用。','',
       '## 方法','',
       '- 样本：T051 的 180 圈 + 每场新增 4 位车手（共 8 位/场，新增车手最多 8 圈）；总圈数/观测见下表。',
       '- 差距：OpenF1 `/intervals` 的 `interval`（与前车差距）按时间戳插值到本车到达弯顶时刻；不再用 T051 的圈时序+参考速度估算。',
       '- 进攻/防守：弯顶前后 2 个弯内本车名次变化，或 gap<0.3 s 且弯后 2 s 内 DRS 激活。',
       '- 模型同 T051：车手×弯固定效应，控制胎龄/圈数/DRS，bootstrap 95% CI；分别报告全样本与剔除进攻/防守。','']
    L.append('## 样本与请求')
    L.append('')
    L.append(f'- 观测 {len(obs)} 条；8 位车手/场；进攻/防守标记 {int(obs.attack_defense.sum())} 条（{obs.attack_defense.mean()*100:.1f}%）。')
    ds=json.loads((LOG/'v2_download_summary.json').read_text())
    cache=list(CACHE.glob('*.json'))
    L.append(f'- 下载前估算 **{ds["estimated_before_download"]}** 个请求（≤400）；实际新增 **{ds["actual_new_requests"]}** 个'
             f'（{ds["new_lap_pairs"]} 个新圈对 ×2 + {ds["new_intervals_requests"]} 个 /intervals）；缓存 {len(cache)} 文件 / {sum(p.stat().st_size for p in cache)} 字节。')
    L.append('')
    L.append('## 新旧差距方法对比')
    L.append('')
    L.append(f'- 可配对观测 {oldsumm["n"]} 条；差值(new−old) 均值 {oldsumm["mean_diff"]:+.3f}s、中位 {oldsumm["median_diff"]:+.3f}s、'
             f'|差|中位 {oldsumm["median_abs_diff"]:.3f}s、P5 {oldsumm["p5"]:+.3f}、P95 {oldsumm["p95"]:+.3f}、相关系数 {oldsumm["corr"]:.3f}。')
    L.append('- 旧方法把“最近一次到达同弯的前车”当目标，容易把前一圈/套圈车或非直接前车算进来；新方法用 intervals 的实际 gap。')
    L.append('')
    for name,df in [('全样本',all_loss),('剔除进攻/防守',clean_loss)]:
        L.append(f'## 损失表：{name}'); L.append('')
        L.append('| category | gap bin | n | n_control | control km/h | loss km/h | loss % | 95% CI | equiv downforce % |')
        L.append('|---|---|---:|---:|---:|---:|---:|---|---:|')
        for q in df.itertuples(index=False):
            L.append(f'| {q.category} | {q.bin} | {q.n} | {q.n_control} | {q.control_speed_kmh:.1f} | {q.loss_kmh:+.2f} | {q.loss_pct:+.2f}% | [{q.ci_lo_kmh:+.2f},{q.ci_hi_kmh:+.2f}] | {q.equiv_downforce_loss_pct:+.2f}% |')
        L.append('')
    # 结论
    L.append('## 结论：低速弯损失是否仍最大')
    L.append('')
    def top(df):
        d=df[df['bin'].isin(['0.3-0.6','0.6-1.0'])]
        if d.empty: return None
        m=d.groupby('category').loss_kmh.mean(); return m.idxmax(),m.to_dict()
    a=top(all_loss); c=top(clean_loss)
    L.append(f'- 全样本 0.3–1.0 s 平均损失最大的类别：`{a[0]}`（{a[1]}）。')
    L.append(f'- 剔除进攻/防守后：`{c[0]}`（{c[1]}）。')
    if a[0]==c[0]=='low': L.append('- **低速弯损失最大在两种口径下都成立**；但高速档样本仍少、CI 宽，不能据此定量气动。')
    else: L.append('- **结论对进攻/防守剔除敏感**；不能简单宣称低速弯损失最大。')
    L.append('')
    L.append('## 粗圈时对照')
    L.append('')
    for df,lab in [(ra,'全样本'),(rc,'剔除进攻/防守')]:
        L.append(f'- {lab}：low 0.3–1.0 = ' + ', '.join(f'{x.bin} {x.est_lap_time_loss_s:.2f}s' for x in df[(df.category=='low')&(df.bin.isin(['0.3-0.6','0.6-1.0']))].itertuples()) +
                 '；mid = ' + ', '.join(f'{x.bin} {x.est_lap_time_loss_s:.2f}s' for x in df[(df.category=='mid')&(df.bin.isin(['0.3-0.6','0.6-1.0']))].itertuples()))
    L.append('')
    L.append('## T048 图替换')
    L.append('')
    L.append('- 已把 T048 原图备份到 `analysis/track_geometry/outputs/figures/orig/`，并用带标题、轴标题、色标的版本替换 `figures/<race>.png`。')
    L.append('')
    L.append('## 这些数据不能说明什么')
    L.append('')
    for s in ['intervals 的 interval 仍是官方计时/估计值，不是逐帧视觉距离；插值到弯顶有误差。',
              '进攻/防守标记只覆盖“名次变化/DRS”可见证据，不能识别所有缠斗、让车、轮胎管理。',
              '固定效应仍是描述性，不对应因果气动；低速弯最大可能来自车辆/走线/轮胎差异。',
              '最低速仍是 25 m 窗口采样；高速弯样本少，CI 宽，不能据此定论。',
              '等效下压力使用 v∝sqrt(mu_eff)，把气动/机械/轮胎混在一起。',
              '只覆盖 5 场 2024、最多 8 车手/场，不能外推全赛季/其它赛道。']:
        L.append(f'- {s}')
    L.append('')
    (HERE/'report_v2.md').write_text('\n'.join(L),encoding='utf-8')

if __name__=='__main__': main()
