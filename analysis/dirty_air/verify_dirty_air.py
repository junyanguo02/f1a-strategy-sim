#!/usr/bin/env python3
"""T051 独立校验：弯/样本/损失表/请求/图/T048 修复/两次运行。"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OUT=HERE/'outputs'; LOG=HERE/'logs'; CACHE=ROOT/'data/raw/openf1_track'; TRACK=ROOT/'analysis/track_geometry/outputs'
FAIL,PASS=[],[]
def check(n,ok,d=''):
    (PASS if ok else FAIL).append(n); print(('PASS: ' if ok else 'FAIL: ')+n+(f' — {d}' if d else ''))
def main():
    loss=pd.read_csv(OUT/'loss_by_bin.csv'); rough=pd.read_csv(OUT/'lap_time_rough.csv')
    obs=pd.read_csv(OUT/'corner_observations.csv'); corners=pd.read_csv(OUT/'corners.csv')
    check('loss 12 rows',len(loss)==12,str(len(loss)))
    check('corners 74',len(corners)==74,str(len(corners)))
    check('corner categories',corners.category.value_counts().to_dict()=={'low':30,'mid':28,'high':16},str(corners.category.value_counts().to_dict()))
    check('observations 2664',len(obs)==2664,str(len(obs)))
    check('selected 180 laps',obs[['race_id','driver_number','lap']].drop_duplicates().shape[0]==180)
    check('four bins',set(loss.bin)=={'0.3-0.6','0.6-1.0','1.0-1.5','1.5-3.0'})
    check('three categories',set(loss.category)=={'low','mid','high'})
    for c in ['loss_kmh','loss_pct','ci_lo_kmh','ci_hi_kmh','control_speed_kmh',
              'equiv_downforce_loss_pct','tyre_coef','lap_coef','drs_coef','se_kmh']:
        check(f'loss finite {c}',np.isfinite(loss[c]).all())
    check('loss sample n >0',(loss.n>0).all())
    check('CI contains point',((loss.ci_lo_kmh<=loss.loss_kmh)&(loss.loss_kmh<=loss.ci_hi_kmh)).all())
    # 等效下压力公式
    calc=100*(1-((loss.control_speed_kmh-loss.loss_kmh)/loss.control_speed_kmh)**2)
    check('equiv downforce formula',np.allclose(calc,loss.equiv_downforce_loss_pct,atol=1e-6))
    # 样本/控制
    check('control speed plausible',((loss.control_speed_kmh>80)&(loss.control_speed_kmh<300)).all())
    check('drs columns',set(['drs','drs_active']).issubset(obs.columns))
    check('control count',int(obs.control.sum())>0,str(int(obs.control.sum())))
    # 独立重算 corner 数
    cnt={}
    for rid in ['2024-1','2024-3','2024-10','2024-13','2024-16']:
        d=pd.read_parquet(TRACK/f'{rid}.parquet'); k=np.abs(d.curvature_1pm.to_numpy()); s=d.s_m.to_numpy()
        m=k>0.0025; runs=[]; i=0
        while i<len(m):
            if not m[i]: i+=1; continue
            j=i
            while j+1<len(m) and m[j+1]: j+=1
            runs.append((i,j)); i=j+1
        merged=[]
        for a,b in runs:
            if merged and s[a]-s[merged[-1][1]]<10: merged[-1]=(merged[-1][0],b)
            else: merged.append((a,b))
        cnt[rid]=sum(1 for a,b in merged if s[b]-s[a]>=30)
    check('corner total 74',sum(cnt.values())==74,str(cnt))
    # 请求/缓存
    ds=json.loads((LOG/'download_summary.json').read_text())
    check('estimated 356 <400',ds['estimated_before_download']==356<400)
    check('actual new requests 356',ds['actual_new_requests']==356)
    files=list(CACHE.glob('*.json'))
    check('cache 436 files',len(files)==436,str(len(files)))
    check('cache bytes',sum(p.stat().st_size for p in files)==ds['cache_bytes_after_t051'])
    # 图/报告/修复图
    check('gap figure',(OUT/'figures/gap_vs_loss.png').stat().st_size>1000)
    fixed=list((HERE/'figures').glob('t048_fixed_*.png'))
    check('5 T048 fixed figures',len(fixed)==5,str(len(fixed)))
    rep=(HERE/'report.md').read_text()
    for n in ['T051','固定效应','等效下压力','0.3 s','这些数据不能说明什么','t048_fixed_2024-1.png','356']:
        check(f'report contains {n}',n in rep)
    # 两次运行
    r1,r2=LOG/'run1.sha256',LOG/'run2.sha256'
    check('two run logs',r1.exists() and r2.exists())
    if r1.exists() and r2.exists():
        check('two runs identical',r1.read_text().strip()==r2.read_text().strip())
        cur=[]
        for p in sorted(OUT.glob('*.csv')): cur.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
        for p in [HERE/'report.md']+sorted((OUT/'figures').glob('*.png')): cur.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
        for p in sorted((HERE/'figures').glob('t048_fixed_*.png')): cur.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
        check('current matches run2','\n'.join(cur)==r2.read_text().strip(),f'{len(cur)} files')
        check('two_runs marker',(LOG/'two_runs.txt').exists() and 'TWO RUNS IDENTICAL' in (LOG/'two_runs.txt').read_text())
    print(); print(f'TOTAL: {len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL: print(' -',f)
        sys.exit(1)
    print('ALL CHECKS PASSED')
if __name__=='__main__': main()
