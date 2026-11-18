#!/usr/bin/env python3
"""T053 独立校验：v2 观测/损失/差距对比/请求/T048 替换/两次运行。"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OUT=HERE/'outputs_v2'; LOG=HERE/'logs'; CACHE=ROOT/'data/raw/openf1_track'
T48=ROOT/'analysis/track_geometry/outputs/figures'
FAIL,PASS=[],[]
def check(n,ok,d=''):
    (PASS if ok else FAIL).append(n); print(('PASS: ' if ok else 'FAIL: ')+n+(f' — {d}' if d else ''))
def main():
    obs=pd.read_csv(OUT/'corner_observations_v2.csv'); cmp=pd.read_csv(OUT/'v2_gap_compare_observations.csv')
    all_loss=pd.read_csv(OUT/'loss_all.csv'); clean=pd.read_csv(OUT/'loss_excl_attack.csv')
    corners=pd.read_csv(OUT/'corners_v2.csv')
    check('obs 5032',len(obs)==5032,str(len(obs)))
    check('340 laps',obs[['race_id','driver_number','lap']].drop_duplicates().shape[0]==340)
    check('8 drivers per race',obs.groupby('race_id').driver_number.nunique().eq(8).all(),str(obs.groupby('race_id').driver_number.nunique().to_dict()))
    check('74 corners',len(corners)==74)
    check('attack flagged',int(obs.attack_defense.sum())==103,str(int(obs.attack_defense.sum())))
    check('attack fields',set(['position_change','drs_next','attack_defense','base']).issubset(obs.columns))
    check('new gap finite',np.isfinite(obs.gap_s).all())
    check('old compare 2664',len(cmp)==2664,str(len(cmp)))
    check('gap compare stats',abs(cmp.diff_s.median()-(-0.0124))<0.01
          and abs(cmp.diff_s.abs().median()-0.221)<0.01)
    for df,name in [(all_loss,'all'),(clean,'excl')]:
        check(f'{name} loss 12 rows',len(df)==12,str(len(df)))
        check(f'{name} finite',np.isfinite(df[['loss_kmh','ci_lo_kmh','ci_hi_kmh','loss_pct','equiv_downforce_loss_pct']]).all().all())
        check(f'{name} CI',((df.ci_lo_kmh<=df.loss_kmh)&(df.loss_kmh<=df.ci_hi_kmh)).all())
        calc=100*(1-((df.control_speed_kmh-df.loss_kmh)/df.control_speed_kmh)**2)
        check(f'{name} equiv formula',np.allclose(calc,df.equiv_downforce_loss_pct,atol=1e-6))
    # 低速最大
    for df,name in [(all_loss,'all'),(clean,'excl')]:
        d=df[df['bin'].isin(['0.3-0.6','0.6-1.0'])].groupby('category').loss_kmh.mean()
        check(f'{name} low largest',d.idxmax()=='low',d.to_dict().__str__())
    # 请求/缓存
    ds=json.loads((LOG/'v2_download_summary.json').read_text())
    check('estimated 323<400',ds['estimated_before_download']==323 and ds['estimated_before_download']<400)
    check('actual 323',ds['actual_new_requests']==323)
    files=list(CACHE.glob('*.json'))
    check('cache 760',len(files)==760,str(len(files)))
    check('cache bytes',sum(p.stat().st_size for p in files)==ds['cache_bytes_after_t053'])
    # T048 替换
    orig=list((T48/'orig').glob('*.png')); new=list(T48.glob('*.png'))
    check('T048 orig backup 5',len(orig)==5,str(len(orig)))
    check('T048 new 5',len(new)==5,str(len(new)))
    for rid in ['2024-1','2024-3','2024-10','2024-13','2024-16']:
        check(f'T048 replaced {rid}',hashlib.sha256((T48/f'{rid}.png').read_bytes()).hexdigest()
              != hashlib.sha256((T48/'orig'/f'{rid}.png').read_bytes()).hexdigest())
    # report
    rep=(HERE/'report_v2.md').read_text()
    for n in ['T053','/intervals','323','进攻/防守','低速弯损失最大','T048','这些数据不能说明什么']:
        check(f'report contains {n}',n in rep)
    # two run
    r1,r2=LOG/'v2_run1.sha256',LOG/'v2_run2.sha256'
    check('two run logs',r1.exists() and r2.exists())
    if r1.exists() and r2.exists():
        check('two runs identical',r1.read_text().strip()==r2.read_text().strip())
        check('two_runs marker',(LOG/'v2_two_runs.txt').exists() and 'TWO RUNS IDENTICAL' in (LOG/'v2_two_runs.txt').read_text())
        cur=[]
        for p in sorted(OUT.glob('*.csv')): cur.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
        for p in [HERE/'report_v2.md']+sorted((OUT/'figures').glob('*.png')): cur.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
        for p in sorted(T48.glob('*.png'))+sorted((T48/'orig').glob('*.png')): cur.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
        check('current matches run2','\n'.join(cur)==r2.read_text().strip(),f'{len(cur)} files')
    print(); print(f'TOTAL: {len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL: print(' -',f)
        sys.exit(1)
    print('ALL CHECKS PASSED')
if __name__=='__main__': main()
