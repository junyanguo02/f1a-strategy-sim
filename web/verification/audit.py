"""From web/: ../.venv/bin/python verification/audit.py. No network or extra packages."""
import json,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy import stats,special
root=Path.cwd()
tmp=root/'.qa039'
tmp.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'src/sim'))
from reference_repeats import RaceConfig,TyreParams,Strategy,common_draws,simulate
subprocess.run(['node_modules/.bin/esbuild','verification/sample.ts','--platform=node','--format=esm','--bundle','--outfile=.qa039/sample.mjs'],check=True)
ts_job=subprocess.Popen(['node','.qa039/sample.mjs'])
out={'seeds':{'Python':'490000–490099','TS':'390000–390099'},'tost':[],'distributions':[],'lap_rates':{},'hashes':{},'pit_diagnostic':[]}
for rnd in [1,3,10]:
    folder=root/f'public/data/races/2024-{rnd}'
    hashes=[hashlib.sha256((folder/(n+'.json')).read_bytes()).hexdigest() for n in ['sim_inputs_prerace','strategies']]
    doc=json.loads((folder/'sim_inputs_prerace.json').read_text())
    candidates=json.loads((folder/'strategies.json').read_text())['candidates'][:3]
    fcy=doc['fcy'];loss=doc['pit_loss'];N=doc['total_laps']
    race=RaceConfig(race_id=doc['race_id'],total_laps=N,base_lap_s=doc['base_lap_time_s'],pit_normal=np.array(loss['green']),
        pit_fcy={(t,int(k)):np.array(v) for t,p in loss['fcy'].items() for k,v in p.items()},
        **{k:{t:e[k] for t,e in fcy.items()} for k in ('hazard','lap1_prob','durations','lap_mult')})
    tyre=TyreParams(**{k:np.asarray(v) for k,v in doc['tyre_posterior'].items() if k!='n_draws'})
    plans=[Strategy(c['label'],c['compounds'][0],list(zip(c['pit_laps'],c['compounds'][1:]))) for c in candidates]
    py=[];pit_means=[]; samples={k:[] for k in ('noise','nu','stint','SC','VSC')}; starts=np.zeros((2,N),int)
    for k in range(100):
        d=common_draws(race,tyre,5000,490000+k,max_stints=4)
        results=[simulate(race,tyre,p,draws=d) for p in plans]
        a,b,c=[r['total'] for r in results]
        pit_means.append([(r['pit_loss']-results[0]['pit_loss']).mean() for r in results])
        py.append([0,(b-a).mean(),(c-a).mean()])
        if k<24:
            samples['noise'].extend(d['t_noise'][:,0]);samples['nu'].extend(tyre.nu[d['d']])
            samples['stint'].extend(d['u_stint'][:,0]*tyre.sd_stint[d['d']])
        for j,name in enumerate(['VSC','SC'],1):
            sel=d['status']==j
            starts[j-1]+=((d['phase']==1)&sel).sum(0)
            # Include right-censored final spells identically on both implementations.
            ends=np.column_stack([d['phase'][:,1:]<=1,np.ones(5000,bool)])
            take=120000-len(samples[name])
            if take:samples[name].extend(d['phase'][sel&ends][:take])
    if ts_job.wait()!=0: raise RuntimeError('TS sampler failed')
    ts=json.loads((tmp/f'ts{rnd}.json').read_text())
    assert ts['hashes']==hashes
    out['hashes'][rnd]=hashes
    for i,plan in enumerate(plans):
        a,b=np.asarray(ts['means'])[:,i],np.asarray(py)[:,i];diff=a.mean()-b.mean()
        va,vb=a.var(ddof=1)/100,b.var(ddof=1)/100
        if i:
            for engine,tot,loss in [('TS',a,np.array(ts['pitMeans'])[:,i]),('Python',b,np.array(pit_means)[:,i])]:
                out['pit_diagnostic'].append(dict(race=rnd,label=plan.name,engine=engine,
                    total_sd=tot.std(ddof=1),pit_sd=loss.std(ddof=1),remaining_sd=(tot-loss).std(ddof=1),
                    pit_mean=loss.mean(),remaining_mean=(tot-loss).mean()))
        if va+vb:
            se=np.sqrt(va+vb);df=(va+vb)**2/((va**2+vb**2)/99)
            lo,hi=diff+stats.t.ppf([.025,.975],df)*se
            p=max(stats.t.sf((diff+.3)/se,df),stats.t.cdf((diff-.3)/se,df))
        else:lo=hi=p=0.
        out['tost'].append(dict(race=rnd,label=plan.name,ts=a.mean(),py=b.mean(),diff=diff,lo=lo,hi=hi,p=p,passed=bool(p<.05)))
    rows=out['distributions']
    def record(key,a,b,D,mean,var,n):
        rows.append(dict(race=rnd,quantity=key,n=n,ts_mean=np.mean(a),ref_mean=mean,ts_var=np.var(a),ref_var=var,KS=D))
    for key in ['uniform','normal','noise','stint','SC','VSC']:
        a=np.asarray(ts['samples'][key])
        if key in ['uniform','normal']:
            D=stats.kstest(a,'uniform' if key=='uniform' else 'norm').statistic
            m,v=(.5,1/12) if key=='uniform' else (0,1)
        else:
            b=np.asarray(samples[key]);m,v=b.mean(),b.var();D=stats.ks_2samp(a,b).statistic
        assert len(a)>=100000 and (key in ('uniform','normal') or len(b)>=100000)
        record(key,a,None,D,m,v,len(a))
    u=special.stdtr(np.asarray(ts['samples']['nu']),np.asarray(ts['samples']['noise']))
    record('t CDF conditional on nu',u,None,stats.kstest(u,'uniform').statistic,.5,1/12,len(u))
    for j,key in enumerate(['VSC','SC']):
        prob=np.asarray(ts['starts'][j])/500000; ref=starts[j]/500000
        out['lap_rates'][f'{rnd}-{key}']={'TS':prob.tolist(),'Python':ref.tolist()}
        # On each lap the binary start flag has KS = |p_TS-p_Py|.
        rows.append(dict(race=rnd,quantity=key+' starts/lap',n=500000,ts_mean=prob.mean(),ref_mean=ref.mean(),
            ts_var=(prob*(1-prob)).mean(),ref_var=(ref*(1-ref)).mean(),KS=np.max(abs(prob-ref))))
    print('round',rnd,out['tost'][-2:],flush=True)
(root/'verification/random-audit.json').write_text(json.dumps(out,indent=2)+'\n')
