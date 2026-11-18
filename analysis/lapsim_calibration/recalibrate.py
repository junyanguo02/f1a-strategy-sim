"""T052 fixed-power fits. Outputs are separate from T050."""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from calibrate import HERE, CarParams, load_tracks, posterior, sensitivity, solve_lap

POWERS = (700, 750, 800)


def align(tracks):
    v1 = pd.read_csv(HERE/"outputs/profiles.csv")
    rows = []
    for t in tracks:
        a = v1[(v1.race==t["race_id"]) & (v1["mode"]=="shared")]
        np.testing.assert_allclose(a.s,t["s"])
        np.testing.assert_allclose(a.ref,t["ref"])
        # Positive d compares sim(s) to observed(s+d), advancing the observed trace.
        scores = []
        for cells in range(-12,13):
            rmse = np.sqrt(np.mean((a.sim-np.roll(t["ref"],-cells))**2))
            scores.append((rmse,cells))
            rows.append(dict(race=t["race_id"],delta_m=cells*5,v1_rmse=rmse))
        _,cells = min(scores)
        t["shift"] = cells
        for key in ("ref","low","high","weight"):
            t["aligned_"+key] = np.roll(t[key],-cells)
    return pd.DataFrame(rows)


def fit(tracks,power,separate,seed):
    count = len(tracks) if separate else 1
    names = ["cda_m2","cla_m2","brake_max_mps2"] + (["mu_"+t["race_id"] for t in tracks] if separate else ["mu"])
    lower=np.r_[.50,1.,10.,np.repeat(.8,count)]
    upper=np.r_[3.,9.,65.,np.repeat(2.8,count)]

    @lru_cache(maxsize=256)
    def block(i,cda,cla,brake,mu):
        t=tracks[i]
        car=CarParams(mass_kg=t["mass"],power_kW=power,cda_m2=cda,cla_m2=cla,brake_max_mps2=brake,mu=mu)
        lap=solve_lap(t["s"],t["k"],car)
        return np.r_[np.sqrt(t["aligned_weight"])*(3.6*lap.v-t["aligned_ref"])/10,
                     (lap.total-t["reference_lap_time_s"])/2]

    def residual(x):
        return np.concatenate([block(i,*x[:3],x[3+i if separate else 3]) for i in range(len(tracks))])

    sols=[]
    for start in (seed,[1.5,4.,22.,1.95]):
        sol=least_squares(residual,np.r_[start[:3],np.repeat(start[3],count)],bounds=(lower,upper),
                          x_scale=upper-lower,diff_step=1e-4,ftol=2e-6,xtol=2e-6,gtol=1e-6,max_nfev=150)
        if not sol.success:
            raise RuntimeError(sol.message)
        sols.append(sol)
    best=min(sols,key=lambda s:s.cost)
    jac=best.jac*(upper-lower)
    cov=np.linalg.pinv(jac.T@jac,rcond=1e-12)
    diag=np.sqrt(cov.diagonal())
    corr=np.divide(cov,np.outer(diag,diag),out=np.full_like(cov,np.nan),where=np.outer(diag,diag)>0)
    return best.x, [2*s.cost for s in sols], pd.DataFrame(corr,index=names,columns=names)


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    tracks=load_tracks()
    align(tracks).to_csv(out/"alignment.csv",index=False)
    post=posterior()
    params,metrics,curves,correlations=[],[],[],[]
    for power in POWERS:
        seed=[1.1,5.,32.,1.7]
        for mode in ["shared","track_mu"]:
            x,loss,corr=fit(tracks,power,mode=="track_mu",seed)
            if mode=="shared":
                seed=x
            print(power,mode,min(loss),flush=True)
            for field in corr:
                for field2 in corr:
                    correlations.append(dict(power=power,mode=mode,param=field,other=field2,r=corr.loc[field,field2]))
            params.append(dict(power=power,mode=mode,cda=x[0],cla=x[1],brake=x[2],power_per_cda=power/x[0],
                               objective=min(loss),start1=loss[0],start2=loss[1]))
            for i,t in enumerate(tracks):
                rid=t["race_id"]
                mu=x[3+i if mode=="track_mu" else 3]
                car=CarParams(mass_kg=t["mass"],power_kW=power,cda_m2=x[0],cla_m2=x[1],brake_max_mps2=x[2],mu=mu)
                lap=solve_lap(t["s"],t["k"],car)
                sim=3.6*lap.v
                ref=t["aligned_ref"]
                idx,gamma,q05,q95=post[rid]
                slope=sensitivity(car,(t["s"],t["k"]),"mass_kg",5.)
                low,mid,high=slope*np.array([105,107.5,110])/t["race_laps"]
                metrics.append(dict(power=power,mode=mode,race=rid,delta_m=5*t["shift"],mu=mu,
                                    mass=t["mass"],laps=t["race_laps"],sim_s=lap.total,dt=lap.total-t["reference_lap_time_s"],
                                    rmse=np.sqrt(np.mean((sim-ref)**2)),rmse_unaligned=np.sqrt(np.mean((sim-t["ref"])**2)),
                                    dmax=sim.max()-ref.max(),dmin=sim.min()-ref.min(),dt_dm=slope,
                                    fuel_low=low,fuel=mid,fuel_high=high,gamma_index=idx,gamma=gamma,q05=q05,q95=q95))
                curves.append(pd.DataFrame(dict(power=power,mode=mode,race=rid,s=t["s"],sim=sim,ref_raw=t["ref"],
                                                ref=ref,low=t["aligned_low"],high=t["aligned_high"])))
    for name,df in dict(parameters=pd.DataFrame(params),metrics=pd.DataFrame(metrics),
                        profiles=pd.concat(curves),correlations=pd.DataFrame(correlations)).items():
        df.to_csv(out/(name+".csv"),index=False)
    from evolution import estimate
    estimate(out)
    from report_v2 import render
    render(out)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=HERE/"outputs_v2")
    out=ap.parse_args().out.resolve()
    if out==(HERE/"outputs").resolve():
        ap.error("v1 outputs are reserved")
    run(out)
