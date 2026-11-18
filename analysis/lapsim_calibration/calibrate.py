"""Fit both T050 models at full resolution. Run twice with different --out directories."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1a.lapsim import CarParams, solve_lap, sensitivity

GEOM = ROOT / "analysis/track_geometry/outputs"
POST = ROOT / "models/tyre_v3/outputs31_2024/posterior_core.npz"
FIELDS = ["power_kW", "cda_m2", "cla_m2", "brake_max_mps2", "mu"]
LOW = [450, .50, 1., 10., .8]
HIGH = [1050, 2., 9., 65., 2.8]


def load_tracks():
    tracks = json.loads((GEOM/"tracks.json").read_text())["tracks"]
    for t in tracks:
        frame = pd.read_parquet(GEOM / (t["race_id"]+".parquet"))
        s, k, ref, low, high = (frame[c].to_numpy() for c in
            ["s_m","curvature_1pm","speed_ref_kmh_mean","speed_ref_kmh_min","speed_ref_kmh_max"])
        assert np.isfinite([s,k,ref,low,high]).all() and np.allclose(np.diff(s),5)
        assert np.all(low <= ref + 1e-8) and np.all(ref <= high + 1e-8)
        assert np.isclose(s[-1]-s[0]+5,t["centerline_length_m"])
        mean_lap = np.mean([lap["lap"] for lap in t["selected_laps"]])
        mass = 798 + 107.5*(1-(mean_lap-.5)/t["race_laps"])
        weight = 1/(10**2 + ((high-low)/2)**2)
        t.update(s=s,k=k,ref=ref,low=low,high=high,mass=mass,weight=weight/weight.sum())
    return tracks


def fit(tracks, separate, shared=None):
    count = len(tracks) if separate else 1
    names = FIELDS[:4] + (["mu_"+t["race_id"] for t in tracks] if separate else ["mu"])
    lower, upper = np.r_[LOW[:4], [LOW[4]]*count], np.r_[HIGH[:4], [HIGH[4]]*count]

    @lru_cache(maxsize=256)
    def block(i, power, drag, load, brake, mu):
        t = tracks[i]
        car = CarParams(mass_kg=t["mass"],power_kW=power,cda_m2=drag,cla_m2=load,brake_max_mps2=brake,mu=mu)
        lap = solve_lap(t["s"],t["k"],car)
        return np.r_[np.sqrt(t["weight"])*(lap.v*3.6-t["ref"])/10,
                     (lap.total-t["reference_lap_time_s"])/2]

    def residual(x):
        return np.concatenate([block(i,*x[:4],x[4+i if separate else 4]) for i in range(len(tracks))])

    starts = [[775,1.1,5.,32,1.7],[680,1.3,4.,22,1.95]]
    if separate:
        starts = [shared,starts[1]]
    results = []
    for start in starts:
        x0 = np.r_[start[:4],[start[4]]*count]
        sol = least_squares(residual,x0,bounds=(lower,upper),x_scale=upper-lower,
                            diff_step=1e-4,ftol=2e-6,xtol=2e-6,gtol=1e-6,max_nfev=150)
        if not sol.success:
            raise RuntimeError(sol.message)
        print(names[-1],sol.nfev,2*sol.cost,flush=True)
        results.append(sol)
    best = min(results,key=lambda sol:sol.cost)
    # Local Gauss-Newton geometry. Flat columns get NA, not a spurious zero uncertainty.
    jac = best.jac*(upper-lower)
    norm = np.linalg.norm(jac,axis=0)
    keep = norm > 1e-7
    active = jac[:,keep]/norm[keep]
    cov = np.linalg.pinv(active.T@active,rcond=1e-12)
    corr = np.full((len(names),len(names)),np.nan)
    corr[np.ix_(keep,keep)] = cov / np.sqrt(np.outer(cov.diagonal(),cov.diagonal()))
    summary = dict(params=dict(zip(names,best.x)), objective=float(2*best.cost),
                   start_objectives=[float(2*x.cost) for x in results], bounds=dict(zip(names,zip(lower,upper))),
                   active_bounds=dict(zip(names,best.active_mask.tolist())),
                   jac_column_norm=dict(zip(names,norm)),scaled_singular_values=np.linalg.svd(jac,compute_uv=False).tolist())
    profile = []
    for fixed in (350., 450., 775.):
        fixed_sol = least_squares(lambda free: residual(np.r_[fixed, free]), best.x[1:],
                                  bounds=(lower[1:],upper[1:]),x_scale=(upper-lower)[1:],
                                  diff_step=1e-4,ftol=2e-6,xtol=2e-6,gtol=1e-6,max_nfev=150)
        if not fixed_sol.success:
            raise RuntimeError(fixed_sol.message)
        params = np.r_[fixed, fixed_sol.x]
        mids = []
        for i,t in enumerate(tracks):
            car = CarParams(mass_kg=t["mass"], **dict(zip(FIELDS,[*params[:4],params[4+i if separate else 4]])))
            mids.append(sensitivity(car,(t["s"],t["k"]),"mass_kg",5.)*107.5/t["race_laps"])
        profile.append(dict(power=fixed, objective=float(2*fixed_sol.cost), params=dict(zip(names,params)),
                            fuel_mid=dict(zip([t["race_id"] for t in tracks],mids))))
    summary["power_profile"] = profile
    return best.x, summary, pd.DataFrame(corr,index=names,columns=names)


def posterior():
    table = pd.read_parquet(ROOT/"data/processed/model_laps.parquet",columns=["year","race_id"])
    ids = table.loc[table.year==2024,"race_id"].unique()
    with np.load(POST) as f:
        gamma = f["gamma"]
    assert gamma.shape == (4000,len(ids))
    return {race:(i,draws.mean(),*np.quantile(draws,[.05,.95])) for i,(race,draws) in enumerate(zip(ids,gamma.T))}


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    tracks = load_tracks()
    metrics, fuels, curves, regions, fits = [], [], [], [], {}
    post = posterior()
    shared = None
    for mode in ["shared","track_mu"]:
        x, info, corr = fit(tracks,mode=="track_mu",shared)
        if mode=="shared":
            shared = x
        fits[mode] = info
        corr.to_csv(out/f"correlation_{mode}.csv")
        for i,t in enumerate(tracks):
            rid = t["race_id"]
            car = CarParams(mass_kg=t["mass"],**dict(zip(FIELDS,[*x[:4],x[4+i if mode=="track_mu" else 4]])))
            lap = solve_lap(t["s"],t["k"],car)
            sim = 3.6*lap.v
            error = sim-t["ref"]
            metrics.append(dict(mode=mode,race=rid,mass=car.mass_kg,mu=car.mu,ref_s=t["reference_lap_time_s"],
                                sim_s=lap.total,dt=lap.total-t["reference_lap_time_s"],rmse=np.sqrt(np.mean(error**2)),
                                dmax=sim.max()-t["ref"].max(),dmin=sim.min()-t["ref"].min()))
            curves.append(pd.DataFrame(dict(mode=mode,race=rid,s=t["s"],ref=t["ref"],low=t["low"],high=t["high"],sim=sim)))
            # Three worst disjoint 100 m windows, by local RMS; include the finish seam.
            n=len(sim)
            windows=(np.arange(n)[:,None]+np.arange(20))%n
            scores=np.sqrt(np.mean(error[windows]**2,axis=1))
            used=np.zeros(n,dtype=bool)
            for j in range(3):
                ix=np.argmax(np.where(used[windows].any(axis=1),-1,scores))
                used[windows[ix]]=True
                regions.append(dict(mode=mode,race=rid,rank=j+1,start_m=t["s"][ix],
                                    end_m=(t["s"][ix]+100)%t["centerline_length_m"],rmse=scores[ix]))
            deriv=sensitivity(car,(t["s"],t["k"]),"mass_kg",5.)
            idx,mean,q05,q95=post[rid]
            low,mid,high=deriv*np.array([105,107.5,110])/t["race_laps"]
            fuels.append(dict(mode=mode,race=rid,laps=t["race_laps"],gamma_index=idx,dt_dm=deriv,
                              low=low,mid=mid,high=high,gamma=mean,q05=q05,q95=q95,mid_inside=q05<=mid<=q95,
                              intervals_overlap=low<=q95 and high>=q05))
    (out/"fit.json").write_text(json.dumps(fits,indent=2)+'\n')
    for name,frame in dict(metrics=pd.DataFrame(metrics),fuel=pd.DataFrame(fuels),profiles=pd.concat(curves),
                           regions=pd.DataFrame(regions)).items():
        frame.to_csv(out/f"{name}.csv",index=False)
    from report import make_report
    make_report(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=HERE/"outputs")
    run(parser.parse_args().out)
