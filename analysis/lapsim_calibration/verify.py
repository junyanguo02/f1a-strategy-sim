"""Read exported results and independently reconstruct fit and fuel summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from calibrate import HERE, GEOM, ROOT, POST, CarParams, solve_lap


def verify(out, other=None):
    fits=json.loads((out/"fit.json").read_text())
    meta=json.loads((GEOM/"tracks.json").read_text())["tracks"]
    m,f,p,r=[pd.read_csv(out/(name+".csv")) for name in ["metrics","fuel","profiles","regions"]]
    assert len(m)==len(f)==10 and len(r)==30
    model_laps=pd.read_parquet(ROOT/"data/processed/model_laps.parquet",columns=["year","race_id"])
    mapping=list(dict.fromkeys(model_laps[model_laps.year.eq(2024)].race_id))
    with np.load(POST) as archive:
        gamma=archive["gamma"]
    print("mode         circuit       dt(s)   RMSE(km/h)   fuel vs gamma (s/lap)")
    for mode,fit in fits.items():
        objective=0.
        for t in meta:
            rid=t["race_id"]
            def select(frame):
                return frame[(frame["mode"]==mode)&(frame.race==rid)]
            a,b=select(m).squeeze(),select(f).squeeze()
            curve,regions=select(p),select(r)
            raw=pd.read_parquet(GEOM/(rid+".parquet"))
            np.testing.assert_allclose(curve.ref,raw.speed_ref_kmh_mean)
            np.testing.assert_allclose(curve.low,raw.speed_ref_kmh_min)
            np.testing.assert_allclose(curve.high,raw.speed_ref_kmh_max)
            np.testing.assert_allclose(curve.s,raw.s_m)
            params=fit["params"]
            def simulate(mass):
                car=CarParams(mass_kg=mass,power_kW=params["power_kW"],cda_m2=params["cda_m2"],
                              cla_m2=params["cla_m2"],brake_max_mps2=params["brake_max_mps2"],
                              mu=params["mu"] if mode=="shared" else params["mu_"+rid])
                return solve_lap(raw.s_m,raw.curvature_1pm,car)
            mass=798+107.5*np.mean([(t["race_laps"]-q["lap"]+.5)/t["race_laps"] for q in t["selected_laps"]])
            np.testing.assert_allclose(a.mass,mass)
            sim=simulate(mass)
            np.testing.assert_allclose(curve.sim,sim.v*3.6)
            error=curve.sim-curve.ref
            dt=sim.total-t["reference_lap_time_s"]
            np.testing.assert_allclose([a["dt"],a.rmse,a.dmax,a.dmin],
                 [dt,np.mean(error**2)**.5,curve.sim.max()-curve.ref.max(),curve.sim.min()-curve.ref.min()])
            weights=1/(100+((curve.high-curve.low)/2)**2)
            objective+=np.average(error**2,weights=weights)/100+(dt/2)**2
            used=set()
            for region in regions.sort_values("rank").itertuples():
                start=int(region.start_m/5)
                indices={(start+j)%len(curve) for j in range(20)}
                assert used.isdisjoint(indices)
                np.testing.assert_allclose(region.rmse,np.mean(error.to_numpy()[sorted(indices)]**2)**.5)
                used.update(indices)
            slope=(simulate(mass+5).total-simulate(mass-5).total)/10
            expected=slope*np.array([105.,107.5,110.])/t["race_laps"]
            np.testing.assert_allclose([b.low,b.mid,b.high],expected)
            index=mapping.index(rid)
            assert b.gamma_index==index
            draws=gamma[:,index]
            np.testing.assert_allclose([b.gamma,b.q05,b.q95],[draws.mean(),*np.percentile(draws,[5,95])])
            assert b.mid_inside==(b.q05<=b.mid<=b.q95)
            assert b.intervals_overlap==(b.low<=b.q95 and b.high>=b.q05)
            assert (out/"figures"/(rid+".png")).read_bytes().startswith(b"\x89PNG")
            print(f"{mode:10s}   {rid:8s}   {dt:+6.3f}     {a.rmse:5.2f}       {b.mid:.4f} / {b.gamma:.4f}")
        np.testing.assert_allclose(objective,fit["objective"])
    if other:
        def hashes(directory):
            files=sorted(p for p in directory.rglob('*') if p.is_file())
            return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        assert hashes(out)==hashes(other)
        assert (out.parent/"report.md").read_bytes()==(other.parent/"report.md").read_bytes()
        print(f"Repeat: {len(hashes(out))} output files + report.md identical (SHA-256 / byte comparison)")
    print("PASS")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=HERE/"outputs")
    parser.add_argument("--compare",type=Path)
    args=parser.parse_args()
    verify(args.out,args.compare)
