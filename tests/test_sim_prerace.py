"""Time-split and policy checks. Outcome poisoning must not change forecast inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from f1a.sim import prerace as pre
from f1a.sim import Strategy, TyreParams

ROOT = Path(__file__).resolve().parents[1]


def test_cutoff_uses_calendar_not_round_order():
    schedule = pd.DataFrame({"year":[2024]*4, "round":[4,1,2,3],
                             "date":pd.to_datetime(["2024-01-01","2024-03-01","2024-03-01","2024-04-01"])})
    samples = schedule[["year","round"]].assign(value=[2,9999,9999,9999])
    assert pre.before(samples, schedule, pd.Timestamp("2024-03-01")).value.tolist() == [2]


def test_prior_draws_keep_joint_rows_and_use_reference_hard(prior_inputs):
    prior = pd.read_parquet(ROOT/"data/processed/model_laps.parquet").query("year==2023")
    with np.load(ROOT/"models/tyre_v3/outputs31_2023/posterior_core.npz") as z:
        take = np.sort(np.random.default_rng(5).choice(len(z["nu"]), 12, replace=False))
        for circuit in ("bahrain", "absent"):
            ty = TyreParams.from_prerace(z, prior, circuit, 57, 12, 5)
            if circuit == "absent":
                np.testing.assert_array_equal(ty.k0[:,0], np.zeros(12))
                np.testing.assert_array_equal(ty.k0[:,1:], z["mu_k0"][take,0])
                np.testing.assert_array_equal(ty.beta,z["mu_beta"][take,0])
                np.testing.assert_array_equal(ty.gamma,z["mu_gamma"][take,0])
            else:
                i = prior.race_id.unique().tolist().index(ty.meta["source_race"])
                for k in ("k0","beta","gamma"):
                    np.testing.assert_array_equal(getattr(ty,k),z[k][take,i])
            for k in ("nu","sd_stint","kappa"):
                np.testing.assert_array_equal(getattr(ty,k),z[k][take])
            np.testing.assert_array_equal(ty.sigma,np.median(z["sigma"][take],axis=1))
            assert ty.rem_mean == 28.5


@pytest.mark.parametrize(("year","rnd","reason"),[(2023,1,"no_prior_season"),(2024,5,"no_base_lap")])
def test_rule_skips(year,rnd,reason,request):
    if year == 2024:
        request.getfixturevalue("prior_inputs")
    with pytest.raises(pre.Unavailable,match=reason):
        pre.build_inputs(ROOT,year,rnd,12,5)


def test_pace_recent_three_and_season_opener():
    rows, calendar = [], []
    for year, offsets in [(2023,[99,1,2,3]),(2024,[9,8,7,1])]:
        for rnd, delta in enumerate(offsets,1):
            date = pd.Timestamp(year, rnd, 1)
            calendar.append((year,rnd,date))
            rows.extend([(year,rnd,"A",100+delta),(year,rnd,"B",100-delta)])
    schedule = pd.DataFrame(calendar,columns=["year","round","date"])
    data = pd.DataFrame(rows,columns=["year","round","driver","lap_time_s"])
    for month,expected in [(1,2),(2,9),(5,7)]:
        past=schedule[schedule.date < pd.Timestamp(2024,month,1)]
        assert pre.driver_paces(data,past,2024,month==1)["A"]==expected


def test_grid_and_opponent_plan_ignore_outcomes():
    opt = pd.DataFrame({"sequence":["M-H","S-M-S"],"stops":[(23,),(12,30)],"mean":[100,102]})
    best=pre.fastest_plan(opt)
    grid=[{"grid":2,"Driver":{"code":"NEW"},"laps":100000,"strategy":"SHS"},
          {"grid":1,"Driver":{"code":"VET"},"laps":1,"strategy":"HHH"}]
    cars,new=pre.cars_from_prerace(grid,{"VET":-.3},best)
    assert [(c.name,c.grid,c.pace) for c in cars]==[("VET",1,-.3),("NEW",2,.5)]
    assert new==["NEW"] and all(c.strategy==Strategy("M-H","MEDIUM",[(23,"HARD")]) for c in cars)
    mutated=[dict(r,position=999,laps=99999,strategy="W") for r in reversed(grid)]
    assert pre.cars_from_prerace(mutated,{"VET":-.3},best)==(cars,new)
    with pytest.raises(pre.Unavailable,match="unresolved_starting_grid"):
        pre.cars_from_prerace([dict(grid[0],grid=0)],{},best)


def test_poison_target_and_all_later_samples(prior_inputs, monkeypatch):
    y,rnd=2024,10
    original=pre.build_inputs(ROOT,y,rnd,16,42)
    cutoff=original.cutoff
    calendar=pre.calendar(ROOT)
    banned=set(calendar.loc[calendar.date>=cutoff,["year","round"]].itertuples(index=False,name=None))
    parquet,csv,load=pd.read_parquet,pd.read_csv,np.load
    poisoned=set()
    def poison(df,path):
        if "year" not in df or "round" not in df:
            return df
        mask=pd.MultiIndex.from_frame(df[["year","round"]]).isin(banned)
        if not mask.any(): return df
        for col in ("lap_time_s","laps_remaining","relative_loss_s","n_laps","start_lap","total_laps"):
            if col in df: df.loc[mask,col]=999999
        if "driver" in df: df.loc[mask,"driver"]="FUTURE_ONLY"
        if "dt_group" in df: df.loc[mask,"dt_group"]="宣布后0-60秒"
        poisoned.add(Path(path).name)
        return df
    monkeypatch.setattr(pd,"read_parquet",lambda path:poison(parquet(path),path))
    def read_csv(path):
        frame=csv(path)
        # Scheduled distance is a declared, pre-start target input.
        return frame if Path(path).name=="scheduled_laps.csv" else poison(frame,path)
    monkeypatch.setattr(pd,"read_csv",read_csv)
    def previous_only(path):
        assert Path(path).parent.name=="outputs31_2023"
        return load(path)
    monkeypatch.setattr(np,"load",previous_only)
    result=pre.build_inputs(ROOT,y,rnd,16,42)
    assert poisoned=={"model_laps.parquet","laps_pit_filled.parquet","pit_loss_relative.parquet",
                      "pit_match_detail.csv","fcy_episodes.parquet","fcy_race_summary.csv"}
    assert result.paces==original.paces and "FUTURE_ONLY" not in result.paces
    old,race=original.race,result.race
    assert race.base_lap_s==old.base_lap_s and race.total_laps==66 and result.tyre.rem_mean==33
    clean=parquet(ROOT/"data/processed/laps_pit_filled.parquet")
    spain=calendar[(calendar.year==2023)&calendar.circuit_id.eq("catalunya")].iloc[0]
    assert race.base_lap_s==clean.loc[(clean.year==2023)&(clean["round"]==spain["round"])&clean.is_clean,"lap_time_s"].median()
    assert result.historical_end < cutoff
    assert race.lap1_prob==old.lap1_prob and race.lap_mult==old.lap_mult
    np.testing.assert_array_equal(race.pit_normal,old.pit_normal)
    for family in ("durations","hazard","pit_fcy"):
        for k,v in getattr(old,family).items():
            np.testing.assert_array_equal(getattr(race,family)[k],v)
    for k in ("beta","gamma","k0","nu","sigma","kappa","sd_stint"):
        np.testing.assert_array_equal(getattr(result.tyre,k),getattr(original.tyre,k))


@pytest.mark.parametrize("local_count", [9,10])
def test_pit_threshold_is_applied_after_cutoff(prior_inputs, local_count,monkeypatch):
    parquet = pd.read_parquet
    def with_pits(path):
        if Path(path).name!="pit_loss_relative.parquet":
            return parquet(path)
        rows = [(2023,1,"bahrain",2.0)] * local_count + [(2023,2,"jeddah",4.0)]
        rows += [(2024,1,"bahrain",9999.0)] * 30
        return pd.DataFrame(rows,columns=["year","round","circuit_id","relative_loss_s"]).assign(relative_usable=True,condition="正常")
    monkeypatch.setattr(pd,"read_parquet",with_pits)
    result=pre.build_inputs(ROOT,2024,1,8,4)
    expected=[2.0]*local_count + ([4.0] if local_count<10 else [])
    np.testing.assert_array_equal(result.race.pit_normal,expected)


def test_candidate_forecasts_never_see_actual_driver_strategies(prior_inputs, monkeypatch):
    from f1a.sim import field
    inputs=pre.build_inputs(ROOT,2024,1,8,4)
    opt=pd.DataFrame({"sequence":["M-H","S-M-S"],"stops":[(22,),(13,34)],"mean":[100.,104.]})
    grid=[{"grid":i+1,"Driver":{"code":name},"actual":"WET","position":100-i}
          for i,name in enumerate("ABCDEFGH")]
    def evaluate(race,tyre,cars,ego,candidates,fp,n_sims,seed):
        assert n_sims==300
        assert all(c.strategy.label()=="M-H @22" for c in cars)
        assert all(s.name!="WET" for s in candidates)
        return pd.DataFrame([dict(strategy=candidates[0].name,label=candidates[0].label(),exp_pos=3.)])
    monkeypatch.setattr(field,"evaluate_ego",evaluate)
    a=pre.predict_field(inputs,grid,opt,{"B"},300,3)
    b=pre.predict_field(inputs,list(reversed(grid)),opt,{"B","C"},300,3)
    pd.testing.assert_frame_equal(a,b.iloc[:1])


def test_sensitivity_changes_one_frozen_parameter_at_a_time():
    from dataclasses import asdict
    from f1a.sim.field import FieldParams
    base=FieldParams()
    cases=pre.sensitivity_params(base)
    for name,(parameter,value) in {"pass_minus":("pass_a",-3.18),"pass_plus":("pass_a",-2.18),
                                  "restart_050":("sc_restart_gap",.5),"restart_120":("sc_restart_gap",1.2)}.items():
        actual=asdict(cases[name])
        assert actual.pop(parameter)==pytest.approx(value)
        expected=asdict(base)
        del expected[parameter]
        assert actual==expected
    assert base.pass_a==-2.68 and base.sc_restart_gap==.81
