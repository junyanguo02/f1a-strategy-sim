"""Matched laps at constant race, driver, compound and tyre age; fuel is an offset."""
import numpy as np
import pandas as pd
from calibrate import ROOT


def estimate(out):
    metrics=pd.read_csv(out/"metrics.csv")
    primary=metrics.query("power == 750 and mode == 'track_mu'").set_index("race")
    df=pd.read_parquet(ROOT/"data/processed/model_laps.parquet")
    traffic=pd.read_parquet(ROOT/"analysis/traffic_features/traffic_features.parquet")
    keys=["year","round","driver","lap"]
    df=df[df.race_id.isin(primary.index)].merge(traffic[keys+["gap_ahead_prev"]],on=keys,how="left",validate="one_to_one")
    df["traffic"]=pd.cut(df.gap_ahead_prev,[-np.inf,1,2,np.inf],right=False,labels=["close","medium","clear"])
    pairs=[]
    for sample in ["matched","same_traffic","clear_air"]:
        data=df
        group=["race_id","driver","compound","tyre_life"]
        if sample=="same_traffic":
            data=df.dropna(subset=["traffic"])
            group=group+["traffic"]
        if sample=="clear_air":
            data=df[df.traffic=="clear"]
        for _,g in data.groupby(group,observed=True):
            g=g.sort_values("lap")
            a,b=g.iloc[0],g.iloc[-1]
            span=b.lap-a.lap
            if span<10 or a.stint==b.stint:
                continue
            pairs.append(dict(sample=sample,race=a.race_id,driver=a.driver,compound=a.compound,
                              age=a.tyre_life,early=a.lap,late=b.lap,span=span,gain=a.lap_time_s-b.lap_time_s))
    pairs=pd.DataFrame(pairs)
    pairs.to_csv(out/"matched_pairs.csv",index=False)
    results=[]
    for (sample,rid),g in pairs.groupby(["sample","race"],sort=True):
        # Slope through zero for differenced times. Positive = later laps faster.
        sums=g.assign(xy=g.span*g.gain,xx=g.span**2).groupby("driver")[["xy","xx"]].sum().to_numpy()
        slope=sums[:,0].sum()/sums[:,1].sum()
        rng=np.random.default_rng(20261119)
        draw=sums[rng.integers(0,len(sums),(4000,len(sums)))].sum(axis=1)
        fuel=primary.loc[rid,"fuel"]
        resid=slope-fuel
        lo,hi=np.quantile(draw[:,0]/draw[:,1]-fuel,[.025,.975])
        gap=primary.loc[rid,"gamma"]-fuel
        results.append(dict(sample=sample,race=rid,n=len(g),drivers=len(sums),total_trend=slope,physical=fuel,
                            residual=resid,lo=lo,hi=hi,posterior_gap=gap,ratio=resid/gap if abs(gap)>1e-9 else np.nan))
    pd.DataFrame(results).to_csv(out/"evolution.csv",index=False)
