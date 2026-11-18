"""F1-A 轮胎衰减分层模型 v3.1（NumPyro）。

v3.1 相对 v3 只改参数化，不改模型含义（T014 发现 SOFT 配方偏移 mu_k0 r_hat 1.6、ess 3，
一条链与其余三条差约 0.15 秒）：
- stint 随机截距在每个“站×配方”格子内中心化（格内训练 stint 的 u 均值为 0）。v3 里配方偏移 k0 与
  该格 stint 截距的均值只靠先验区分，是 SOFT 混合慢的主因；中心化后 k0 唯一承担格均值。
- k0_raw、beta_raw、gamma、alpha、u 全部改为非中心参数化，减轻漏斗。
- max_tree_depth 9 → 10。

---- 以下为 v3 原说明 ----

相对 v2：
1. 按赛季分层：各站的燃油项 gamma、衰减 beta、配方偏移 k0 围绕“该赛季”的均值波动，赛季均值再围绕总体均值。
2. 交通项：上一圈结束时距前车 <1 秒、1–2 秒两个指示变量；时间差缺失单独一个指示变量。
3. stint 随机截距：吸收同一段轮胎内误差的持续偏移（T010：stint 内残差自相关约 0.4）。

y = alpha[race,driver] + k0[race,c] + gamma[race]*rem_c + beta[race,c]*age_c + kappa[c]*age_c^2
    + d1*1[gap<1] + d2*1[1<=gap<2] + d_na*1[gap 缺失] + u[stint] + eps
eps ~ StudentT(nu, 0, sigma[driver])

用法：
    python fit_tyre_v3_1.py --data ../../data/processed/model_laps.parquet \
        --traffic ../../analysis/traffic_features/traffic_features.parquet \
        --years 2023 2024 2025 2026 --out outputs --chains 4 --warmup 1000 --samples 1000
环境变量 CHAIN_METHOD=parallel|sequential（默认 parallel）。
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import gelman_rubin, effective_sample_size

COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]
AGE_REF = 10.0
MIN_LAPS, MIN_STINTS = 40, 3


def prepare(df, tr, holdout_frac, seed):
    df = df.merge(tr[["year", "round", "driver", "lap", "gap_ahead_prev"]], on=["year", "round", "driver", "lap"], how="left")
    df = df.reset_index(drop=True)
    df["season_i"] = pd.factorize(df["year"], sort=True)[0]
    df["race_i"] = pd.factorize(df["race_id"])[0]
    df["rd_i"] = pd.factorize(df["race_id"] + "|" + df["driver"])[0]
    df["drv_i"] = pd.factorize(df["driver"])[0]
    df["st_i"] = pd.factorize(df["stint_uid"].astype(str) + "|" + df["race_id"])[0]
    df["c_i"] = df["compound"].map({c: i for i, c in enumerate(COMPOUNDS)}).astype(int)
    df["y"] = df["lap_time_s"] - df.groupby("race_id")["lap_time_s"].transform("median")
    df["rem_c"] = df["laps_remaining"] - df.groupby("race_id")["laps_remaining"].transform("mean")
    df["age_c"] = df["tyre_life"] - AGE_REF
    g = df["gap_ahead_prev"]
    df["t1"] = (g < 1.0).astype(float)
    df["t2"] = ((g >= 1.0) & (g < 2.0)).astype(float)
    df["tna"] = g.isna().astype(float)
    cell = df.groupby(["race_i", "c_i"]).agg(n=("lap", "size"), s=("st_i", "nunique")).reset_index()
    n_race = int(df.race_i.max()) + 1
    mask = np.zeros((n_race, len(COMPOUNDS)), dtype=bool)
    ok = cell[(cell.n >= MIN_LAPS) & (cell.s >= MIN_STINTS)]
    mask[ok.race_i.values, ok.c_i.values] = True
    race_season = df.drop_duplicates("race_i").sort_values("race_i")["season_i"].values
    rng = np.random.default_rng(seed)
    n_st = df.groupby("rd_i")["st_i"].transform("nunique")
    cand = df.loc[n_st >= 2, "st_i"].unique()
    hold = set(rng.choice(cand, size=int(holdout_frac * df["st_i"].nunique()), replace=False))
    df["holdout"] = df["st_i"].isin(hold)
    st = df.drop_duplicates("st_i").sort_values("st_i")
    st_cell = (st["race_i"].values * len(COMPOUNDS) + st["c_i"].values).astype(int)
    st_train = (~st["st_i"].isin(hold)).values.astype(float)
    meta = dict(st_cell=st_cell, st_train=st_train, n_cell=n_race * len(COMPOUNDS), mask=mask, race_season=race_season, n_race=n_race, n_season=int(df.season_i.max()) + 1,
                n_rd=int(df.rd_i.max()) + 1, n_drv=int(df.drv_i.max()) + 1, n_st=int(df.st_i.max()) + 1)
    return df, meta


def model(d, meta, y=None):
    nC = len(COMPOUNDS); nS = meta["n_season"]; rs = meta["race_season"]; mask = meta["mask"]
    # 总体 → 赛季
    m_g = numpyro.sample("m_gamma", dist.Normal(0.05, 0.05))
    m_b = numpyro.sample("m_beta", dist.Normal(0.05, 0.1).expand([nC]).to_event(1))
    m_k = numpyro.sample("m_k0", dist.Normal(0.0, 1.0).expand([nC - 1]).to_event(1))
    s_season = numpyro.sample("s_season", dist.HalfNormal(0.05))
    with numpyro.plate("season", nS):
        mu_g = numpyro.sample("mu_gamma", dist.Normal(m_g, s_season))
        mu_b = numpyro.sample("mu_beta", dist.Normal(m_b, s_season).to_event(1))
        mu_k = numpyro.sample("mu_k0", dist.Normal(m_k, 0.5).to_event(1))
    tau_g = numpyro.sample("tau_gamma", dist.HalfNormal(0.05))
    tau_b = numpyro.sample("tau_beta", dist.HalfNormal(0.1))
    tau_k = numpyro.sample("tau_k0", dist.HalfNormal(0.5))
    with numpyro.plate("race", meta["n_race"]):
        z_g = numpyro.sample("z_gamma", dist.Normal(0.0, 1.0))
        z_b = numpyro.sample("z_beta", dist.Normal(0.0, 1.0).expand([nC]).to_event(1))
        z_k = numpyro.sample("z_k0", dist.Normal(0.0, 1.0).expand([nC - 1]).to_event(1))
    gamma = numpyro.deterministic("gamma", mu_g[rs] + tau_g * z_g)
    beta_raw = mu_b[rs] + tau_b * z_b
    k0_raw = mu_k[rs] + tau_k * z_k
    beta = numpyro.deterministic("beta", jnp.where(mask, beta_raw, mu_b[rs]))
    k0f = jnp.where(mask[:, 1:], k0_raw, mu_k[rs])
    k0 = numpyro.deterministic("k0", jnp.concatenate([jnp.zeros((meta["n_race"], 1)), k0f], axis=1))
    kappa = numpyro.sample("kappa", dist.Normal(0.0, 0.003).expand([nC]).to_event(1))
    d1 = numpyro.sample("d_gap_lt1", dist.Normal(0.0, 0.5))
    d2 = numpyro.sample("d_gap_1to2", dist.Normal(0.0, 0.5))
    dna = numpyro.sample("d_gap_na", dist.Normal(0.0, 0.5))
    s_a = numpyro.sample("sd_alpha", dist.HalfNormal(2.0))
    s_u = numpyro.sample("sd_stint", dist.HalfNormal(0.5))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    with numpyro.plate("rd", meta["n_rd"]):
        z_a = numpyro.sample("z_alpha", dist.Normal(0.0, 1.0))
    alpha = numpyro.deterministic("alpha", s_a * z_a)
    with numpyro.plate("stint", meta["n_st"]):
        z_u = numpyro.sample("z_u", dist.Normal(0.0, 1.0))
    u0 = s_u * z_u
    cell = jnp.asarray(meta["st_cell"]); w = jnp.asarray(meta["st_train"])
    csum = jax.ops.segment_sum(u0 * w, cell, num_segments=meta["n_cell"])
    ccnt = jax.ops.segment_sum(w, cell, num_segments=meta["n_cell"])
    cmean = jnp.where(ccnt > 0, csum / jnp.maximum(ccnt, 1.0), 0.0)
    # 训练 stint：格内中心化；留出 stint：保持先验抽样（预测新 stint）
    u = numpyro.deterministic("u", u0 - cmean[cell] * w)
    with numpyro.plate("drv", meta["n_drv"]):
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    ri, ci = d["race_i"], d["c_i"]
    mu = (alpha[d["rd_i"]] + k0[ri, ci] + gamma[ri] * d["rem"] + beta[ri, ci] * d["age"] + kappa[ci] * d["age"] ** 2
          + d1 * d["t1"] + d2 * d["t2"] + dna * d["tna"] + u[d["st_i"]] * d["seen"])
    with numpyro.plate("obs", d["age"].shape[0]):
        numpyro.sample("y", dist.StudentT(nu, mu, sigma[d["drv_i"]]), obs=y)


def arrays(d):
    f = lambda c: jnp.array(d[c].values)
    return dict(race_i=f("race_i"), rd_i=f("rd_i"), drv_i=f("drv_i"), st_i=f("st_i"), c_i=f("c_i"),
                age=jnp.array(d.age_c.values, dtype=jnp.float64), rem=jnp.array(d.rem_c.values, dtype=jnp.float64),
                t1=f("t1"), t2=f("t2"), tna=f("tna"), seen=jnp.ones(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--traffic", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    ap.add_argument("--out", default="outputs"); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000); ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--holdout", type=float, default=0.1); ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    numpyro.set_host_device_count(a.chains)
    raw = pd.read_parquet(a.data); raw = raw[raw["year"].isin(a.years)]
    tr = pd.read_parquet(a.traffic)
    df, meta = prepare(raw, tr, a.holdout, a.seed)
    trn, tst = df[~df.holdout], df[df.holdout]
    t0 = time.time()
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=10, init_strategy=numpyro.infer.init_to_median),
                num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
                chain_method=os.environ.get("CHAIN_METHOD", "parallel"), progress_bar=False)
    mcmc.run(jax.random.PRNGKey(a.seed), arrays(trn), meta, y=jnp.array(trn.y.values, dtype=jnp.float64), extra_fields=("diverging",))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    secs = time.time() - t0  # 取样本后再计时（JAX 异步执行，v3 的 seconds 偏小）
    keep = ["m_gamma", "m_beta", "m_k0", "s_season", "mu_gamma", "mu_beta", "mu_k0", "tau_gamma", "tau_beta", "tau_k0",
            "kappa", "d_gap_lt1", "d_gap_1to2", "d_gap_na", "sd_alpha", "sd_stint", "nu"]
    rows = []
    seasons = sorted(df["year"].unique())
    for k in keep:
        v = post[k]; v2 = v.reshape(v.shape[0], v.shape[1], -1)
        rh = np.atleast_1d(gelman_rubin(v2)) if v.shape[0] > 1 else np.full(v2.shape[2], np.nan)
        es = np.atleast_1d(effective_sample_size(v2))
        shape = v.shape[2:]
        for j in range(v2.shape[2]):
            x = v2[:, :, j].ravel()
            if k in ("mu_beta", "mu_k0"):
                si, cj = np.unravel_index(j, shape); cn = COMPOUNDS[cj] if k == "mu_beta" else COMPOUNDS[cj + 1]
                name = f"{k}[{seasons[si]},{cn}]"
            elif k == "mu_gamma":
                name = f"{k}[{seasons[j]}]"
            elif k in ("m_beta", "kappa"):
                name = f"{k}[{COMPOUNDS[j]}]"
            elif k == "m_k0":
                name = f"{k}[{COMPOUNDS[j + 1]}]"
            else:
                name = k
            rows.append(dict(param=name, mean=x.mean(), sd=x.std(), q05=np.quantile(x, .05), q95=np.quantile(x, .95),
                             r_hat=float(rh[j]), ess=float(es[j])))
    summ = pd.DataFrame(rows).set_index("param"); summ.to_csv(f"{a.out}/summary_hyper.csv")
    S = {k: v.reshape(-1, *v.shape[2:]) for k, v in post.items()}
    np.savez_compressed(f"{a.out}/posterior_core.npz", **{k: S[k] for k in ["gamma", "beta", "k0", "kappa", "sigma", "nu", "sd_stint",
                        "d_gap_lt1", "d_gap_1to2", "d_gap_na", "mu_gamma", "mu_beta", "mu_k0"]})
    thin = np.linspace(0, len(S["nu"]) - 1, 200).astype(int)

    def ppc(d, key, new_stints):
        rng = np.random.default_rng(a.seed + key); n = len(d)
        ri, ci = d.race_i.values, d.c_i.values
        pr = np.empty((len(thin), n))
        st_codes, st_inv = np.unique(d.st_i.values, return_inverse=True)
        for j, t in enumerate(thin):
            mu = (S["alpha"][t][d.rd_i.values] + S["k0"][t][ri, ci] + S["gamma"][t][ri] * d.rem_c.values
                  + S["beta"][t][ri, ci] * d.age_c.values + S["kappa"][t][ci] * d.age_c.values ** 2
                  + S["d_gap_lt1"][t] * d.t1.values + S["d_gap_1to2"][t] * d.t2.values + S["d_gap_na"][t] * d.tna.values)
            if new_stints:
                mu = mu + rng.normal(0, S["sd_stint"][t], size=len(st_codes))[st_inv]
            else:
                mu = mu + S["u"][t][d.st_i.values]
            pr[j] = mu + S["sigma"][t][d.drv_i.values] * rng.standard_t(S["nu"][t], size=n)
        yv = d.y.values; lo, hi = np.quantile(pr, [0.05, 0.95], axis=0); med = np.median(pr, axis=0)
        return dict(n=int(n), cover90=float(((yv >= lo) & (yv <= hi)).mean()), cover50=float(((yv >= np.quantile(pr, .25, 0)) & (yv <= np.quantile(pr, .75, 0))).mean()),
                    rmse=float(np.sqrt(((yv - med) ** 2).mean())), mae=float(np.abs(yv - med).mean()))
    res = dict(years=a.years, chains=a.chains, warmup=a.warmup, samples=a.samples, seconds=round(secs, 1),
               chain_method=os.environ.get("CHAIN_METHOD", "parallel"), n_train=int(len(trn)), n_test=int(len(tst)),
               n_races=meta["n_race"], n_stints=meta["n_st"], n_cells_estimated=int(meta["mask"].sum()),
               gap_missing_share=float(df.tna.mean()), divergences=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
               rhat_max_hyper=float(np.nanmax(summ["r_hat"])), ess_min_hyper=float(summ["ess"].min()),
               train=ppc(trn, 1, False), test=ppc(tst, 2, True))
    json.dump(res, open(f"{a.out}/fit_report.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False)); print(summ.round(4).to_string())


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing local input: {exc.filename}. See scripts/download_data.md.") from None
