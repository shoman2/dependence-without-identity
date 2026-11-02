"""리뷰 C2: (a) 비AI 공급사 위약 (b) 예상치 못한 설비투자 충격(AR(4) 잔차) (c) 익명−실명 계수 차이의 95% CI."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

sys.path.insert(0, str(Path(__file__).parent))
from capex_transmission import HYPER  # noqa: E402
from revealed_network import CAPEX, REV, quarterly, yoy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"


def hc_series():
    g = pd.DataFrame({c: yoy(quarterly(c, CAPEX, ytd=True)) for c in HYPER}).loc["2012Q1":"2026Q2"].clip(-1, 1)
    h = g.mean(1, skipna=True)
    hc = ((h + h.shift(1)) / 2).dropna()
    # 예상치 못한 충격: AR(4) 잔차
    df = pd.DataFrame({"h": h})
    for k in range(1, 5):
        df[f"l{k}"] = df["h"].shift(k)
    df = df.dropna()
    res = sm.OLS(df["h"], sm.add_constant(df[[f"l{k}" for k in range(1, 5)]])).fit().resid
    surprise = ((res + res.shift(1)) / 2).dropna()
    return hc, surprise


def panel(ciks, dep):
    rows = []
    for c in ciks:
        s = yoy(quarterly(c, REV))
        if len(s) < 8:
            continue
        d = pd.DataFrame({"g": s.clip(-1.5, 1.5)}); d["cik"] = c; d["q"] = d.index; d["fy_prev"] = d.index.year - 1
        rows.append(d.reset_index(drop=True))
    return pd.concat(rows).merge(dep.rename(columns={"fy": "fy_prev"}), on=["cik", "fy_prev"])


def est(P, shock, label):
    P = P.copy(); P["S"] = P["q"].map(shock); P = P.dropna(subset=["S"])
    P["S_anon"] = P["S"] * P["anon_dep"]; P["S_named"] = P["S"] * P["named_dep"]
    P["tq"] = pd.Categorical(P["q"].astype(str)).codes
    z = P.set_index(["cik", "tq"])
    m = PanelOLS(z["g"], z[["S_anon", "S_named", "anon_dep", "named_dep"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    b = m.params; V = m.cov
    diff = b["S_anon"] - b["S_named"]; se = np.sqrt(V.loc["S_anon", "S_anon"] + V.loc["S_named", "S_named"] - 2 * V.loc["S_anon", "S_named"])
    return {"sample": label, "N": int(m.nobs), "firms": z.index.get_level_values(0).nunique(), "S×AnonDep": b["S_anon"], "t_anon": m.tstats["S_anon"],
            "S×NamedDep": b["S_named"], "t_named": m.tstats["S_named"], "diff_anon−named": diff, "diff_ci_low": diff - 1.96 * se, "diff_ci_high": diff + 1.96 * se}


def main():
    hc, surprise = hc_series()
    rel = pd.read_parquet(INT / "relations.parquet")
    rel = rel[rel["current"] & rel["cls"].isin(["anonymous", "named"])].copy()
    rel["fy"] = rel["ddate"].astype(str).str[:4].astype(int) - (rel["ddate"].astype(str).str[4:6].astype(int) < 7)
    rel["cik"] = rel["cik"].astype(int)
    dep = rel.groupby(["cik", "fy"]).apply(lambda x: pd.Series({"anon_dep": x.loc[x["cls"] == "anonymous", "pct"].sum(),
                                                                "named_dep": x.loc[x["cls"] == "named", "pct"].sum()})).reset_index()
    dep[["anon_dep", "named_dep"]] = dep[["anon_dep", "named_dep"]].clip(0, 1)
    t = pd.read_parquet(INT / "quarterly_targets.parquet")
    ai = sorted(set(t.loc[~t["hyper"], "cik"]) & set(dep["cik"]))
    nonai = sorted(set(pd.read_parquet(INT / "nonai_targets.parquet")["cik"]) & set(dep["cik"]))
    Pai, Pnon = panel(ai, dep), panel(nonai, dep)
    rows = [est(Pai, hc, "AI hardware/exposed suppliers — capex growth"),
            est(Pai, surprise, "AI hardware/exposed suppliers — capex surprise (AR(4) residual)"),
            est(Pnon, hc, "Placebo: non-AI suppliers — capex growth"),
            est(Pnon, surprise, "Placebo: non-AI suppliers — capex surprise")]
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string(index=False))
    tab.to_csv(RES / "rev_c2_transmission.csv", index=False)


if __name__ == "__main__":
    main()
