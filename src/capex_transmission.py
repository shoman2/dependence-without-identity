"""경제적 결과: 보이지 않는 관계도 실물 충격을 전달하는가.

공급사-분기 패널 (AI 노출 공급사 + AI 하드웨어 업종, companyfacts 확보분).
  g_iq   = 공급사 분기 매출의 전년동기 대비 로그증가율
  HC_q   = 하이퍼스케일러 5사(MSFT·AMZN·GOOGL·META·ORCL) 합산 분기 설비투자의 전년동기 대비 로그증가율 (q, q-1 평균)
  dep    = 전년도 10-K 의 주요고객 매출비중 합 — 익명(anon_dep) / 실명(named_dep) 분리
  g_iq = β1·HC_q×anon_dep + β2·HC_q×named_dep + γ1·anon_dep + γ2·named_dep + 기업FE + 분기FE
분기 FE 가 HC_q 의 주효과를 흡수. 기업 군집 표준오차. β1>0 이면 익명 관계도 AI 설비투자 충격을 실어 나른다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

sys.path.insert(0, str(Path(__file__).parent))
from revealed_network import CAPEX, REV, quarterly, yoy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"
HYPER = [789019, 1018724, 1652044, 1326801, 1341439]


def main():
    caps = pd.DataFrame({c: quarterly(c, CAPEX, ytd=True) for c in HYPER}).loc["2012Q1":"2026Q2"]
    agg = caps.dropna(how="any").sum(1)
    hc = yoy(agg)
    hc = ((hc + hc.shift(1)) / 2).dropna().clip(-1, 1)
    print("하이퍼스케일러 합산 설비투자 yoy (연말 분기):", hc[hc.index.quarter == 4].round(2).to_dict())

    rel = pd.read_parquet(INT / "relations.parquet")
    rel = rel[rel["current"] & rel["cls"].isin(["anonymous", "named"])].copy()
    rel["fy"] = rel["ddate"].astype(str).str[:4].astype(int) - (rel["ddate"].astype(str).str[4:6].astype(int) < 7)
    rel["cik"] = rel["cik"].astype(int)
    dep = rel.groupby(["cik", "fy"]).apply(lambda x: pd.Series({
        "anon_dep": x.loc[x["cls"] == "anonymous", "pct"].sum(), "named_dep": x.loc[x["cls"] == "named", "pct"].sum()})).reset_index()
    dep[["anon_dep", "named_dep"]] = dep[["anon_dep", "named_dep"]].clip(0, 1)

    targets = pd.read_parquet(INT / "quarterly_targets.parquet")
    sup = sorted(set(targets.loc[~targets["hyper"], "cik"]) & set(dep["cik"]))
    rows = []
    for c in sup:
        g = yoy(quarterly(c, REV))
        if len(g) < 8:
            continue
        d = pd.DataFrame({"g": g.clip(-1.5, 1.5)})
        d["cik"] = c
        d["q"] = d.index
        d["fy_prev"] = d.index.year - 1  # 전년도 10-K 의존도 (사전 정보)
        rows.append(d.reset_index(drop=True))
    p = pd.concat(rows).merge(dep.rename(columns={"fy": "fy_prev"}), on=["cik", "fy_prev"], how="inner")
    p["HC"] = p["q"].map(hc)
    p = p.dropna(subset=["HC"])
    p["HC_anon"] = p["HC"] * p["anon_dep"]
    p["HC_named"] = p["HC"] * p["named_dep"]
    p["t"] = p["q"].astype(str)
    print(f"패널: 공급사-분기 {len(p)}, 공급사 {p['cik'].nunique()}, 분기 {p['q'].min()}–{p['q'].max()}")
    out = []
    for label, sub in [("All periods", p), ("FY2016+", p[p["q"].dt.year >= 2016]), ("2020Q1+", p[p["q"] >= pd.Period("2020Q1")])]:
        z = sub.copy()
        z["tq"] = pd.Categorical(z["t"]).codes
        z = z.set_index(["cik", "tq"])
        m = PanelOLS(z["g"], z[["HC_anon", "HC_named", "anon_dep", "named_dep"]], entity_effects=True, time_effects=True).fit(
            cov_type="clustered", cluster_entity=True)
        out.append({"sample": label, "N": int(m.nobs), "firms": z.index.get_level_values(0).nunique(),
                    "HC×anon_dep": m.params["HC_anon"], "t_anon": m.tstats["HC_anon"],
                    "HC×named_dep": m.params["HC_named"], "t_named": m.tstats["HC_named"]})
    tab = pd.DataFrame(out)
    print("\n== 하이퍼스케일러 설비투자 충격의 전달 (기업·분기 FE, 기업 군집) ==\n", tab.round(3).to_string(index=False))
    tab.to_csv(RES / "capex_transmission.csv", index=False)
    p.to_parquet(INT / "capex_panel.parquet")


if __name__ == "__main__":
    main()
