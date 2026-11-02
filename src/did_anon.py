"""AI 붐과 공급망 불투명성: 이중차분 + 사건연구.
단위: 공급사-회계연도 (주요고객 공시가 있는 해). 종속변수: 주요고객 중 익명 비율.
처치: AI 하드웨어 공급망 업종(SIC). 이후: FY2023~ (ChatGPT 2022-11-30 이후 첫 완전 회계연도).
기업·연도 고정효과, 기업 군집 표준오차."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"
AI_SIC = {3674, 3572, 3576, 3577, 3571, 3672, 3679, 3661, 3663, 3612, 3613, 3621, 3585}
POST = int(sys.argv[1]) if len(sys.argv) > 1 else 2023


def panel():
    r = pd.read_parquet(INT / "relations.parquet")
    r = r[r["current"] & r["cls"].isin(["anonymous", "named"])].copy()
    r["fy"] = r["ddate"].astype(str).str[:4].astype(int) - (r["ddate"].astype(str).str[4:6].astype(int) < 7)
    r["cik"] = r["cik"].astype(int)
    p = r.groupby(["cik", "fy"]).agg(anon_share=("cls", lambda s: (s == "anonymous").mean()),
                                     any_anon=("cls", lambda s: float((s == "anonymous").any())),
                                     all_anon=("cls", lambda s: float((s == "anonymous").all())),
                                     n_cust=("member", "nunique"), max_share=("pct", "max"),
                                     sic=("sic", "first")).reset_index()
    f = pd.read_parquet(INT / "fundamentals.parquet")
    f["ln_at"] = np.log(f["at"].where(f["at"] > 0)); f["roa"] = (f["ni"] / f["at"]).clip(-2, 2)
    f["rnd"] = (f["rnd"].fillna(0) / f["rev"]).where(f["rev"] > 0).clip(0, 5)
    p = p.merge(f[["cik", "year", "ln_at", "roa", "rnd"]].rename(columns={"year": "fy"}), on=["cik", "fy"], how="left")
    p["sic"] = pd.to_numeric(p["sic"], errors="coerce")
    p["ai"] = p["sic"].isin(AI_SIC).astype(float)
    p["post"] = (p["fy"] >= POST).astype(float)
    p["ai_post"] = p["ai"] * p["post"]
    return p[(p["fy"] >= 2014) & (p["fy"] <= 2025)]


def did(p, y, controls):
    d = p.dropna(subset=[y] + controls).set_index(["cik", "fy"])
    X = d[["ai_post"] + controls]
    m = PanelOLS(d[y], X, entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    return m.params["ai_post"], m.tstats["ai_post"], int(m.nobs), d.index.get_level_values(0).nunique()


def main():
    p = panel()
    print(f"패널: {len(p)} 공급사-연도, 공급사 {p['cik'].nunique()}, AI 업종 공급사 {p.loc[p.ai == 1, 'cik'].nunique()}")
    print("\n평균 익명 비율 (AI 업종 vs 기타):")
    print(p.groupby(["fy", "ai"])["anon_share"].mean().unstack().round(3).rename(columns={0.0: "기타", 1.0: "AI 하드웨어"}).to_string())
    rows = []
    for y in ["anon_share", "any_anon", "all_anon"]:
        for ctrl in [[], ["ln_at", "roa", "rnd", "n_cust", "max_share"]]:
            b, t, n, nf = did(p, y, ctrl)
            rows.append({"outcome": y, "controls": "yes" if ctrl else "no", "AI×Post": round(b, 4), "t": round(t, 2), "N": n, "firms": nf})
    tab = pd.DataFrame(rows)
    print(f"\n== 이중차분 (Post = FY≥{POST}, 기업·연도 FE, 기업 군집) ==\n", tab.to_string(index=False))
    # 사건연구: AI × 연도 (기준 FY2021)
    d = p.dropna(subset=["anon_share"]).copy()
    yrs = [y for y in range(2014, 2026) if y != 2021]
    for y in yrs:
        d[f"ai_{y}"] = d["ai"] * (d["fy"] == y)
    d = d.set_index(["cik", "fy"])
    m = PanelOLS(d["anon_share"], d[[f"ai_{y}" for y in yrs]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    es = pd.DataFrame({"year": yrs, "coef": [m.params[f"ai_{y}"] for y in yrs], "se": [m.std_errors[f"ai_{y}"] for y in yrs]})
    es = pd.concat([es, pd.DataFrame({"year": [2021], "coef": [0.0], "se": [0.0]})]).sort_values("year")
    print("\n== 사건연구 (AI × 연도, 기준 FY2021) ==\n", es.round(3).to_string(index=False))
    tab.to_csv(RES / f"did_anon_post{POST}.csv", index=False)
    es.to_csv(RES / "event_study_anon.csv", index=False)
    p.to_parquet(INT / "did_panel.parquet")


if __name__ == "__main__":
    main()
