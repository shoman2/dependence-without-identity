"""강건성: (1) 위약 처치연도 (2) Fama-French(2001)식 성향 분해 (3) 기업 내 실명→익명 전환율."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"


def main():
    p = pd.read_parquet(INT / "did_panel.parquet")
    e = pd.read_parquet(INT / "item1_exposure.parquet")
    e = e[e["status"] == "ok"].copy()
    e["kw"] = e[["ai", "dc", "compute", "infra"]].sum(1)
    e["ai_top10"] = (e["kw"] >= e["kw"].quantile(0.9)).astype(float)
    d = p.merge(e[["cik", "ai_top10"]], on="cik")
    # (1) 위약: FY≤2021 표본에서 가짜 처치연도
    out = []
    for fake in [2018, 2019, 2020]:
        x = d[d["fy"] <= 2021].copy()
        x["tp"] = x["ai_top10"] * (x["fy"] >= fake)
        for y in ["anon_share", "all_anon"]:
            z = x.dropna(subset=[y]).set_index(["cik", "fy"])
            m = PanelOLS(z[y], z[["tp"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
            out.append({"placebo_post": fake, "outcome": y, "coef": m.params["tp"], "t": m.tstats["tp"], "N": int(m.nobs)})
    pl = pd.DataFrame(out)
    print("== 위약 처치연도 (FY≤2021, AI 상위10%) ==\n", pl.round(3).to_string(index=False))

    # (2) 성향 분해: 기준기간(FY2012–2014) 분수 로짓으로 '특성상 기대 익명비율' → 연도별 실제와 비교
    q = p.dropna(subset=["anon_share", "ln_at", "roa", "rnd"]).copy()
    q["loss"] = (q["roa"] < 0).astype(float)
    q["sic1"] = (q["sic"] // 1000).fillna(0).astype(int).astype(str)
    X = pd.get_dummies(q[["ln_at", "roa", "rnd", "loss", "sic1"]], columns=["sic1"], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    base = q["fy"].between(2012, 2014)
    glm = sm.GLM(q.loc[base, "anon_share"], X[base], family=sm.families.Binomial(), freq_weights=q.loc[base, "n_cust"]).fit()
    q["expected"] = glm.predict(X)
    dec = q.groupby("fy").agg(actual=("anon_share", "mean"), expected=("expected", "mean"), n=("anon_share", "size"))
    dec["propensity_gap"] = dec["actual"] - dec["expected"]
    b = dec.loc[2012:2014, ["actual", "expected"]].mean()
    dec["chg_actual_vs_base"] = dec["actual"] - b["actual"]
    dec["chg_due_to_characteristics"] = dec["expected"] - b["expected"]
    dec["chg_due_to_propensity"] = dec["chg_actual_vs_base"] - dec["chg_due_to_characteristics"]
    print("\n== 성향 분해 (기준 FY2012–2014) ==\n", dec.loc[2012:2025].round(3).to_string())

    # (3) 기업 내 전환: t-1에 실명 고객이 있던 공급사가 t에 모든 고객을 익명 처리한 비율
    s = p.sort_values(["cik", "fy"]).copy()
    s["prev_any_named"] = s.groupby("cik")["all_anon"].shift(1).eq(0) & s.groupby("cik")["fy"].diff().eq(1)
    s["prev_all_anon"] = s.groupby("cik")["all_anon"].shift(1).eq(1) & s.groupby("cik")["fy"].diff().eq(1)
    sw = pd.DataFrame({
        "named_to_all_anon_%": s[s["prev_any_named"]].groupby("fy")["all_anon"].mean() * 100,
        "all_anon_to_named_%": s[s["prev_all_anon"]].groupby("fy")["all_anon"].apply(lambda v: (1 - v.mean()) * 100)})
    print("\n== 기업 내 전환율 ==\n", sw.loc[2013:2025].round(1).to_string())
    pl.to_csv(RES / "placebo.csv", index=False); dec.to_csv(RES / "propensity_decomposition.csv"); sw.to_csv(RES / "switching.csv")


if __name__ == "__main__":
    main()
