"""리뷰 C1: AI 처치 강화 — 사전 균형표, 처치군 연도별 관측 수, 처치 이전 계수 결합검정, 산업×연도 FE, 매칭 DiD."""
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"


def load():
    p = pd.read_parquet(INT / "did_panel.parquet")
    e = pd.read_parquet(INT / "item1_exposure.parquet")
    e = e[e["status"] == "ok"].copy()
    e["kw"] = e[["ai", "dc", "compute", "infra"]].sum(1)
    e["top"] = (e["kw"] >= e["kw"].quantile(0.9)).astype(float)
    d = p.merge(e[["cik", "top", "kw"]], on="cik")
    d["sic2"] = (d["sic"] // 100).fillna(0).astype(int)
    return d


def fe(d, y, xs, sic_year=False):
    z = d.dropna(subset=[y] + xs).copy()
    if sic_year:  # 산업×연도 효과: 연도별 SIC2 평균을 뺀 within 변환 후 기업 FE
        for v in [y] + xs:
            z[v] = z[v] - z.groupby(["sic2", "fy"])[v].transform("mean")
        z = z.set_index(["cik", "fy"])
        return PanelOLS(z[y], z[xs], entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
    z = z.set_index(["cik", "fy"])
    return PanelOLS(z[y], z[xs], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)


def main():
    d = load()
    # (a) 사전 균형 (FY2019–2021 평균)
    pre = d[d["fy"].between(2019, 2021)].groupby("cik").agg(top=("top", "first"), anon_share=("anon_share", "mean"), ln_at=("ln_at", "mean"),
                                                            roa=("roa", "mean"), rnd=("rnd", "mean"), n_cust=("n_cust", "mean"), max_share=("max_share", "mean"))
    bal = []
    for v in ["anon_share", "ln_at", "roa", "rnd", "n_cust", "max_share"]:
        a, b = pre.loc[pre["top"] == 1, v].dropna(), pre.loc[pre["top"] == 0, v].dropna()
        a, b = a.clip(*a.quantile([.01, .99])), b.clip(*b.quantile([.01, .99]))
        t = stats.ttest_ind(a, b, equal_var=False)
        bal.append({"variable": v, "treated_mean": a.mean(), "control_mean": b.mean(), "diff": a.mean() - b.mean(), "t": t.statistic,
                    "std_diff": (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)})
    bal = pd.DataFrame(bal)
    print("== 사전 균형 (FY2019–21 평균, 처치 vs 비교) ==  N treated", int((pre.top == 1).sum()), "control", int((pre.top == 0).sum()))
    print(bal.round(3).to_string(index=False))
    # (b) 연도별 처치군 관측 수
    cnt = d.groupby(["fy", "top"]).size().unstack().rename(columns={0.0: "control", 1.0: "treated"})
    print("\n연도별 관측 수:\n", cnt.loc[2014:2025].T.to_string())
    # (c) 사건연구 + 처치 이전 결합검정
    yrs = [y for y in range(2014, 2026) if y != 2021]
    for y in yrs:
        d[f"t_{y}"] = d["top"] * (d["fy"] == y)
    m = fe(d, "anon_share", [f"t_{y}" for y in yrs])
    pre_terms = [f"t_{y}" for y in yrs if y < 2021]
    w = m.wald_test(formula=" = ".join(pre_terms) + " = 0" if False else None, restrictions=None) if False else None
    R = np.zeros((len(pre_terms), len(yrs)))
    for i, tname in enumerate(pre_terms):
        R[i, yrs.index(int(tname[2:]))] = 1
    b = m.params.values; V = m.cov.values
    stat = float(b @ R.T @ np.linalg.inv(R @ V @ R.T) @ R @ b)
    pval = 1 - stats.chi2.cdf(stat, len(pre_terms))
    es = pd.DataFrame({"year": yrs, "coef": m.params.values, "ci_low": m.params.values - 1.96 * m.std_errors.values,
                       "ci_high": m.params.values + 1.96 * m.std_errors.values})
    print(f"\n처치 이전(2014–2020) 계수 결합 Wald χ²({len(pre_terms)}) = {stat:.2f}, p = {pval:.3f}")
    # (d) 산업×연도 FE, 매칭
    d["tp"] = d["top"] * (d["fy"] >= 2023)
    rows = []
    for lab, kw in [("Supplier + year FE (baseline)", {}), ("Supplier + SIC2×year FE", {"sic_year": True})]:
        for ctrl in [[], ["ln_at", "roa", "rnd", "n_cust", "max_share"]]:
            r = fe(d, "anon_share", ["tp"] + ctrl, **kw)
            rows.append({"spec": lab + (" + controls" if ctrl else ""), "coef": r.params["tp"], "t": r.tstats["tp"], "N": int(r.nobs)})
    # 최근접 매칭: 같은 SIC1, 사전 규모·R&D·익명비중 거리 최소 비교기업 3곳
    pre2 = pre.join(d.drop_duplicates("cik").set_index("cik")["sic2"]).dropna(subset=["ln_at", "rnd", "anon_share"])
    pre2["sic1"] = pre2["sic2"] // 10
    zc = pre2[["ln_at", "rnd", "anon_share"]].apply(lambda s: (s - s.mean()) / s.std())
    keep = set()
    for c in pre2.index[pre2["top"] == 1]:
        pool = pre2[(pre2["top"] == 0) & (pre2["sic1"] == pre2.loc[c, "sic1"])]
        if len(pool) == 0:
            continue
        dist = ((zc.loc[pool.index] - zc.loc[c]) ** 2).sum(1)
        keep |= set(dist.nsmallest(3).index) | {c}
    dm = d[d["cik"].isin(keep)]
    r = fe(dm, "anon_share", ["tp"])
    rows.append({"spec": "Matched sample (3 nearest controls, same SIC1)", "coef": r.params["tp"], "t": r.tstats["tp"], "N": int(r.nobs)})
    tab = pd.DataFrame(rows)
    print("\n== C1 강건성 ==\n", tab.round(3).to_string(index=False))
    bal.to_csv(RES / "rev_c1_balance.csv", index=False); es.to_csv(RES / "rev_c1_eventstudy.csv", index=False)
    tab.to_csv(RES / "rev_c1_did.csv", index=False); cnt.to_csv(RES / "rev_c1_counts.csv")
    pd.DataFrame([{"pretrend_chi2": stat, "df": len(pre_terms), "p": pval}]).to_csv(RES / "rev_c1_pretrend.csv", index=False)


if __name__ == "__main__":
    main()
