"""재심사 대응: (1) AI 노출 10분위별 DiD (2) 표 5를 본 표본(재무 필수)으로 재계산 + 표본 조정 (3) 5사 집합 상한 LP."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parent))
from bounds import AI_SIC, CUST, THETA, capacity  # noqa: E402
from build_gt import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"


def fy_of(dd):
    s = dd.astype(str)
    return s.str[:4].astype(int) - (s.str[4:6].astype(int) < 7)


def deciles():
    p = pd.read_parquet(INT / "did_panel.parquet")
    e = pd.read_parquet(INT / "item1_exposure.parquet"); e = e[e["status"] == "ok"].copy()
    e["kw"] = e[["ai", "dc", "compute", "infra"]].sum(1)
    e["dec"] = pd.qcut(e["kw"].rank(method="first"), 10, labels=False) + 1
    d = p.merge(e[["cik", "dec", "kw"]], on="cik")
    d["post"] = (d["fy"] >= 2023).astype(float)
    xs = []
    for k in range(2, 11):
        d[f"d{k}"] = (d["dec"] == k) * d["post"]; xs.append(f"d{k}")
    z = d.dropna(subset=["anon_share"]).set_index(["cik", "fy"])
    m = PanelOLS(z["anon_share"], z[xs], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    out = pd.DataFrame({"decile": range(2, 11), "coef": m.params.values, "se": m.std_errors.values, "t": m.tstats.values})
    out = pd.concat([pd.DataFrame({"decile": [1], "coef": [0.0], "se": [0.0], "t": [np.nan]}), out])
    kw_rng = e.groupby("dec")["kw"].agg(["min", "max"]).round(1)
    out = out.merge(kw_rng, left_on="decile", right_index=True)
    # 단조 추세 검정: 분위 번호를 연속 처치로
    d["decpost"] = d["dec"] * d["post"]
    z2 = d.dropna(subset=["anon_share"]).set_index(["cik", "fy"])
    m2 = PanelOLS(z2["anon_share"], z2[["decpost"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    zero = (e["kw"] == 0).mean()
    print("== AI 노출 10분위 × Post (기준: 1분위) ==\n", out.round(3).to_string(index=False))
    print(f"선형 분위 추세: {m2.params.iloc[0]:+.4f} per decile (t={m2.tstats.iloc[0]:.2f}) | 키워드 0 기업 비율 {zero:.1%}")
    out.to_csv(RES / "rev2_decile_did.csv", index=False)
    return out


def table5_main():
    rel = pd.read_parquet(INT / "relations.parquet")
    r = rel[rel["current"] & rel["cls"].isin(["anonymous", "named"])].copy()
    r["fy"] = fy_of(r["ddate"]); r["cik"] = r["cik"].astype(int)
    f = pd.read_parquet(INT / "fundamentals.parquet")[["cik", "year", "at", "rev"]].dropna()
    main = r.merge(f.rename(columns={"year": "fy"}), on=["cik", "fy"])
    main = main[main["fy"].between(2010, 2025)]
    sy_all = main.groupby(["cik", "fy"]).agg(n=("cls", "size"), a=("cls", lambda s: (s == "anonymous").sum())).reset_index()
    print(f"\n본 표본 (재무 필수, FY2010–2025): 공급사-연도 {len(sy_all)}, 링크 {int(sy_all['n'].sum())}")
    sub = load("sub"); k = sub[sub["form"].isin(["10-K", "10-K/A"])].copy(); k["cik"] = k["cik"].astype(int)
    k["fy0"] = fy_of(k["period"].fillna("20000101").astype(str).str[:8])
    first10k = k.groupby("cik")["fy0"].min()
    yrs = sy_all.groupby("cik")["fy"].apply(set)
    def etype(c, y):
        prev = {v for v in yrs[c] if v < y}
        if prev:
            return "Continuing" if (y - 1) in prev else "Re-entrant"
        return "New SEC registrant" if y - first10k.get(c, y) <= 1 else "Existing registrant, first disclosure"
    sy_all["type"] = [etype(c, y) for c, y in zip(sy_all["cik"], sy_all["fy"])]
    sy = sy_all[sy_all["fy"].between(2015, 2025)]
    t5 = sy.groupby("type").agg(supplier_years=("cik", "size"), links=("n", "sum"), anon=("a", "sum"))
    t5["anon_share_%"] = 100 * t5["anon"] / t5["links"]
    print("== 표 5 (본 표본, FY2015–2025) ==\n", t5.round(1).to_string())
    print("합계: 공급사-연도", int(t5["supplier_years"].sum()), "링크", int(t5["links"].sum()),
          "| FY2010–2014 제외분: 공급사-연도", len(sy_all) - len(sy), "링크", int(sy_all["n"].sum() - sy["n"].sum()))
    by = sy.groupby(["fy", "type"]).apply(lambda x: 100 * x["a"].sum() / x["n"].sum()).unstack()
    print(by.loc[[2015, 2017, 2019, 2021, 2023, 2025]].round(1).to_string())
    e25 = sy[(sy["fy"] == 2025) & (sy["type"] != "Continuing")]
    print("2025 비계속 링크 구성:", (e25.groupby("type")["n"].sum() / e25["n"].sum()).round(3).to_dict())
    t5.to_csv(RES / "rev2_table5_main.csv"); by.to_csv(RES / "rev2_table5_main_by_year.csv")


def group_lp():
    """5사 집합 상한: 익명 링크 j 를 고객 h 에 x_jh∈[0,1] 비율로 배정. 링크별 Σ_h x_jh ≤ 1 (배타),
    고객별 Σ_j v_j x_jh ≤ θ·3년 총매입여력 (집계 상한), 연도별 개별 가능성(v_j ≤ θ·연 매입여력) 아니면 x_jh=0. 목적: Σ v_j x_jh 최대."""
    rel = pd.read_parquet(INT / "relations.parquet")
    rr = rel[rel["current"] & (rel["cls"] == "anonymous") & rel["est_cust_rev"].notna()].copy()
    rr["fy"] = fy_of(rr["ddate"]); rr = rr[rr["fy"].between(2023, 2025)]
    e = pd.read_parquet(INT / "item1_exposure.parquet"); e = e[e["status"] == "ok"]; e["kw"] = e[["ai", "dc", "compute", "infra"]].sum(1)
    top = set(e.loc[e["kw"] >= e["kw"].quantile(0.9), "cik"])
    rr = rr[pd.to_numeric(rr["sic"], errors="coerce").isin(AI_SIC) | rr["cik"].astype(int).isin(top)].reset_index(drop=True)
    hyper = [c for c, n in CUST.items() if n in ["Microsoft", "Amazon", "Alphabet", "Meta", "Oracle"]]
    caps = {h: {fy: capacity(h, fy) for fy in [2023, 2024, 2025]} for h in hyper}
    v = rr["est_cust_rev"].to_numpy() / 1e9
    J, H = len(rr), len(hyper)
    feas = np.array([[(rr.loc[j, "est_cust_rev"] <= THETA * caps[h][rr.loc[j, "fy"]]) and int(rr.loc[j, "cik"]) != h for h in hyper] for j in range(J)])
    c = -np.repeat(v, H) * feas.ravel()
    A, b = [], []
    for j in range(J):  # 배타
        row = np.zeros(J * H); row[j * H:(j + 1) * H] = 1; A.append(row); b.append(1)
    for hi, h in enumerate(hyper):  # 고객별 집계 상한
        row = np.zeros(J * H); row[hi::H] = v; A.append(row); b.append(THETA * sum(caps[h].values()) / 1e9)
    bounds = [(0, 1 if f else 0) for f in feas.ravel()]
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    alloc = (res.x.reshape(J, H) * v[:, None]).sum(0)
    print(f"\n== 5사 집합 LP 상한 (θ={THETA}) == 익명 링크 {J}개, 총액 ${v.sum():.1f}B → 배정 가능 최대 ${-res.fun:.1f}B")
    print("  고객별 배정(한 해법):", {CUST[h]: round(a, 1) for h, a in zip(hyper, alloc)})
    pd.DataFrame([{"anonymous_total_$B": v.sum(), "lp_max_assignable_$B": -res.fun}]).to_csv(RES / "rev2_group_lp.csv", index=False)


if __name__ == "__main__":
    deciles(); table5_main(); group_lp()
