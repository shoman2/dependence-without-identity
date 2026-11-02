"""리뷰 C3 진입 유형 / C4 공동 배분 상한 / C5 공변량 이동·부트스트랩·평가 흐름 / 정부 고객 민감도 / Oracle 하한 확인."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bounds import AI_SIC, CUST, THETA, capacity  # noqa: E402
from build_gt import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"


def fy_of(dd):
    s = dd.astype(str)
    return s.str[:4].astype(int) - (s.str[4:6].astype(int) < 7)


def main():
    rel = pd.read_parquet(INT / "relations.parquet")
    r = rel[rel["current"] & rel["cls"].isin(["anonymous", "named"])].copy()
    r["fy"] = fy_of(r["ddate"]); r["cik"] = r["cik"].astype(int)
    r = r[r["fy"].between(2012, 2025)]

    # ── C3: 진입 유형. 기준 = 각 CIK 의 첫 10-K (FSN sub, 2009~) vs 첫 주요고객 공시 연도
    sub = load("sub")
    k = sub[sub["form"].isin(["10-K", "10-K/A"])].copy()
    k["cik"] = k["cik"].astype(int); k["fy0"] = fy_of(k["period"].fillna("0").astype(str).str[:8].replace("0", "20000101"))
    first10k = k.groupby("cik")["fy0"].min()
    sy = r.groupby(["cik", "fy"]).agg(n=("cls", "size"), a=("cls", lambda s: (s == "anonymous").sum())).reset_index()
    yrs_by = sy.groupby("cik")["fy"].apply(set)
    def etype(c, y):
        prev = {v for v in yrs_by[c] if v < y}
        if prev:
            return "re-entrant (disclosed before, gap)" if (y - 1) not in prev else "continuing"
        f = first10k.get(c, y)
        return "new SEC registrant (first 10-K within 1 year)" if y - f <= 1 else "existing registrant, first major-customer disclosure"
    sy["type"] = [etype(c, y) for c, y in zip(sy["cik"], sy["fy"])]
    t3 = sy[sy["fy"].between(2015, 2025)].groupby("type").agg(supplier_years=("cik", "size"), links=("n", "sum"), anon=("a", "sum"))
    t3["anon_share_%"] = 100 * t3["anon"] / t3["links"]
    print("== C3 진입 유형별 익명 비중 (FY2015–2025) ==\n", t3.round(1).to_string())
    by = sy[sy["fy"].between(2015, 2025)].groupby(["fy", "type"]).apply(lambda x: 100 * x["a"].sum() / x["n"].sum()).unstack()
    print(by.round(1).to_string())
    # 2014→2025 기여: 2025 진입자 링크를 유형별로
    e25 = sy[(sy["fy"] == 2025) & (sy["type"] != "continuing")]
    print("2025 비존속 공급사 링크 유형 구성:", (e25.groupby("type")["n"].sum() / e25["n"].sum()).round(3).to_dict())
    t3.to_csv(RES / "rev_c3_entry_types.csv"); by.to_csv(RES / "rev_c3_entry_types_by_year.csv")

    # ── 정부 고객 민감도: Government 계열 generic 멤버를 '식별된 고객'으로 포함
    g = rel[rel["current"]].copy(); g["fy"] = fy_of(g["ddate"])
    gov = g["member"].fillna("").str.contains(r"Government|Federal|USGov|Navy|Army|AirForce|DepartmentOf|Dod$|StateAndLocal|Agency|Medicare|Medicaid", case=False)
    g = g[g["fy"].between(2012, 2025) & (g["cls"].isin(["anonymous", "named"]) | (g["cls"].eq("generic") & gov))]
    s_gov = g.groupby("fy")["cls"].apply(lambda s: (s == "anonymous").mean())
    s_base = r.groupby("fy")["cls"].apply(lambda s: (s == "anonymous").mean())
    print("\n== 정부 고객을 식별 고객으로 포함 시 익명 비중 ==\n", pd.DataFrame({"baseline": s_base, "incl_government_as_named": s_gov}).loc[[2012, 2016, 2020, 2025]].mul(100).round(1).to_string())
    print("정부 멤버 링크 수:", int((g["cls"] == "generic").sum()))

    # ── C4: 공동 배분 상한 — 고객별 3년 총매입여력 상한 + 하이퍼스케일러 5사 집합 상한
    rr = rel[rel["current"] & rel["cls"].isin(["anonymous", "named"]) & rel["est_cust_rev"].notna()].copy()
    rr["fy"] = fy_of(rr["ddate"]); rr = rr[rr["fy"].between(2023, 2025)]
    e = pd.read_parquet(INT / "item1_exposure.parquet"); e = e[e["status"] == "ok"]; e["kw"] = e[["ai", "dc", "compute", "infra"]].sum(1)
    top = set(e.loc[e["kw"] >= e["kw"].quantile(0.9), "cik"])
    rr = rr[pd.to_numeric(rr["sic"], errors="coerce").isin(AI_SIC) | rr["cik"].astype(int).isin(top)]
    rr["cust"] = pd.to_numeric(rr["cust_cik"], errors="coerce")
    anon = rr[rr["cls"] == "anonymous"]
    out = []
    for c, n in CUST.items():
        lo = rr.loc[(rr["cls2"] == "named_linked") & (rr["cust"] == c), "est_cust_rev"].sum()
        caps = {fy: capacity(c, fy) for fy in [2023, 2024, 2025]}
        feas = sum(anon[(anon["fy"] == fy) & (anon["cik"].astype(int) != c) & (anon["est_cust_rev"] <= THETA * caps[fy])]["est_cust_rev"].sum() for fy in caps)
        cap3 = THETA * sum(caps.values())
        out.append({"customer": n, "lower_$B": lo / 1e9, "lower_exact_$M": lo / 1e6, "upper_separate_$B": (lo + feas) / 1e9,
                    "aggregate_capacity_cap_$B": cap3 / 1e9, "upper_joint_$B": min(lo + feas, max(lo, cap3)) / 1e9})
    b = pd.DataFrame(out)
    hyper = ["Microsoft", "Amazon", "Alphabet", "Meta", "Oracle"]
    hb = b[b["customer"].isin(hyper)]
    set_upper = min(anon["est_cust_rev"].sum(), hb["aggregate_capacity_cap_$B"].sum() * 1e9) / 1e9 + hb["lower_$B"].sum()
    print("\n== C4 상한 (θ=0.5, FY2023–25) ==\n", b.round(2).to_string(index=False))
    print(f"하이퍼스케일러 5사 집합: 하한 ${hb['lower_$B'].sum():.2f}B / 공동 상한 ${set_upper:.1f}B / 익명 총액 ${anon['est_cust_rev'].sum()/1e9:.1f}B")
    for th in [0.25, 1.0]:
        tot = 0
        for c in [k for k, v in CUST.items() if v in hyper]:
            tot += th * sum(capacity(c, fy) for fy in [2023, 2024, 2025])
        print(f"  θ={th}: 5사 총매입여력 상한 ${tot/1e9:.0f}B")
    b.to_csv(RES / "rev_c4_bounds.csv", index=False)

    # ── C5: GT3 vs 실제 익명 관계의 공변량 차이
    rel2 = rel[rel["current"] & rel["est_cust_rev"].notna()].copy(); rel2["fy"] = fy_of(rel2["ddate"])
    f = pd.read_parquet(INT / "fundamentals.parquet"); f["ln_at"] = np.log(f["at"].where(f["at"] > 0))
    f["rnd_s"] = (f["rnd"].fillna(0) / f["rev"]).where(f["rev"] > 0).clip(0, 5)
    rel2 = rel2.assign(cik=rel2["cik"].astype(int)).merge(f[["cik", "year", "ln_at", "rnd_s"]].rename(columns={"year": "fy"}), on=["cik", "fy"], how="left")
    rel2["ln_rel"] = np.log(rel2["est_cust_rev"].where(rel2["est_cust_rev"] > 0))
    rel2["ai"] = pd.to_numeric(rel2["sic"], errors="coerce").isin(AI_SIC)
    grp = {"GT3 (named, linked)": rel2[rel2["cls2"] == "named_linked"], "All anonymous": rel2[rel2["cls"] == "anonymous"]}
    cs = []
    for v, lab in [("ln_at", "Supplier ln(assets)"), ("rnd_s", "Supplier R&D/sales"), ("pct", "Customer revenue share"), ("ln_rel", "ln(relationship $)"), ("ai", "AI hardware supplier")]:
        a, b2 = grp["GT3 (named, linked)"][v].astype(float).dropna(), grp["All anonymous"][v].astype(float).dropna()
        cs.append({"variable": lab, "GT3_mean": a.mean(), "anonymous_mean": b2.mean(), "std_diff": (a.mean() - b2.mean()) / np.sqrt((a.var() + b2.var()) / 2)})
    cs = pd.DataFrame(cs)
    print("\n== C5 공변량 이동: GT3 vs 실제 익명 관계 ==\n", cs.round(3).to_string(index=False))
    cs.to_csv(RES / "rev_c5_covshift.csv", index=False)
    # GT1a 부트스트랩 신뢰구간 + 평가 흐름
    res = pd.concat([pd.read_parquet(p) for p in RES.glob("ranks_*.parquet")]).drop_duplicates(["year", "track", "model", "cik", "cust"])
    rng = np.random.default_rng(0)
    ci = []
    for m in ["B3 industry+size", "LGBM ranker (no graph features)", "Decoder only + product-similarity features", "GAE + product-similarity & location graph"]:
        d = res[(res["track"] == "gt1a") & (res["model"] == m)]
        rr_ = 1 / d["rank"].to_numpy()
        boots = [rng.choice(rr_, len(rr_)).mean() for _ in range(5000)]
        ci.append({"model": m, "n": len(rr_), "MRR": rr_.mean(), "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)})
    ci = pd.DataFrame(ci); print("\n== GT1a MRR 부트스트랩 95% CI ==\n", ci.round(3).to_string(index=False)); ci.to_csv(RES / "rev_c5_gt1a_ci.csv", index=False)
    gt3 = pd.read_parquet(INT / "gt3_named.parquet"); gt3["fy"] = fy_of(gt3["ddate"])
    g3e = res[(res["track"] == "gt3") & (res["model"] == "B3 industry+size")]
    flow = [("GT3 links (named, linked, current year)", len(gt3)),
            ("… with estimated relationship value", int(gt3["est_cust_rev"].notna().sum())),
            ("… in evaluation years FY2015–2025", int((gt3["est_cust_rev"].notna() & gt3["fy"].between(2015, 2025)).sum())),
            ("… customer in candidate pool (t−1 financials)", len(g3e)),
            ("… new relationships (no prior named link)", int((~g3e["history"]).sum())),
            ("… persistent relationships", int(g3e["history"].sum()))]
    print("\n== 평가 표본 흐름 ==");  [print(f"  {a}: {b_}") for a, b_ in flow]
    pd.DataFrame(flow, columns=["step", "n"]).to_csv(RES / "rev_c5_flow.csv", index=False)


if __name__ == "__main__":
    main()
