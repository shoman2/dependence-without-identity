"""논문 Table 1~3 (데이터 섹션). 모든 수치는 파이프라인 산출물에서 계산 — 예시값 없음.

Table 1  표본 구축 과정 / 연도별 / FF-12 산업별 익명·실명 분포
Table 2  그래프 위상 + 회계적 상한 제약에 따른 후보 풀 축소(실제 고객 보존율 포함) + 정답셋 요약
Table 3  익명 vs 실명 공급사, 전체 익명 vs 정답셋 익명 공급사 특성 비교 (Welch t, Wilcoxon z)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from build_gt import CONSOLIDATED_AXES, load, parse_segments  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"
THETAS = [0.25, 0.5, 1.0]


def fyear(ddate: pd.Series) -> pd.Series:
    d = ddate.astype(str)
    return d.str[:4].astype(int) - (d.str[4:6].astype(int) < 7).astype(int)


def ff12_map():
    m, cur = {}, None
    for line in (ROOT / "data" / "raw" / "Siccodes12.txt").read_text().splitlines():
        h = re.match(r"\s*(\d+)\s+(\w+)\s+(.*)", line)
        r = re.match(r"\s*(\d{4})-(\d{4})", line)
        if h and not r:
            cur = h.group(2)
        elif r and cur:
            for s in range(int(r.group(1)), int(r.group(2)) + 1):
                m[s] = cur
    return lambda sic: m.get(int(sic), "Other") if pd.notna(sic) and str(sic).isdigit() else "Other"


def table1(rel, cust, sub):
    k = sub[sub["form"].isin(["10-K", "10-K/A", "10-KT"])][["adsh", "cik"]]
    c = cust.merge(k, on="adsh")
    steps = [("10-K XBRL filings with MajorCustomers facts", c)]
    seg = c["segments"].map(parse_segments)
    c2 = c[seg.map(lambda d: set(d) <= CONSOLIDATED_AXES)]
    steps.append(("Consolidated facts only (no segment/product/geographic axes)", c2))
    cur = rel[rel["current"]]
    steps.append(("Revenue-based concentration, current fiscal year", cur))
    cur2 = cur[cur["cls2"] != "generic"]
    steps.append(("Exclude aggregate / government / category members", cur2))
    fund = pd.read_parquet(INT / "fundamentals.parquet")
    cur3 = cur2.assign(year=fyear(cur2["ddate"]), cik=cur2["cik"].astype(int)).merge(
        fund[["cik", "year", "at", "rev"]].dropna(), on=["cik", "year"])
    steps.append(("Require supplier total assets and revenue (SEC frames)", cur3))
    pa = []
    for name, d in steps:
        member_col = "member" if "member" in d.columns else "segments"
        pa.append({"Step": name, "Filings": d["adsh"].nunique(), "Suppliers": d["cik"].nunique(),
                   "Customer edges": d.drop_duplicates(["adsh", member_col]).shape[0]})
    for lab, mask in [("  Named (linked to SEC filer)", cur3["cls2"] == "named_linked"),
                      ("  Named (not linked: private/foreign/abbrev.)", cur3["cls2"] == "named_unlinked"),
                      ("  Anonymous", cur3["cls2"] == "anonymous")]:
        d = cur3[mask]
        pa.append({"Step": lab, "Filings": d["adsh"].nunique(), "Suppliers": d["cik"].nunique(), "Customer edges": len(d)})
    pa = pd.DataFrame(pa)

    final = cur3
    yb = final.groupby("year").agg(supplier_years=("cik", "nunique"), edges=("member", "size"),
                                   named=("cls", lambda s: (s == "named").sum()),
                                   anonymous=("cls", lambda s: (s == "anonymous").sum()))
    yb["anon_ratio_%"] = 100 * yb["anonymous"] / yb["edges"]
    yb["anon_mean_share_%"] = final[final["cls"] == "anonymous"].groupby("year")["pct"].mean() * 100
    tot = yb[["supplier_years", "edges", "named", "anonymous"]].sum()
    yb.loc["Total"] = [*tot, 100 * tot["anonymous"] / tot["edges"],
                       100 * final.loc[final["cls"] == "anonymous", "pct"].mean()]

    ff = ff12_map()
    final = final.assign(ff12=final["sic"].map(ff))
    pc = final.groupby("ff12").agg(edges=("member", "size"), named=("cls", lambda s: (s == "named").sum()),
                                   anonymous=("cls", lambda s: (s == "anonymous").sum()), mean_share_pct=("pct", "mean"))
    pc["anon_ratio_%"] = 100 * pc["anonymous"] / pc["edges"]
    pc["mean_share_pct"] *= 100
    return pa, yb.round(1), pc.sort_values("anon_ratio_%", ascending=False).round(1), final


def table2(final, gt1, gt3):
    fund = pd.read_parquet(INT / "fundamentals.parquet")
    named = final[final["cls2"] == "named_linked"]
    n_sup = final["cik"].nunique()
    n_cust = named["cust_cik"].nunique()
    known = named.drop_duplicates(["cik", "cust_cik"])
    pool = fund.dropna(subset=["at"]).groupby("year")["cik"].nunique()

    # 회계적 상한: 공급사 매출 × 비중(= 고객이 공급사에서 산 금액) ≤ θ × 고객 매출원가(없으면 매출)
    cap = fund.assign(purch=fund["cogs"].fillna(fund["rev"]))[["cik", "year", "purch", "at"]].dropna(subset=["purch"])
    rows = []
    for th in THETAS:
        # (a) 익명 관계: 후보 풀 크기 변화
        an = final[(final["cls"] == "anonymous") & final["est_cust_rev"].notna()]
        sizes = []
        for y, g in an.groupby("year"):
            cp = np.sort(cap.loc[cap["year"] == y, "purch"].to_numpy())
            if len(cp) == 0:
                continue
            need = g["est_cust_rev"].to_numpy() / th
            sizes.append(pd.DataFrame({"before": len(cp), "after": len(cp) - np.searchsorted(cp, need, side="left")}))
        sz = pd.concat(sizes) if sizes else pd.DataFrame({"before": [np.nan], "after": [np.nan]})
        # (b) 실명 정답(GT3): 실제 고객이 제약을 통과하는 비율 = 제약의 재현율
        g3 = gt3.assign(year=fyear(gt3["ddate"]), cust_cik=gt3["cust_cik"].astype(int)).merge(
            cap.rename(columns={"cik": "cust_cik"}), on=["cust_cik", "year"])
        g3 = g3[g3["est_cust_rev"].notna()]
        keep = (g3["est_cust_rev"] <= th * g3["purch"]).mean() if len(g3) else np.nan
        rows.append({"theta": th, "pool_before_mean": sz["before"].mean(), "pool_after_mean": sz["after"].mean(),
                     "reduction_%": 100 * (1 - sz["after"].mean() / sz["before"].mean()),
                     "true_customer_retained_% (GT3)": 100 * keep, "GT3_n_checked": len(g3)})
    pa = pd.DataFrame(rows).round(1)
    topo = pd.DataFrame([
        ("Supplier nodes (final sample)", n_sup),
        ("Linked customer nodes (SEC filers)", n_cust),
        ("Known supplier→customer edges (named & linked, unique pairs)", len(known)),
        ("Anonymous edges (target)", int((final["cls"] == "anonymous").sum())),
        ("Bipartite density of known edges", f"{len(known) / max(n_sup * n_cust, 1):.2e}"),
        ("Candidate pool: SEC filers with assets, mean per year", round(pool.mean())),
    ], columns=["Metric", "Value"])

    def gsum(df, cust_col, label, mech):
        if df is None or len(df) == 0:
            return {"Track": label, "Mechanism": mech, "Pairs": 0, "Suppliers": 0, "Customers": 0,
                    "Time to reveal (yrs, mean)": np.nan, "Customer ln(assets), mean": np.nan}
        d = df.assign(year=fyear(df["ddate"]), cc=df[cust_col].astype(int))
        ca = d.merge(fund[["cik", "year", "at"]].rename(columns={"cik": "cc"}), on=["cc", "year"], how="left")["at"]
        ttr = np.nan
        if "filed_named_first" in d:
            ttr = ((pd.to_datetime(d["filed_named_first"].astype(str)) - pd.to_datetime(d["filed"].astype(str))).dt.days / 365.25).mean()
        return {"Track": label, "Mechanism": mech, "Pairs": len(d.drop_duplicates(["cik", "ddate", "cc"])),
                "Suppliers": d["cik"].nunique(), "Customers": d["cc"].nunique(),
                "Time to reveal (yrs, mean)": round(ttr, 2) if pd.notna(ttr) else np.nan,
                "Customer ln(assets), mean": round(np.log(ca[ca > 0]).mean(), 2)}
    g1 = gt1[gt1["current"]]
    pb = pd.DataFrame([
        gsum(g1[g1["direction"] == "revealed_later"], "gt_cust_cik", "GT1a revealed later", "later 10-K names it at same period & share"),
        gsum(g1[g1["direction"] == "anonymized_later"], "gt_cust_cik", "GT1b anonymized later", "earlier 10-K named it at same period & share"),
        gsum(None, "x", "GT2 cross-filing text", "counterparty 10-K/8-K mention (pending)"),
        gsum(gt3, "cust_cik", "GT3 masked named", "named & linked, label masked at evaluation"),
    ])
    return topo, pa, pb


def table3(final, gt1):
    fund = pd.read_parquet(INT / "fundamentals.parquet")
    fund["ln_at"] = np.log(fund["at"].where(fund["at"] > 0))
    fund["roa"] = fund["ni"] / fund["at"]
    fund["rnd_sales"] = (fund["rnd"].fillna(0) / fund["rev"]).where(fund["rev"] > 0)
    fund["gross_margin"] = (1 - fund["cogs"] / fund["rev"]).where(fund["rev"] > 0)
    fund["book_to_float"] = (fund["ceq"] / fund["pubfloat"]).where(fund["pubfloat"] > 0)
    sic = final.drop_duplicates("cik").set_index("cik")["sic"]
    f2 = fund.merge(sic.rename("sic").reset_index(), on="cik", how="left")
    f2["sic3"] = f2["sic"].astype(str).str[:3]
    sh = f2.dropna(subset=["rev"]).query("rev > 0")
    hhi = sh.groupby(["sic3", "year"])["rev"].apply(lambda r: ((r / r.sum()) ** 2).sum()).rename("hhi_sic3").reset_index()

    sy = final.groupby(["cik", "year"]).agg(
        any_anon=("cls", lambda s: (s == "anonymous").any()), any_named=("cls", lambda s: (s == "named").any()),
        max_share=("pct", "max"), est_cust_rev=("est_cust_rev", "mean"), sic=("sic", "first")).reset_index()
    sy["sic3"] = sy["sic"].astype(str).str[:3]
    sy = sy.merge(fund[["cik", "year", "ln_at", "roa", "rnd_sales", "gross_margin", "book_to_float"]], on=["cik", "year"], how="left")
    sy = sy.merge(hhi, on=["sic3", "year"], how="left")
    gt_keys = set(zip(gt1.loc[gt1["current"], "cik"].astype(int), fyear(gt1.loc[gt1["current"], "ddate"])))
    sy["gt_anon"] = [(c, y) in gt_keys for c, y in zip(sy["cik"], sy["year"])]
    g_named, g_anon, g_gt = sy[sy["any_named"] & ~sy["any_anon"]], sy[sy["any_anon"]], sy[sy["any_anon"] & sy["gt_anon"]]

    def wins(x):
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = x.quantile([0.01, 0.99]) if len(x) > 20 else (x.min(), x.max())
        return x.clip(lo, hi)
    out = []
    labels = {"max_share": "Largest customer share", "est_cust_rev": "Est. sales to customer ($M)", "ln_at": "ln(Total assets)",
              "roa": "ROA", "rnd_sales": "R&D / Sales", "gross_margin": "Gross margin", "book_to_float": "Book equity / public float",
              "hhi_sic3": "Industry HHI (SIC3, revenue)"}
    for v, lab in labels.items():
        a, b, c = (wins(g[v] / (1e6 if v == "est_cust_rev" else 1)) for g in (g_named, g_anon, g_gt))
        row = {"Variable": lab}
        for name, x in [("(1) Named-only", a), ("(2) Any anonymous", b), ("(3) GT-matched anon", c)]:
            row[name] = f"{x.mean():.3f} [{x.median():.3f}]" if len(x) else "—"
        for name, x, y in [("(2)-(1)", b, a), ("(3)-(2)", c, b)]:
            if len(x) > 2 and len(y) > 2:
                t = stats.ttest_ind(x, y, equal_var=False)
                z = stats.ranksums(x, y)
                row[name] = f"{x.mean() - y.mean():+.3f} (t={t.statistic:.2f}, z={z.statistic:.2f})"
            else:
                row[name] = "—"
        out.append(row)
    n = {"(1) Named-only": len(g_named), "(2) Any anonymous": len(g_anon), "(3) GT-matched anon": len(g_gt)}
    return pd.DataFrame(out), n


def main():
    rel = pd.read_parquet(INT / "relations.parquet")
    gt1 = pd.read_parquet(INT / "gt1_comparative.parquet")
    gt3 = pd.read_parquet(INT / "gt3_named.parquet")
    cust, sub = load("cust"), load("sub").drop_duplicates("adsh")
    cust = cust.drop_duplicates(["adsh", "tag", "ddate", "qtrs", "segments", "value"])
    pa, yb, pc, final = table1(rel, cust, sub)
    topo, cons, gts = table2(final, gt1, gt3)
    t3, n3 = table3(final, gt1)
    with pd.ExcelWriter(RES / "tables_data_section.xlsx") as xw:
        for name, df in [("T1A_selection", pa), ("T1B_year", yb), ("T1C_ff12", pc), ("T2A_topology", topo),
                         ("T2A_constraint", cons), ("T2B_groundtruth", gts), ("T3_characteristics", t3)]:
            df.to_excel(xw, sheet_name=name, index=name in ("T1B_year", "T1C_ff12"))
    pd.set_option("display.width", 220)
    print("== Table 1A ==\n", pa.to_string(index=False))
    print("\n== Table 1B ==\n", yb.to_string())
    print("\n== Table 1C ==\n", pc.to_string())
    print("\n== Table 2A topology ==\n", topo.to_string(index=False))
    print("\n== Table 2A accounting constraint ==\n", cons.to_string(index=False))
    print("\n== Table 2B ground truth ==\n", gts.to_string(index=False))
    print("\n== Table 3 ==  N:", n3, "\n", t3.to_string(index=False))


if __name__ == "__main__":
    main()
