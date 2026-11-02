"""3차 재심사: (a) 노출 구간 차이 직접검정 (b) 매칭 후 균형·사전추세 (c) 좌측 절단 점검."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from build_gt import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"
CTRL = ["ln_at", "roa", "rnd", "n_cust", "max_share"]


def fy_of(dd):
    s = dd.astype(str)
    return s.str[:4].astype(int) - (s.str[4:6].astype(int) < 7)


def base():
    p = pd.read_parquet(INT / "did_panel.parquet")
    e = pd.read_parquet(INT / "item1_exposure.parquet"); e = e[e["status"] == "ok"].copy()
    e["kw"] = e[["ai", "dc", "compute", "infra"]].sum(1)
    p90 = e["kw"].quantile(0.9)
    e["bin"] = np.where(e["kw"] == 0, "zero", np.where(e["kw"] < p90, "low", "high"))
    e["top"] = (e["kw"] >= p90).astype(float)
    d = p.merge(e[["cik", "kw", "bin", "top"]], on="cik")
    d["post"] = (d["fy"] >= 2023).astype(float)
    d["sic2"] = (d["sic"] // 100).fillna(0).astype(int)
    return d


def bins_test(d):
    d = d.copy(); d["low_post"] = (d["bin"] == "low") * d["post"]; d["high_post"] = (d["bin"] == "high") * d["post"]
    out = []
    for ctrl in [[], CTRL]:
        z = d.dropna(subset=["anon_share"] + ctrl).set_index(["cik", "fy"])
        m = PanelOLS(z["anon_share"], z[["low_post", "high_post"] + ctrl], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
        b, V = m.params, m.cov
        diff = b["high_post"] - b["low_post"]; se = np.sqrt(V.loc["high_post", "high_post"] + V.loc["low_post", "low_post"] - 2 * V.loc["high_post", "low_post"])
        out.append({"controls": bool(ctrl), "high−low": diff, "t": diff / se, "p": 2 * (1 - stats.norm.cdf(abs(diff / se)))})
    t = pd.DataFrame(out); print("== 실질 노출 − 언급 노출 차이 검정 ==\n", t.round(3).to_string(index=False)); return t


def matched(d):
    pre = d[d["fy"].between(2019, 2021)].groupby("cik").agg(top=("top", "first"), anon_share=("anon_share", "mean"), ln_at=("ln_at", "mean"),
                                                            roa=("roa", "mean"), rnd=("rnd", "mean"), n_cust=("n_cust", "mean"),
                                                            max_share=("max_share", "mean"), sic2=("sic2", "first"))
    pre = pre.dropna(subset=["ln_at", "rnd", "anon_share"]); pre["sic1"] = pre["sic2"] // 10
    zc = pre[["ln_at", "rnd", "anon_share"]].apply(lambda s: (s - s.mean()) / s.std())
    keep_c, keep_t = set(), set()
    for c in pre.index[pre["top"] == 1]:
        pool = pre[(pre["top"] == 0) & (pre["sic1"] == pre.loc[c, "sic1"])]
        if len(pool) == 0:
            continue
        dist = ((zc.loc[pool.index] - zc.loc[c]) ** 2).sum(1)
        keep_c |= set(dist.nsmallest(3).index); keep_t.add(c)
    def sd(a, b):
        a, b = a.dropna(), b.dropna(); return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    rows = []
    for v in ["anon_share", "ln_at", "roa", "rnd", "n_cust", "max_share"]:
        rows.append({"variable": v, "std_diff_before": sd(pre.loc[pre["top"] == 1, v], pre.loc[pre["top"] == 0, v]),
                     "std_diff_after": sd(pre.loc[list(keep_t), v], pre.loc[list(keep_c), v])})
    bal = pd.DataFrame(rows)
    print(f"\n== 매칭 전후 표준화 차이 (처치 {len(keep_t)}, 매칭 비교 {len(keep_c)}) ==\n", bal.round(3).to_string(index=False))
    dm = d[d["cik"].isin(keep_t | keep_c)].dropna(subset=["anon_share"]).copy()
    yrs = [y for y in range(2014, 2026) if y != 2021]
    for y in yrs:
        dm[f"t_{y}"] = dm["top"] * (dm["fy"] == y)
    z = dm.set_index(["cik", "fy"])
    m = PanelOLS(z["anon_share"], z[[f"t_{y}" for y in yrs]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    pre_idx = [i for i, y in enumerate(yrs) if y < 2021]
    R = np.zeros((len(pre_idx), len(yrs)))
    for k, i in enumerate(pre_idx):
        R[k, i] = 1
    b, V = m.params.values, m.cov.values
    chi = float(b @ R.T @ np.linalg.inv(R @ V @ R.T) @ R @ b); pv = 1 - stats.chi2.cdf(chi, len(pre_idx))
    es = pd.DataFrame({"year": yrs, "coef": b, "se": m.std_errors.values})
    print(f"매칭 표본 사건연구: 처치 이전 결합 χ²({len(pre_idx)}) = {chi:.2f}, p = {pv:.3f}")
    print(es.round(3).to_string(index=False))
    bal.to_csv(RES / "rev3_matched_balance.csv", index=False); es.to_csv(RES / "rev3_matched_eventstudy.csv", index=False)
    pd.DataFrame([{"chi2": chi, "df": len(pre_idx), "p": pv}]).to_csv(RES / "rev3_matched_pretrend.csv", index=False)


def censoring():
    """좌측 절단: (i) 이력 판정을 본 표본이 아니라 '모든 주요고객 공시'로, (ii) FY2018–2025 로 한정(룩백 ≥ 7년)."""
    rel = pd.read_parquet(INT / "relations.parquet")
    allr = rel[rel["current"] & rel["cls"].isin(["anonymous", "named", "generic"])].copy()
    allr["fy"] = fy_of(allr["ddate"]); allr["cik"] = allr["cik"].astype(int)
    hist = allr.groupby("cik")["fy"].apply(set)  # 재무 조건과 무관한 전체 공시 이력
    r = rel[rel["current"] & rel["cls"].isin(["anonymous", "named"])].copy(); r["fy"] = fy_of(r["ddate"]); r["cik"] = r["cik"].astype(int)
    f = pd.read_parquet(INT / "fundamentals.parquet")[["cik", "year", "at", "rev"]].dropna()
    main = r.merge(f.rename(columns={"year": "fy"}), on=["cik", "fy"])
    sy = main.groupby(["cik", "fy"]).agg(n=("cls", "size"), a=("cls", lambda s: (s == "anonymous").sum())).reset_index()
    sub = load("sub"); k = sub[sub["form"].isin(["10-K", "10-K/A"])].copy(); k["cik"] = k["cik"].astype(int)
    k["fy0"] = fy_of(k["period"].fillna("20000101").astype(str).str[:8]); first10k = k.groupby("cik")["fy0"].min()
    def etype(c, y):
        prev = {v for v in hist.get(c, set()) if v < y}
        if prev:
            return "Continuing" if (y - 1) in prev else "Re-entrant"
        return "New SEC registrant" if y - first10k.get(c, y) <= 1 else "Existing registrant, first disclosure"
    sy["type"] = [etype(c, y) for c, y in zip(sy["cik"], sy["fy"])]
    out = []
    for lab, lo in [("FY2015–2025, history from all disclosures", 2015), ("FY2018–2025, history from all disclosures", 2018)]:
        x = sy[sy["fy"].between(lo, 2025)].groupby("type").agg(sy_n=("cik", "size"), links=("n", "sum"), a=("a", "sum"))
        x["anon_%"] = 100 * x["a"] / x["links"]; x["sample"] = lab; out.append(x.reset_index())
    t = pd.concat(out)
    print("\n== 좌측 절단 점검: 유형별 익명 비중 ==\n", t[["sample", "type", "sy_n", "links", "anon_%"]].round(1).to_string(index=False))
    early = (first10k <= 2011).mean()
    print(f"첫 10-K가 데이터 시작(2009–2011)에 있는 공급사 비율: {early:.1%} (FY2015 이후 '신규 등록' 분류 대상 아님)")
    t.to_csv(RES / "rev3_censoring.csv", index=False)


if __name__ == "__main__":
    d = base(); bins_test(d).to_csv(RES / "rev3_bins_diff.csv", index=False); matched(d); censoring()
