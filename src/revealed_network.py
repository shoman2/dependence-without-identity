"""A: 드러난 AI 생산 네트워크 — 분기 시계열 반응으로 공급사의 고객을 식별.

고객 신호: 플랫폼 = 분기 설비투자, 하드웨어 OEM = 분기 매출원가(부품 매입 대리).
공급사: 분기 매출. 모두 전년동기 대비 로그증가율(계절성 제거), 후보 평균을 빼 공통 AI 사이클 제거.
식별 점수: 공급사 증가율 ~ 후보 고유 증가율(시차 0·1분기) 회귀의 R² (+ 거래액 상한 제약).
검증: 후보 중 하나를 실명 고객으로 공시한 공급사-연도 → 실제 고객의 순위. 무작위 1위 확률 = 1/후보수.
적용: AI 노출 공급사의 익명 고객(2022~) → 추정 고객 분포.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import CACHE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"
PLATFORM = {789019: "Microsoft", 1018724: "Amazon", 1652044: "Alphabet", 1326801: "Meta", 1341439: "Oracle",
            1318605: "Tesla", 1769628: "CoreWeave"}
OEM = {1045810: "NVIDIA", 2488: "AMD", 1571996: "Dell", 1645590: "HPE", 1375365: "SuperMicro", 858877: "Cisco",
       320193: "Apple", 50863: "Intel"}
CANDS = {**PLATFORM, **OEM}
REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
         "PaymentsForProceedsFromProductiveAssets", "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"]
COGS = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"]


def facts(cik):
    p = CACHE / "companyfacts" / f"{cik}.json"
    if not p.exists():
        return None
    return json.loads(p.read_bytes()).get("facts", {}).get("us-gaap", {})


def quarterly(cik, tags, ytd=False):
    """분기(약 3개월) 값 시계열. ytd=True 이면 같은 회계연도 누적값 차분(현금흐름표 항목)."""
    f = facts(cik)
    if not f:
        return pd.Series(dtype=float)
    best = None
    for t in tags:
        u = f.get(t, {}).get("units", {}).get("USD", [])
        if not u:
            continue
        d = pd.DataFrame(u)
        d = d[d["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"]) & d["start"].notna()]
        d["start"], d["end"] = pd.to_datetime(d["start"]), pd.to_datetime(d["end"])
        d["days"] = (d["end"] - d["start"]).dt.days
        d = d.sort_values("filed").drop_duplicates(["start", "end"], keep="last")
        q = d[d["days"].between(80, 100)].set_index("end")["val"]
        q = q[~q.index.duplicated(keep="last")]  # 같은 분기말에 기간이 다른 값이 중복된 공시 정리
        if ytd or len(q) < 8:
            # 누적값(같은 시작일) 차분으로 분기값 복원
            rows = {}
            for s, g in d.groupby("start"):
                g = g.sort_values("end")
                prev = 0.0
                for _, r in g.iterrows():
                    rows[r["end"]] = r["val"] - prev
                    prev = r["val"]
            q2 = pd.Series(rows)
            q = q.combine_first(q2) if len(q) else q2
        q = q[~q.index.duplicated()].sort_index()
        if best is None or len(q) > len(best):
            best = q
    if best is None:
        return pd.Series(dtype=float)
    best.index = pd.PeriodIndex(best.index, freq="Q")
    return best.groupby(level=0).last()


def yoy(s):
    s = s[s > 0]
    return np.log(s).diff(4).dropna() if len(s) > 4 else pd.Series(dtype=float)


def main():
    cand_sig = {}
    for c, n in PLATFORM.items():
        cand_sig[c] = yoy(quarterly(c, CAPEX, ytd=True))
    for c, n in OEM.items():
        cand_sig[c] = yoy(quarterly(c, COGS))
    C = pd.DataFrame(cand_sig).loc["2013Q1":"2026Q2"]
    C = C.clip(-1.5, 1.5)
    common = C.mean(1)
    Cres = C.sub(common, axis=0)  # 공통 AI 사이클 제거
    lvl = {c: (quarterly(c, CAPEX, ytd=True) if c in PLATFORM else quarterly(c, COGS)) for c in CANDS}
    print("후보 신호 분기 수:", {CANDS[c]: int(C[c].notna().sum()) for c in C})

    rel = pd.read_parquet(INT / "relations.parquet")
    rel = rel[rel["current"] & rel["cls"].isin(["anonymous", "named"])].copy()
    rel["fy_end"] = pd.to_datetime(rel["ddate"].astype(str))
    rel["cik"] = rel["cik"].astype(int)
    targets = set(pd.read_parquet(INT / "quarterly_targets.parquet")["cik"])

    def score_supplier(sup, end, est_rev):
        rv = yoy(quarterly(sup, REV))
        q_end = pd.Period(end, freq="Q")
        win = rv.loc[q_end - 11: q_end].clip(-1.5, 1.5) if len(rv) else rv
        if len(win) < 8:
            return None
        win = win - common.reindex(win.index).fillna(0)
        out = {}
        for c in CANDS:
            x0 = Cres[c].reindex(win.index)
            x1 = Cres[c].shift(1).reindex(win.index)
            d = pd.DataFrame({"y": win, "x0": x0, "x1": x1}).dropna()
            if len(d) < 6:
                continue
            X = np.c_[np.ones(len(d)), d["x0"], d["x1"]]
            beta, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
            resid = d["y"].to_numpy() - X @ beta
            r2 = 1 - resid.var() / max(d["y"].var(), 1e-12)
            pos = beta[1] + beta[2] > 0  # 고객이 늘리면 공급사 매출도 늘어야
            # 거래액 상한: 공급사→고객 연 거래액 ≤ 고객의 연간 설비투자/매출원가
            ann = lvl[c].loc[q_end - 3: q_end].sum() if len(lvl[c]) else np.nan
            ok = not (np.isfinite(ann) and est_rev > ann)
            out[c] = r2 if (pos and ok) else -1.0
        return out

    # ── 검증: 후보를 실명 고객으로 공시한 관계
    val = rel[(rel["cls2"] == "named_linked") & rel["cust_cik"].astype(float).isin(CANDS) & rel["cik"].isin(targets)]
    val = val.assign(cust=val["cust_cik"].astype(float).astype(int)).drop_duplicates(["cik", "ddate", "cust"])
    vrows = []
    for r in val.itertuples():
        sc = score_supplier(r.cik, r.fy_end, r.est_cust_rev if pd.notna(r.est_cust_rev) else 0)
        if not sc or r.cust not in sc:
            continue
        ranked = sorted(sc, key=lambda c: -sc[c])
        vrows.append({"supplier": r.name, "year": r.fy_end.year, "true": CANDS[r.cust], "rank": ranked.index(r.cust) + 1,
                      "n_cand": len(sc), "top": CANDS[ranked[0]]})
    v = pd.DataFrame(vrows)
    v.to_csv(RES / "revealed_validation.csv", index=False)
    if len(v):
        print(f"\n== 검증: 실명 고객 {len(v)}건 (공급사 {v['supplier'].nunique()}) ==")
        print("Hit@1", round((v["rank"] == 1).mean(), 3), "| Hit@3", round((v["rank"] <= 3).mean(), 3),
              "| MRR", round((1 / v["rank"]).mean(), 3), "| 무작위 Hit@1 ≈", round((1 / v["n_cand"]).mean(), 3),
              "| 무작위 Hit@3 ≈", round((3 / v["n_cand"]).clip(upper=1).mean(), 3))
        print("실제 고객별:", v.groupby("true")["rank"].agg(["size", lambda r: (r == 1).mean()]).round(2).to_dict("index"))
        print(v.sample(min(12, len(v)), random_state=0).to_string(index=False))

    # ── 적용: AI 노출 공급사 익명 고객 (FY2022~)
    an = rel[(rel["cls"] == "anonymous") & rel["cik"].isin(targets) & (rel["fy_end"] >= "2022-07-01")]
    an = an.sort_values("pct", ascending=False).drop_duplicates(["cik", "ddate", "member"])
    arows = []
    for r in an.itertuples():
        sc = score_supplier(r.cik, r.fy_end, r.est_cust_rev if pd.notna(r.est_cust_rev) else 0)
        if not sc:
            continue
        ranked = sorted(sc, key=lambda c: -sc[c])
        arows.append({"supplier": r.name, "fy_end": r.fy_end.date(), "member": r.member, "pct": r.pct,
                      "top1": CANDS[ranked[0]], "r2_top1": round(sc[ranked[0]], 3), "top2": CANDS[ranked[1]],
                      "r2_top2": round(sc[ranked[1]], 3)})
    a = pd.DataFrame(arows)
    a.to_csv(RES / "revealed_anonymous.csv", index=False)
    if len(a):
        print(f"\n== 적용: 익명 고객 {len(a)}건 → 추정 1위 고객 분포 ==")
        print(a["top1"].value_counts().to_dict())


if __name__ == "__main__":
    main()
