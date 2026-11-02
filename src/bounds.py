"""부분식별: 주요 AI 고객별 '공급사로부터 사들이는 공시된 금액'의 상·하한 (FY2023–2025 합산).
하한 = 그 고객을 실명으로 공시한 관계의 추정 거래액 합
상한 = 하한 + 그 고객이 매입 여력상 가능한(추정 거래액 ≤ θ × 고객의 연간 매입여력) 익명 관계 거래액 합
매입여력: 플랫폼 = 매출원가 + 설비투자, 하드웨어 OEM = 매출원가. 익명 관계 하나는 여러 고객의 상한에 동시에 들어갈 수 있다(고객별 상한)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from revealed_network import CAPEX, COGS, OEM, PLATFORM, quarterly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"
THETA = 0.5
CUST = {**{c: n for c, n in PLATFORM.items() if n != "CoreWeave"}, **OEM}


FUND = pd.read_parquet(INT / "fundamentals.parquet")
AI_SIC = {3674, 3572, 3576, 3577, 3571, 3672, 3679, 3661, 3663, 3612, 3613, 3621, 3585}


def capacity(c, fy):
    cg = quarterly(c, COGS)
    cg = cg[cg.index.year == fy].sum() if len(cg) else 0.0
    if cg <= 0:  # 분기 시계열이 없으면 연간 frames 값으로 보충
        r = FUND[(FUND["cik"] == c) & (FUND["year"] == fy)]
        cg = float(r["cogs"].fillna(r["rev"]).iloc[0]) if len(r) and pd.notna(r["cogs"].fillna(r["rev"]).iloc[0]) else 0.0
    if c in PLATFORM:
        cx = quarterly(c, CAPEX, ytd=True)
        cg += cx[cx.index.year == fy].sum() if len(cx) else 0.0
    return cg


def main():
    rel = pd.read_parquet(INT / "relations.parquet")
    rel = rel[rel["current"] & rel["cls"].isin(["anonymous", "named"]) & rel["est_cust_rev"].notna()].copy()
    rel["fy"] = rel["ddate"].astype(str).str[:4].astype(int) - (rel["ddate"].astype(str).str[4:6].astype(int) < 7)
    rel = rel[rel["fy"].between(2023, 2025)]
    rel["cust"] = pd.to_numeric(rel["cust_cik"], errors="coerce")
    e = pd.read_parquet(INT / "item1_exposure.parquet"); e = e[e["status"] == "ok"]
    e["kw"] = e[["ai", "dc", "compute", "infra"]].sum(1)
    top = set(e.loc[e["kw"] >= e["kw"].quantile(0.9), "cik"])
    rel = rel[pd.to_numeric(rel["sic"], errors="coerce").isin(AI_SIC) | rel["cik"].astype(int).isin(top)]  # AI 하드웨어·노출 공급사만
    out = []
    for c, n in CUST.items():
        lo = rel.loc[(rel["cls2"] == "named_linked") & (rel["cust"] == c), "est_cust_rev"].sum()
        n_named = rel.loc[(rel["cls2"] == "named_linked") & (rel["cust"] == c), "cik"].nunique()
        up_add, n_feas = 0.0, set()
        for fy in [2023, 2024, 2025]:
            cap = capacity(c, fy)
            a = rel[(rel["cls"] == "anonymous") & (rel["fy"] == fy) & (rel["cik"].astype(int) != c)]
            feas = a[a["est_cust_rev"] <= THETA * cap]
            up_add += feas["est_cust_rev"].sum()
            n_feas |= set(feas["cik"])
        out.append({"customer": n, "identified_$B (lower)": lo / 1e9, "upper_$B": (lo + up_add) / 1e9,
                    "upper/lower": (lo + up_add) / lo if lo > 0 else np.inf,
                    "suppliers naming it": n_named, "suppliers possibly hiding it": len(n_feas)})
    t = pd.DataFrame(out).sort_values("upper_$B", ascending=False)
    pd.set_option("display.width", 200)
    print("== FY2023–2025 AI 하드웨어·노출 공급사로부터의 공시된 매입액 상·하한 (θ=0.5) ==\n", t.round(1).to_string(index=False))
    t.to_csv(RES / "bounds_ai_customers.csv", index=False)


if __name__ == "__main__":
    main()
