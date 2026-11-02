"""C: 부분관측 공급망 네트워크의 측정 왜곡.
익명 엣지는 정체는 모르지만 가중치(매출비중, 추정 거래액)는 안다 → 실명 엣지만 쓰는 네트워크가 놓치는 양을 정확히 계산.
  (1) 거래액 기준 '어두운' 비중: Σ익명 추정거래액 / Σ(익명+실명) 추정거래액
  (2) 공급사 커버리지: 주요고객이 모두 식별 가능(실명·EDGAR 연결) / 일부만 / 전혀 안 됨
  (3) 고객 집중도(HHI) 왜곡: 실명 엣지만으로 계산한 HHI vs 전체 엣지 HHI
  (4) 실명 네트워크로 잡히는 공급사 의존도의 과소측정 비율 (실명 비중 / 전체 비중)
AI 하드웨어 업종과 전체를 나눠 연도별로."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"
AI_SIC = {3674, 3572, 3576, 3577, 3571, 3672, 3679, 3661, 3663, 3612, 3613, 3621, 3585}


def main():
    r = pd.read_parquet(INT / "relations.parquet")
    r = r[r["current"] & r["cls"].isin(["anonymous", "named"])].copy()
    r["fy"] = r["ddate"].astype(str).str[:4].astype(int) - (r["ddate"].astype(str).str[4:6].astype(int) < 7)
    r = r[(r["fy"] >= 2012) & (r["fy"] <= 2025)]
    r["ai"] = pd.to_numeric(r["sic"], errors="coerce").isin(AI_SIC)
    r["vis"] = r["cls2"] == "named_linked"  # 실명 네트워크 연구가 실제로 쓸 수 있는 엣지
    out = []
    for grp, d in [("All", r), ("AI hardware", r[r["ai"]])]:
        for fy, g in d.groupby("fy"):
            v = g["est_cust_rev"].fillna(0)
            s = g.groupby(["cik"]).apply(lambda x: pd.Series({
                "tot": x["pct"].sum(), "vis": x.loc[x["vis"], "pct"].sum(),
                "hhi_all": (x["pct"] ** 2).sum(), "hhi_vis": (x.loc[x["vis"], "pct"] ** 2).sum(),
                "n_vis": x["vis"].sum(), "n": len(x)}))
            out.append({"group": grp, "fy": fy,
                        "dark_$_share": v[g["cls"] == "anonymous"].sum() / max(v.sum(), 1),
                        "unusable_$_share": v[~g["vis"]].sum() / max(v.sum(), 1),
                        "suppliers": len(s),
                        "fully_visible_%": 100 * (s["n_vis"] == s["n"]).mean(),
                        "partly_visible_%": 100 * ((s["n_vis"] > 0) & (s["n_vis"] < s["n"])).mean(),
                        "invisible_%": 100 * (s["n_vis"] == 0).mean(),
                        "dependence_captured_%": 100 * (s["vis"] / s["tot"]).mean(),
                        "HHI_all": s["hhi_all"].mean(), "HHI_named_only": s["hhi_vis"].mean()})
    t = pd.DataFrame(out)
    t["HHI_understated_%"] = 100 * (1 - t["HHI_named_only"] / t["HHI_all"])
    t.to_csv(RES / "dark_network.csv", index=False)
    pd.set_option("display.width", 220)
    for g in ["All", "AI hardware"]:
        print(f"\n== {g} ==")
        print(t[t["group"] == g].drop(columns="group").set_index("fy").round(3).to_string())


if __name__ == "__main__":
    main()
