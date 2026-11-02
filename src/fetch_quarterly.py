"""A 준비: 하이퍼스케일러와 AI 노출 공급사의 companyfacts (분기 매출·설비투자 계산용)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import CACHE, get_json  # noqa: E402

INT = Path(__file__).resolve().parents[1] / "data" / "interim"
HYPER = {789019: "Microsoft", 1018724: "Amazon", 1652044: "Alphabet", 1326801: "Meta", 1341439: "Oracle",
         320193: "Apple", 1318605: "Tesla", 1769628: "CoreWeave", 1045810: "NVIDIA", 2488: "AMD", 1571996: "Dell",
         1645590: "HPE", 1375365: "SuperMicro", 858877: "Cisco", 50863: "Intel"}
AI_SIC = {3674, 3572, 3576, 3577, 3571, 3672, 3679, 3661, 3663, 3612, 3613, 3621, 3585}
p = pd.read_parquet(INT / "did_panel.parquet")
e = pd.read_parquet(INT / "item1_exposure.parquet")
e = e[e["status"] == "ok"]
e["kw"] = e[["ai", "dc", "compute", "infra"]].sum(1)
top = set(e.loc[e["kw"] >= e["kw"].quantile(0.9), "cik"])
sup = set(p.loc[(p["fy"] >= 2016) & (p["sic"].isin(AI_SIC) | p["cik"].isin(top)), "cik"])
ciks = sorted(set(HYPER) | sup)
print("대상 기업", len(ciks), flush=True)
for i, c in enumerate(ciks, 1):
    get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(c):010d}.json", CACHE / "companyfacts" / f"{int(c)}.json")
    if i % 50 == 0:
        print(i, flush=True)
pd.Series(ciks, name="cik").to_frame().assign(hyper=lambda d: d["cik"].isin(HYPER)).to_parquet(INT / "quarterly_targets.parquet")
print("done", flush=True)
