"""C2 위약용: 비AI 공급사(주요고객 공시 보유, AI 업종·AI 노출 상위 10% 제외) 무작위 400곳 companyfacts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import CACHE, get_json  # noqa: E402

INT = Path(__file__).resolve().parents[1] / "data" / "interim"
p = pd.read_parquet(INT / "did_panel.parquet")
t = pd.read_parquet(INT / "quarterly_targets.parquet")
cand = sorted(set(p.loc[p["fy"] >= 2016, "cik"]) - set(t["cik"]))
rng = np.random.default_rng(42)
pick = sorted(rng.choice(cand, min(400, len(cand)), replace=False).tolist())
for i, c in enumerate(pick, 1):
    get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(c):010d}.json", CACHE / "companyfacts" / f"{int(c)}.json")
    if i % 100 == 0:
        print(i, flush=True)
pd.Series(pick, name="cik").to_frame().to_parquet(INT / "nonai_targets.parquet")
print("done", len(pick), flush=True)
