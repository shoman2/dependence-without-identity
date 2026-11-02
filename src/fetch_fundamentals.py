"""공급사·후보 고객사 재무변수 (SEC frames API, 전 제출기업) + FF-12 산업 매핑."""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import CACHE, UA, get_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DUR = {"Revenues": "rev", "RevenueFromContractWithCustomerExcludingAssessedTax": "rev606", "SalesRevenueNet": "rev_old",
       "NetIncomeLoss": "ni", "ResearchAndDevelopmentExpense": "rnd", "GrossProfit": "gp",
       "CostOfRevenue": "cogs1", "CostOfGoodsAndServicesSold": "cogs2", "CostOfGoodsSold": "cogs3"}
INST = {"Assets": "at", "StockholdersEquity": "ceq"}


def frame(tax, concept, period):
    js = get_json(f"https://data.sec.gov/api/xbrl/frames/{tax}/{concept}/USD/{period}.json",
                  CACHE / "frames" / f"{tax}_{concept}_{period}.json")
    return pd.DataFrame(js["data"])[["cik", "val"]] if js else pd.DataFrame(columns=["cik", "val"])


def main():
    rows = []
    for y in range(2010, 2026):
        for c, v in DUR.items():
            rows.append(frame("us-gaap", c, f"CY{y}").assign(year=y, var=v))
        for c, v in INST.items():
            rows.append(frame("us-gaap", c, f"CY{y}Q4I").assign(year=y, var=v))
        rows.append(frame("dei", "EntityPublicFloat", f"CY{y}Q2I").assign(year=y, var="pubfloat"))
        print(y, flush=True)
    p = pd.concat(rows).pivot_table(index=["cik", "year"], columns="var", values="val", aggfunc="first").reset_index()
    p["rev"] = p["rev"].fillna(p.get("rev606")).fillna(p.get("rev_old"))
    p["cogs"] = p["cogs1"].fillna(p["cogs2"]).fillna(p["cogs3"])
    p.loc[p["cogs"].isna() & p["gp"].notna(), "cogs"] = p["rev"] - p["gp"]
    p.to_parquet(ROOT / "data" / "interim" / "fundamentals.parquet")
    print(p.shape, p[["rev", "cogs", "at", "rnd", "pubfloat"]].notna().mean().round(3).to_dict())
    # FF-12 SIC 매핑 (Ken French Data Library)
    r = requests.get("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes12.zip",
                     headers={"User-Agent": UA}, timeout=60)
    txt = zipfile.ZipFile(io.BytesIO(r.content)).read("Siccodes12.txt").decode("latin-1")
    (ROOT / "data" / "raw" / "Siccodes12.txt").write_text(txt)
    print(txt[:300])


if __name__ == "__main__":
    main()
