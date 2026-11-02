"""SEC Financial Statement and Notes 데이터셋 전체(2009~) → 주요고객 팩트 + 연결 매출/매출원가 + 제출정보.

파일마다: 다운로드 → MajorCustomers 차원 팩트, 차원 없는 연간(qtrs=4) 매출·매출원가, sub 추출 → parquet 저장 → zip 삭제.
같은 공시가 분기·월별 파일에 중복될 수 있으므로 최종 단계에서 adsh 기준 중복 제거.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "fsn"
OUT.mkdir(parents=True, exist_ok=True)
TOTAL_TAGS = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
              "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
              "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfGoodsAndServicesSoldExcludingDepreciationDepletionAndAmortization",
              "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"}
SUB_COLS = ["adsh", "cik", "name", "sic", "countryba", "stprba", "former", "afs", "fye", "form", "period", "fy", "fp", "filed"]
READ = dict(sep="\t", dtype=str, quoting=3, on_bad_lines="skip")


def process(url: str):
    stem = url.rsplit("/", 1)[-1].replace(".zip", "")
    done = OUT / f"{stem}.done"
    if done.exists():
        return "skip"
    zpath = ROOT / "data" / "raw" / f"{stem}.zip"
    get("https://www.sec.gov" + url, zpath)
    z = zipfile.ZipFile(zpath)
    names = {n.split(".")[0]: n for n in z.namelist()}
    dim = pd.read_csv(z.open(names["dim"]), **READ)
    mc_hash = dim[dim["segments"].str.contains("MajorCustomers=", na=False)][["dimhash", "segments"]]
    nodim = set(dim.loc[dim["segments"].isna() | (dim["segments"] == ""), "dimhash"]) | {"0x00000000"}
    cust, tot = [], []
    for ch in pd.read_csv(z.open(names["num"]), chunksize=2_000_000,
                          usecols=["adsh", "tag", "version", "ddate", "qtrs", "uom", "dimh", "value"], **READ):
        cust.append(ch[ch["dimh"].isin(set(mc_hash["dimhash"]))])
        tot.append(ch[ch["tag"].isin(TOTAL_TAGS) & ch["dimh"].isin(nodim) & (ch["qtrs"] == "4") & (ch["uom"] == "USD")])
    cust = pd.concat(cust).merge(mc_hash, left_on="dimh", right_on="dimhash").drop(columns=["dimhash"])
    tot = pd.concat(tot).drop(columns=["dimh"])
    sub = pd.read_csv(z.open(names["sub"]), **READ)
    sub = sub[[c for c in SUB_COLS if c in sub.columns]]
    cust.assign(src=stem).to_parquet(OUT / f"{stem}_cust.parquet")
    tot.assign(src=stem).to_parquet(OUT / f"{stem}_tot.parquet")
    sub.assign(src=stem).to_parquet(OUT / f"{stem}_sub.parquet")
    z.close()
    if stem != "2024q1_notes":
        zpath.unlink()
    done.touch()
    return f"cust {len(cust)} tot {len(tot)} sub {len(sub)}"


def main():
    urls = [u.strip() for u in (ROOT / "data" / "fsn_urls.txt").read_text().split() if u.strip()]
    for i, u in enumerate(reversed(urls), 1):  # 오래된 것부터
        try:
            print(f"[{i}/{len(urls)}] {u.rsplit('/', 1)[-1]}: {process(u)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(urls)}] {u.rsplit('/', 1)[-1]}: ERROR {e!r}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
