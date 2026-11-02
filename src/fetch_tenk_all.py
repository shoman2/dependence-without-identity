"""익명 고객이 있는 모든 10-K(2012~2025) 본문 → 실명 고객 문장 추출. HTML 은 처리 후 삭제(문장만 보관)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import CACHE, get, get_json  # noqa: E402
from text_customers import clean_html, extract  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
OUT = INT / "tenk_text"
OUT.mkdir(exist_ok=True)


def primary_docs(cik):
    js = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", CACHE / "submissions" / f"{cik}.json")
    pages = [js["filings"]["recent"]] + [get_json(f"https://data.sec.gov/submissions/{f['name']}", CACHE / "submissions" / f["name"])
                                         for f in js["filings"].get("files", [])]
    return {a: d for p in pages for a, d in zip(p["accessionNumber"], p["primaryDocument"])}


def main():
    rel = pd.read_parquet(INT / "relations.parquet")
    rel = rel[rel["current"] & (rel["cls"] == "anonymous") & rel["form"].isin(["10-K", "10-K/A"])]
    todo = rel.drop_duplicates("adsh")[["adsh", "cik"]].assign(cik=lambda d: d["cik"].astype(int))
    done = {p.stem for p in OUT.glob("*.parquet")}
    todo = todo[~todo["adsh"].isin(done)]
    print("대상 10-K", len(todo), flush=True)
    for i, (cik, g) in enumerate(todo.groupby("cik"), 1):
        try:
            docs = primary_docs(cik)
        except Exception as e:  # noqa: BLE001
            print("submissions error", cik, e, flush=True); continue
        for adsh in g["adsh"]:
            doc = docs.get(adsh)
            rows = []
            if doc:
                tmp = CACHE / "tenk_tmp" / f"{adsh}.htm"
                try:
                    b = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{doc}", tmp)
                    if b:
                        rows = extract(clean_html(b.decode("utf-8", "ignore")))
                finally:
                    tmp.unlink(missing_ok=True)
            pd.DataFrame(rows or [{"kind": None, "name": None, "pct": None, "sentence": None}]).assign(adsh=adsh, cik=cik, has_doc=bool(doc)).to_parquet(OUT / f"{adsh}.parquet")
        if i % 100 == 0:
            print(i, "suppliers", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
