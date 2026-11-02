"""FY2021(ChatGPT 이전) 10-K 본문 Item 1 → AI 인프라 키워드 강도. 대상: DiD 패널 공급사."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import CACHE, get, get_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
TERMS = {
    "ai": r"artificial intelligence|\bAI\b|machine learning|generative AI|large language model",
    "dc": r"data cent(?:er|re)s?|hyperscale\w*|cloud service providers?|colocation",
    "compute": r"\bGPUs?\b|graphics processing units?|accelerators?|high[- ]performance computing|\bHPC\b|servers?\b",
    "infra": r"liquid cooling|thermal management|power distribution|optical (?:transceivers?|interconnects?)|switch(?:es|ing)? (?:silicon|chips?)",
}


def item1(html: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = re.sub(r"&#\d+;|&[a-z]+;", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    ms = [m.start() for m in re.finditer(r"item\s*1\.?\s*business", txt, re.I)]
    if not ms:
        return txt[:60000]
    start = ms[1] if len(ms) > 1 else ms[0]  # 첫 등장은 보통 목차
    end = re.search(r"item\s*1a\.?\s*risk\s*factors", txt[start + 200:], re.I)
    return txt[start: start + 200 + end.start()] if end else txt[start: start + 80000]


def main():
    p = pd.read_parquet(INT / "did_panel.parquet")
    ciks = sorted(p["cik"].unique())
    rows = []
    for i, cik in enumerate(ciks, 1):
        try:
            js = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", CACHE / "submissions" / f"{cik}.json")
            rec = js["filings"]["recent"]
            cand = [(rd, a, d) for f, a, rd, d in zip(rec["form"], rec["accessionNumber"], rec["reportDate"], rec["primaryDocument"])
                    if f == "10-K" and "2021-07-01" <= rd <= "2022-06-30"]
            if not cand:
                rows.append({"cik": cik, "status": "no_fy2021_10k"}); continue
            rd, acc, doc = sorted(cand)[-1]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
            b = get(url, CACHE / "tenk" / f"{cik}_{acc}.htm")
            t = item1(b.decode("utf-8", "ignore"))
            words = max(len(t.split()), 1)
            r = {"cik": cik, "status": "ok", "report_date": rd, "words": words}
            for k, pat in TERMS.items():
                r[k] = len(re.findall(pat, t, re.I)) / words * 10000
            rows.append(r)
        except Exception as e:  # noqa: BLE001
            rows.append({"cik": cik, "status": f"error:{e}"})
        if i % 200 == 0:
            print(i, len(ciks), flush=True)
    pd.DataFrame(rows).to_parquet(INT / "item1_exposure.parquet")
    print("done", pd.DataFrame(rows)["status"].str.split(":").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
