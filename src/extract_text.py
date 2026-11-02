"""주석 데이터셋(FSN) 재다운로드 → 10-K 사업개요 텍스트 + 본사 소재지 추출 (num 은 건너뜀). zip 은 처리 후 삭제."""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "text"
OUT.mkdir(parents=True, exist_ok=True)
PAT = re.compile(r"NatureOfOperations|OrganizationConsolidation|BusinessDescription|NatureOfBusiness|DescriptionOfBusiness", re.I)
READ = dict(sep="\t", dtype=str, quoting=3, on_bad_lines="skip")
MAXLEN = 4000


def clean(s: str) -> str:
    s = re.sub(r"&[a-z]+;|&#\d+;", " ", str(s))
    return re.sub(r"\s+", " ", s).strip()[:MAXLEN]


def process(url):
    stem = url.rsplit("/", 1)[-1].replace(".zip", "")
    done = OUT / f"{stem}.done"
    if done.exists():
        return "skip"
    zpath = ROOT / "data" / "raw" / f"{stem}.zip"
    get("https://www.sec.gov" + url, zpath)
    z = zipfile.ZipFile(zpath)
    names = {n.split(".")[0]: n for n in z.namelist()}
    sub = pd.read_csv(z.open(names["sub"]), **READ)
    sub = sub[sub["form"].isin(["10-K", "10-K/A", "10-KT"])]
    keep = set(sub["adsh"])
    parts = []
    if "txt" in names:
        cols = pd.read_csv(z.open(names["txt"]), nrows=0, **READ).columns
        use = [c for c in ["adsh", "tag", "value", "dimh"] if c in cols]
        for ch in pd.read_csv(z.open(names["txt"]), chunksize=500_000, usecols=use, **READ):
            m = ch["adsh"].isin(keep) & ch["tag"].str.contains(PAT, na=False)
            if "dimh" in ch:
                m &= ch["dimh"].isna() | (ch["dimh"] == "0x00000000")
            parts.append(ch.loc[m, ["adsh", "tag", "value"]])
    txt = pd.concat(parts) if parts else pd.DataFrame(columns=["adsh", "tag", "value"])
    txt["value"] = txt["value"].map(clean)
    txt = txt.sort_values("value", key=lambda s: s.str.len(), ascending=False).drop_duplicates("adsh")
    loc = sub[[c for c in ["adsh", "cik", "name", "sic", "cityba", "stprba", "countryba", "period", "filed", "form"] if c in sub.columns]]
    txt.merge(loc, on="adsh", how="right").to_parquet(OUT / f"{stem}.parquet")
    z.close()
    if stem != "2024q1_notes":
        zpath.unlink()
    done.touch()
    return f"10-K {len(sub)} | with text {txt['adsh'].nunique()}"


def main():
    urls = [u.strip() for u in (ROOT / "data" / "fsn_urls.txt").read_text().split() if u.strip()]
    for i, u in enumerate(reversed(urls), 1):
        try:
            print(f"[{i}/{len(urls)}] {u.rsplit('/', 1)[-1]}: {process(u)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(urls)}] {u.rsplit('/', 1)[-1]}: ERROR {e!r}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
