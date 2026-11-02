"""전체 기간: (1) 주석상 익명 공시의 본문 실명 공개율 추이 (2) GT2 = 주석 익명 비중 ≈ 본문 실명 비중."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_gt import build_master, load, norm_name, spaced  # noqa: E402
from text_customers import is_customer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"


def main():
    master = build_master(load("sub").drop_duplicates("adsh"))
    keymap = master.set_index("key")["cik"]
    rel = pd.read_parquet(INT / "relations.parquet")
    rel = rel[rel["current"]].copy()
    rel["fy"] = rel["ddate"].astype(str).str[:4].astype(int) - (rel["ddate"].astype(str).str[4:6].astype(int) < 7)
    rel["cik"] = rel["cik"].astype(int)
    fname = rel.drop_duplicates("adsh").set_index("adsh")["name"]
    t = pd.concat([pd.read_parquet(p) for p in (INT / "tenk_text").glob("*.parquet")], ignore_index=True)
    docs = t.drop_duplicates("adsh")[["adsh", "has_doc"]]
    t = t.dropna(subset=["name"]).rename(columns={"name": "cust_name"})

    def link(n):
        for k in (norm_name(n), norm_name(spaced(n.replace(" ", "")))):
            if k in keymap.index:
                return int(keymap[k])
        return None
    t["cust_cik"] = t["cust_name"].map(link)
    t["keep"] = [is_customer(r.cust_name, r.sentence, fname.get(r.adsh, ""), r.cust_cik is not None) for r in t.itertuples()]
    t.to_parquet(INT / "text_customers_all.parquet")
    tk = t[t["keep"]]

    an = rel[rel["cls"] == "anonymous"]
    per = rel[rel["adsh"].isin(set(an["adsh"]))].groupby("adsh").agg(fy=("fy", "first"), cik=("cik", "first"),
                                                                        anon_only=("cls", lambda s: not (s == "named").any()))
    per = per.merge(docs, left_index=True, right_on="adsh")
    per = per[per["has_doc"]]
    per["text_named"] = per["adsh"].isin(set(tk["adsh"]))
    trend = per[per["anon_only"]].groupby("fy").agg(filings=("adsh", "size"), text_named_rate=("text_named", "mean"))
    print("== 주석상 '모든 주요고객 익명' 10-K 중 본문에 고객 실명+비중이 있는 비율 ==")
    print(trend.loc[2012:2025].round(3).to_string())
    print("전체:", round(per.loc[per["anon_only"], "text_named"].mean(), 3), "| N", int(per["anon_only"].sum()))

    m = an.merge(tk[["adsh", "cust_name", "cust_cik", "pct", "sentence"]], on="adsh", suffixes=("_note", "_text"))
    m = m[(m["pct_note"] - m["pct_text"]).abs() <= 0.005]
    k = m.groupby(["adsh", "member"])["cust_name"].nunique().rename("k").reset_index()
    m = m.merge(k, on=["adsh", "member"])
    m = m[m["k"] == 1].drop_duplicates(["adsh", "member"])
    m.to_parquet(INT / "gt2_text.parquet")
    print(f"\nGT2 (주석 익명 ↔ 본문 실명, 비중 ±0.5%p 일치): {len(m)}건 | 공급사 {m['cik'].nunique()} | EDGAR 연결 {m['cust_cik_text'].notna().sum()}")
    print(m.groupby("fy").size().loc[2012:2025].to_dict())
    trend.to_csv(RES / "text_naming_trend.csv")
    m.sample(min(100, len(m)), random_state=11)[["name", "member", "pct_note", "cust_name", "pct_text", "sentence"]].to_csv(RES / "audit_gt2_sample.csv", index=False)


if __name__ == "__main__":
    main()
