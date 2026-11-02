"""주요고객 관계 테이블 + 정답셋(Ground Truth) 구축.

입력: data/interim/fsn/*_{cust,sub,tot}.parquet (extract_fsn.py 산출)
산출:
  relations.parquet  공급사-고객 관계 (공시·기간·멤버·비중·분류·연결된 고객 CIK)
  gt1_comparative.parquet  비교공시 역추적: 다른 공시에서 같은 (공급사, 기간, 비중)으로 실명 공개된 익명 멤버
  gt3_named.parquet  처음부터 실명으로 공시되고 EDGAR 제출기업에 연결된 관계 (마스킹 평가용)

설계 메모 (제안 설계 대비 수정점)
  · 제출연도 == 회계연도 조건은 쓰지 않는다 (FY t 의 10-K 는 보통 t+1 에 제출) → 공시의 period(기말일)와
    팩트의 ddate 를 비교해 당기/비교기간을 가른다.
  · 비중이 같아도 다른 고객일 수 있으므로, 한 (공급사, 기간) 안에서 같은 비중의 실명 후보가 2개 이상이면 버린다.
  · 같은 공시 안의 같은 익명 라벨은 연도가 달라도 같은 고객이다 → 비교기간에서 확인된 정체를 그 공시의 당기 관계로 전파.
  · 공시마다 'Customer A' 가 가리키는 고객은 바뀔 수 있으므로 공시 경계를 넘어 라벨을 잇지 않는다.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent))
from classify_members import ANON, GENERIC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FSN = ROOT / "data" / "interim" / "fsn"
OUT = ROOT / "data" / "interim"
CONSOLIDATED_AXES = {"ConcentrationRiskByBenchmark", "ConcentrationRiskByType", "MajorCustomers"}
# 매출 기준 벤치마크 멤버 (연도·기업마다 이름이 다양: SalesRevenueNet, RevenuesNet, SalesRevenueGoodsNet, Sales …)
REV_BENCH = re.compile(r"(Revenue|^Sales)", re.I)
NOT_REV_BENCH = re.compile(r"(Receivable|Cost|Purchase|Segment|ProductLine|Expense|Asset|Deposit|Loan|Inventor)", re.I)
PCT_TAGS = {"ConcentrationRiskPercentage1", "ConcentrationRiskPercentage"}
ENTITYWIDE_PCT = "EntityWideRevenueMajorCustomerPercentage"   # 2018년 이전 택소노미: 정의상 매출 기준
ENTITYWIDE_AMT = "EntityWideRevenueMajorCustomerAmount"
REV_TAGS = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet"}
SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|company|co|llc|ltd|limited|plc|lp|l p|holdings?|group|"
                    r"national association|n a|na|sa|ag|nv|se|the|and subsidiaries|stores)\b")


def load(kind):
    parts = [pd.read_parquet(p) for p in sorted(FSN.glob(f"*_{kind}.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def spaced(member: str) -> str:
    m = re.sub(r"Member$", "", member)
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[0-9])(?=[A-Za-z])", " ", m).strip()


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("&", " and "))
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", "", s)


def parse_segments(seg: str) -> dict:
    return dict(p.split("=", 1) for p in seg.strip(";").split(";") if "=" in p)


def classify(member: str) -> str:
    base = re.sub(r"Member$", "", member)
    if ANON.search(base):
        return "anonymous"
    if GENERIC.search(base) or len(base) <= 2:
        return "generic"
    return "named"


def build_master(sub: pd.DataFrame) -> pd.DataFrame:
    """XBRL 제출기업(현재명 + 과거명) 이름 마스터. 고객 후보 = SEC 에 재무제표를 내는 기업.
    같은 이름 키에 CIK 가 여럿이면(신·구 법인 등) 가장 최근까지 공시한 CIK 를 쓴다."""
    last = sub.groupby("cik")["filed"].max().rename("last_filed")
    cur = sub[["cik", "name"]]
    former = sub[["cik", "former"]].dropna().rename(columns={"former": "name"})
    m = pd.concat([cur, former]).dropna().drop_duplicates()
    m["key"] = m["name"].map(norm_name)
    m = m[m["key"].str.len() >= 3].merge(last, left_on="cik", right_index=True)
    return m.sort_values("last_filed", ascending=False).drop_duplicates("key")


def link_names(named_members: pd.Series, master: pd.DataFrame) -> pd.DataFrame:
    """멤버 문자열 → 고객 CIK. 1) 정규화 정확일치 2) 유일 접두일치(두 단어 이상일 때만) 3) 퍼지(ratio ≥ 92, 6자 이상).
    한 단어 이름의 접두일치는 'Partner'→PartnerRe 같은 오연결을 낳으므로 쓰지 않는다."""
    by_key = master.set_index("key")["cik"]
    keys = by_key.index.to_numpy()
    out = []
    for mem in named_members.unique():
        sp = spaced(mem)
        k = norm_name(sp)
        if len(k) < 3:
            continue
        if k in by_key.index:
            out.append((mem, by_key[k], "exact", 100.0))
            continue
        if len(k) >= 6 and len(sp.split()) >= 2:
            ciks = set(by_key[[x for x in keys if x.startswith(k)]])
            if len(ciks) == 1:
                out.append((mem, ciks.pop(), "prefix", 95.0))
                continue
        hit = process.extractOne(k, keys, scorer=fuzz.ratio, score_cutoff=92)
        if hit and len(k) >= 6:
            out.append((mem, by_key[hit[0]], "fuzzy", hit[1]))
    return pd.DataFrame(out, columns=["member", "cust_cik", "link_method", "link_score"])


def main():
    cust, sub, tot = load("cust"), load("sub"), load("tot")
    sub = sub.drop_duplicates("adsh")
    cust = cust.drop_duplicates(["adsh", "tag", "ddate", "qtrs", "segments", "value"])
    print("raw cust facts", len(cust), "| filings", sub.shape[0])

    k_forms = sub[sub["form"].isin(["10-K", "10-K/A", "10-KT"])]
    cust = cust.merge(k_forms[["adsh", "cik", "name", "sic", "form", "period", "fy", "filed"]], on="adsh")
    seg = cust["segments"].map(parse_segments)
    cust["member"] = seg.map(lambda d: d.get("MajorCustomers"))
    cust["bench"] = seg.map(lambda d: d.get("ConcentrationRiskByBenchmark"))
    cust["ctype"] = seg.map(lambda d: d.get("ConcentrationRiskByType"))
    # 전사 기준 팩트를 기본으로 쓰되, 사업부 축만 추가로 붙은 팩트(예: NVIDIA "고객 A = 전체 매출의 12%, Compute & Networking
    # 부문 귀속")는 같은 공시·기간·멤버의 전사 팩트가 없을 때만 살린다 → seg_attr=True 로 표시해 강건성 분석에서 제외 가능.
    axes = seg.map(set)
    is_cons = axes.map(lambda a: a <= CONSOLIDATED_AXES)
    is_seg = axes.map(lambda a: ("BusinessSegments" in a) and (a <= CONSOLIDATED_AXES | {"BusinessSegments"}))
    cust["seg_attr"] = is_seg & ~is_cons
    cust = cust[is_cons | is_seg]
    cust["value"] = pd.to_numeric(cust["value"], errors="coerce")

    bench = cust["bench"].fillna("")
    is_rev_bench = bench.str.contains(REV_BENCH) & ~bench.str.contains(NOT_REV_BENCH)
    pct = cust[((cust["tag"].isin(PCT_TAGS) & is_rev_bench) | (cust["tag"] == ENTITYWIDE_PCT))
               & cust["ctype"].isin([None, np.nan, "CustomerConcentrationRisk"])].copy()
    pct.loc[pct["value"] > 1, "value"] = pct.loc[pct["value"] > 1, "value"] / 100  # 15.0 처럼 % 로 입력한 경우
    pct = pct[(pct["value"] > 0) & (pct["value"] <= 1)]
    amt = cust[(cust["tag"].isin(REV_TAGS) | (cust["tag"] == ENTITYWIDE_AMT)) & (cust["qtrs"] == "4") & (cust["bench"].isna())]

    rel = pct[["adsh", "cik", "name", "sic", "form", "period", "fy", "filed", "ddate", "member", "bench", "value", "seg_attr"]].rename(
        columns={"value": "pct"})
    # 같은 (공시, 기간, 멤버)에 전사 팩트가 있으면 그것만, 없으면 사업부 귀속 팩트 중 최대값
    rel = rel.sort_values(["seg_attr", "pct"], ascending=[True, False]).drop_duplicates(["adsh", "ddate", "member"])
    a = amt.drop_duplicates(["adsh", "ddate", "member"])[["adsh", "ddate", "member", "value"]].rename(columns={"value": "cust_rev"})
    rel = rel.merge(a, on=["adsh", "ddate", "member"], how="left")
    rel["current"] = rel["ddate"] == rel["period"]
    rel["cls"] = rel["member"].map(classify)

    master = build_master(sub)
    links = link_names(rel.loc[rel["cls"] == "named", "member"], master)
    rel = rel.merge(links, on="member", how="left")
    rel.loc[rel["cust_cik"] == rel["cik"], ["cust_cik", "link_method"]] = [np.nan, "self"]
    rel["cls2"] = np.where(rel["cls"] != "named", rel["cls"],
                           np.where(rel["cust_cik"].notna(), "named_linked", "named_unlinked"))

    # 공급사 연결 매출 (같은 공시, 같은 기간) → 고객 매출 추정치 Ŝ = 매출 × 비중
    t = tot[tot["tag"].isin(REV_TAGS)].copy()
    t["value"] = pd.to_numeric(t["value"], errors="coerce")
    t["prio"] = t["tag"].map({"Revenues": 0, "RevenueFromContractWithCustomerExcludingAssessedTax": 1, "SalesRevenueNet": 2,
                              "RevenueFromContractWithCustomerIncludingAssessedTax": 3, "SalesRevenueGoodsNet": 4})
    t = t.sort_values("prio").drop_duplicates(["adsh", "ddate"])[["adsh", "ddate", "value"]].rename(columns={"value": "sup_rev"})
    rel = rel.merge(t, on=["adsh", "ddate"], how="left")
    rel["est_cust_rev"] = rel["cust_rev"].fillna(rel["pct"] * rel["sup_rev"])
    rel.to_parquet(OUT / "relations.parquet")

    # ── GT3: 실명·연결된 당기 관계
    gt3 = rel[(rel["cls2"] == "named_linked") & rel["current"]].drop_duplicates(["cik", "ddate", "cust_cik"])
    gt3.to_parquet(OUT / "gt3_named.parquet")

    # ── GT1: 비교공시 역추적. 익명 팩트 (공시 F, 기간 d, 비중 p) ↔ 같은 공급사의 다른 공시 G 에서 (d, p) 로 공시된 실명·연결 팩트
    rel["p_key"] = (rel["pct"] * 10000).round().astype("Int64")  # 0.01%p 단위
    anon = rel[rel["cls"] == "anonymous"]
    named = rel[rel["cls2"] == "named_linked"]
    m = anon.merge(named[["cik", "ddate", "p_key", "adsh", "member", "cust_cik", "filed"]],
                   on=["cik", "ddate", "p_key"], suffixes=("", "_named"))
    m = m[m["adsh"] != m["adsh_named"]]
    # 같은 (공급사, 기간, 비중)에 실명 후보가 여럿이면 모호 → 제외
    amb = named.groupby(["cik", "ddate", "p_key"])["cust_cik"].nunique()
    m = m.merge(amb.rename("n_cand").reset_index(), on=["cik", "ddate", "p_key"])
    m = m[m["n_cand"] == 1]
    # 같은 공시 F 안에서 같은 라벨이 서로 다른 고객으로 확인되면 충돌 → 제외
    ident = m.groupby(["adsh", "member"])["cust_cik_named"].agg(lambda s: set(s))
    ident = ident[ident.map(len) == 1].map(lambda s: next(iter(s))).rename("gt_cust_cik").reset_index()
    gt1 = anon.merge(ident, on=["adsh", "member"])
    first_named = m.groupby(["adsh", "member"])["filed_named"].min().rename("filed_named_first").reset_index()
    gt1 = gt1.merge(first_named, on=["adsh", "member"])
    gt1["direction"] = np.where(gt1["filed"] < gt1["filed_named_first"], "revealed_later", "anonymized_later")
    gt1.to_parquet(OUT / "gt1_comparative.parquet")

    # ── 요약
    cur = rel[rel["current"]]
    print("\n== 당기 고객관계 (매출 대비 비중 공시) ==")
    print(cur["cls2"].value_counts().to_string())
    print("공급사 수", cur["cik"].nunique(), "| 공시 수", cur["adsh"].nunique())
    print("\n연결 방법:", rel.drop_duplicates("member")["link_method"].value_counts().to_dict())
    print("\n== 정답셋 ==")
    g1c = gt1[gt1["current"]]
    print("GT1 (비교공시 역추적) 당기 익명관계:", len(g1c), "| 공급사", g1c["cik"].nunique(),
          "| 방향", g1c["direction"].value_counts().to_dict())
    print("GT3 (실명 마스킹용):", len(gt3), "| 공급사", gt3["cik"].nunique(), "| 고객", gt3["cust_cik"].nunique())
    yr = cur.assign(y=cur["ddate"].str[:4]).pivot_table(index="y", columns="cls2", values="pct", aggfunc="size").fillna(0).astype(int)
    print("\n연도별 당기 관계 수\n", yr.to_string())
    yr.to_csv(ROOT / "results" / "relations_by_year.csv")


if __name__ == "__main__":
    main()
