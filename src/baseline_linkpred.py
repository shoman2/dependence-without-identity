"""GT3(실명 고객 마스킹) 기준선: 문제 난이도 측정.

각 실명·연결 관계 (공급사 i, 연도 t, 추정 매출 s)에 대해 t년 후보 풀(재무 데이터 있는 SEC 제출기업)에서
진짜 고객을 몇 위에 놓는지 평가. 학습 정보는 t-1년까지 공개된 관계만 사용(시간 순서 준수).
  B0 무작위 / B1 규모순(매입 여력 큰 순) / B2 회계 상한 제약 + 규모순
  B3 산업 쌍 사전확률 P(고객 SIC2 | 공급사 SIC2) × 규모 + 제약
  B4 B3 + 과거에 이 공급사가 실명 공시한 고객이면 가산 (관계 지속성)
지표: Hit@1/10/50, MRR, 후보 풀 크기.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INT, RES = ROOT / "data" / "interim", ROOT / "results"
THETA = 0.5


def fyear(d):
    d = d.astype(str)
    return d.str[:4].astype(int) - (d.str[4:6].astype(int) < 7).astype(int)


def main():
    gt3 = pd.read_parquet(INT / "gt3_named.parquet")
    rel = pd.read_parquet(INT / "relations.parquet")
    fund = pd.read_parquet(INT / "fundamentals.parquet")
    sub_sic = rel.drop_duplicates("cik")[["cik", "sic"]]
    fund["purch"] = fund["cogs"].fillna(fund["rev"])
    sic = pd.concat([pd.read_parquet(p, columns=["cik", "sic"]) for p in sorted((INT / "fsn").glob("*_sub.parquet"))])
    sic = sic.dropna().drop_duplicates("cik", keep="last")
    sic["cik"] = sic["cik"].astype(int)
    sic["sic2"] = sic["sic"].astype(str).str[:2]
    fund = fund.merge(sic[["cik", "sic2"]], on="cik", how="left")

    g = gt3.assign(year=fyear(gt3["ddate"]), cik=gt3["cik"].astype(int), cust=gt3["cust_cik"].astype(int))
    g = g[g["est_cust_rev"].notna()].merge(sic[["cik", "sic2"]], on="cik", how="left")
    edges = g[["cik", "cust", "year", "sic2"]].merge(sic[["cik", "sic2"]].rename(columns={"cik": "cust", "sic2": "csic2"}), on="cust", how="left")

    out = []
    rng = np.random.default_rng(0)
    for y, gy in g.groupby("year"):
        if y < 2014:
            continue
        pool = fund[(fund["year"] == y) & fund["purch"].notna() & (fund["purch"] > 0)][["cik", "purch", "sic2"]].reset_index(drop=True)
        if len(pool) == 0:
            continue
        past = edges[edges["year"] < y]
        pri = (past.groupby(["sic2", "csic2"]).size() + 1).rename("n").reset_index()
        tot = pri.groupby("sic2")["n"].transform("sum")
        pri["p"] = pri["n"] / tot
        past_pairs = set(zip(past["cik"], past["cust"]))
        logp = np.log(pool["purch"].to_numpy())
        for r in gy.itertuples():
            if r.cust not in set(pool["cik"]):
                continue
            ok = (pool["purch"].to_numpy() * THETA >= r.est_cust_rev) & (pool["cik"].to_numpy() != r.cik)
            p_ind = pool["sic2"].map(dict(zip(pri.loc[pri["sic2"] == r.sic2, "csic2"], pri.loc[pri["sic2"] == r.sic2, "p"]))).fillna(1e-4).to_numpy()
            hist = np.array([(r.cik, c) in past_pairs for c in pool["cik"]], dtype=float)
            scores = {
                "B0 random": rng.random(len(pool)),
                "B1 size": logp,
                "B2 constraint+size": np.where(ok, logp, -np.inf),
                "B3 +industry prior": np.where(ok, np.log(p_ind) + 0.3 * logp, -np.inf),
                "B4 +relationship history": np.where(ok, np.log(p_ind) + 0.3 * logp + 10 * hist, -np.inf),
            }
            ti = int(np.where(pool["cik"].to_numpy() == r.cust)[0][0])
            for name, s in scores.items():
                rank = 1 + int((s > s[ti]).sum()) if np.isfinite(s[ti]) else len(pool)
                out.append({"year": y, "model": name, "rank": rank, "pool": int(np.isfinite(s).sum()),
                            "had_history": bool(hist[ti])})
    res = pd.DataFrame(out)
    res.to_parquet(RES / "baseline_linkpred.parquet")
    summ = res.groupby("model").agg(n=("rank", "size"), hit1=("rank", lambda r: (r <= 1).mean()),
                                    hit10=("rank", lambda r: (r <= 10).mean()), hit50=("rank", lambda r: (r <= 50).mean()),
                                    mrr=("rank", lambda r: (1 / r).mean()), median_rank=("rank", "median"),
                                    pool=("pool", "median"))
    print("== GT3 기준선 (2014~, t-1년까지 정보만 사용) ==")
    print(summ.round(3).to_string())
    nh = res[(res["model"] == "B4 +relationship history")]
    print("\n과거 실명 관계가 있던 비율:", round(nh["had_history"].mean(), 3))
    print("관계 이력 없는(신규) 고객만 B3:",
          res[(res["model"] == "B3 +industry prior") & ~res["had_history"]].agg(
              hit10=("rank", lambda r: (r <= 10).mean()), mrr=("rank", lambda r: (1 / r).mean())).round(3).to_dict())
    summ.to_csv(RES / "baseline_linkpred_summary.csv")


if __name__ == "__main__":
    main()
