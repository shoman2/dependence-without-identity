"""링크 예측 공통 특성 (pandas/scipy 만 사용 — torch·lightgbm 은 각 실험 스크립트에서 따로 import).

연도 s 의 '예측 시점 맥락' = s-1 년 재무(후보 특성) + s 년 이전에 공개된 실명 관계(그래프·이력·사전확률).
같은 함수로 학습 예제(과거 연도)와 평가 질의(평가 연도)를 만들어 시점 일관성을 보장한다.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
THETA = 0.5
NODE_COLS = ["log_at", "log_purch", "log_rev", "roa", "gm", "rnd_sales", "cogs_rev", "miss_rev", "miss_cogs"]
PAIR_COLS = ["log_ind_prior", "cap_ratio", "history", "cust_deg", "cust_deg_ind", "cf2hop", "cand_log_purch"]


def fyear(d):
    d = pd.Series(d).astype(str)
    return (d.str[:4].astype(int) - (d.str[4:6].astype(int) < 7).astype(int)).to_numpy()


@lru_cache(maxsize=1)
def base_tables():
    fund = pd.read_parquet(INT / "fundamentals.parquet")
    sic = pd.concat([pd.read_parquet(p, columns=["cik", "sic"]) for p in sorted((INT / "fsn").glob("*_sub.parquet"))])
    sic = sic.dropna().drop_duplicates("cik", keep="last")
    sic["cik"] = sic["cik"].astype(int)
    sic["sic2"] = pd.to_numeric(sic["sic"].astype(str).str[:2], errors="coerce").fillna(0).astype(int)
    fund = fund.merge(sic[["cik", "sic2"]], on="cik", how="left")
    fund["sic2"] = fund["sic2"].fillna(0).astype(int)
    fund["purch"] = fund["cogs"].fillna(fund["rev"])
    fund["log_at"] = np.log1p(fund["at"].clip(lower=0))
    fund["log_purch"] = np.log1p(fund["purch"].clip(lower=0))
    fund["log_rev"] = np.log1p(fund["rev"].clip(lower=0))
    fund["roa"] = (fund["ni"] / fund["at"]).clip(-2, 2)
    fund["gm"] = (1 - fund["cogs"] / fund["rev"]).where(fund["rev"] > 0).clip(-2, 1)
    fund["rnd_sales"] = (fund["rnd"].fillna(0) / fund["rev"]).where(fund["rev"] > 0).clip(0, 5)
    fund["cogs_rev"] = (fund["cogs"] / fund["rev"]).where(fund["rev"] > 0).clip(0, 3)
    fund["miss_rev"] = fund["rev"].isna().astype(float)
    fund["miss_cogs"] = fund["cogs"].isna().astype(float)
    rel = pd.read_parquet(INT / "relations.parquet")
    rel = rel[rel["current"]].copy()
    rel["year"] = fyear(rel["ddate"])
    rel["cik"] = rel["cik"].astype(int)
    named = rel[(rel["cls2"] == "named_linked") & rel["est_cust_rev"].notna()].copy()
    named["cust"] = named["cust_cik"].astype(int)
    named = named.drop_duplicates(["cik", "cust", "year"])
    gt1 = pd.read_parquet(INT / "gt1_comparative.parquet")
    gt1 = gt1[gt1["current"] & gt1["est_cust_rev"].notna()].copy()
    gt1["year"] = fyear(gt1["ddate"])
    gt1["cik"] = gt1["cik"].astype(int)
    gt1["cust"] = gt1["gt_cust_cik"].astype(int)
    return fund, sic, named, gt1


@lru_cache(maxsize=32)
def context(s: int):
    """연도 s 질의에 쓸 맥락 (s-1 재무, s 이전 관계)."""
    fund, sic, named, _ = base_tables()
    pool = fund[(fund["year"] == s - 1) & (fund["purch"] > 0)].copy()
    pool = pool.drop_duplicates("cik").reset_index(drop=True)
    pool[NODE_COLS] = pool[NODE_COLS].fillna(0.0)
    prior = named[named["year"] < s]
    sic2 = dict(zip(sic["cik"], sic["sic2"]))
    prior = prior.assign(ssic=prior["cik"].map(sic2).fillna(0).astype(int), csic=prior["cust"].map(sic2).fillna(0).astype(int))
    # 산업 쌍 사전확률 P(고객 SIC2 | 공급사 SIC2), 라플라스 평활
    cnt = prior.drop_duplicates(["cik", "cust"]).groupby(["ssic", "csic"]).size()
    ind = {}
    for (a, b), n in cnt.items():
        ind.setdefault(a, {})[b] = n
    # 이분 인접행렬 (공급사 × 고객), 고유 쌍
    pairs = prior.drop_duplicates(["cik", "cust"])
    sup_ids = {c: i for i, c in enumerate(sorted(set(pairs["cik"])))}
    cus_ids = {c: i for i, c in enumerate(sorted(set(pairs["cust"]) | set(pool["cik"])))}
    A = sp.csr_matrix((np.ones(len(pairs)), ([sup_ids[c] for c in pairs["cik"]], [cus_ids[c] for c in pairs["cust"]])),
                      shape=(len(sup_ids), len(cus_ids)))
    deg = np.asarray(A.sum(0)).ravel()
    return {"s": s, "pool": pool, "prior": prior, "ind": ind, "sic2": sic2, "A": A, "sup_ids": sup_ids,
            "cus_ids": cus_ids, "deg": deg, "hist": set(zip(pairs["cik"], pairs["cust"]))}


def pair_features(ctx, sup: int, est_rev: float, cand: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """공급사 sup 의 한 질의에 대해 후보 전체(또는 cand 인덱스)의 쌍 특성. 반환: (특성행렬, 제약통과, 후보 cik)."""
    pool = ctx["pool"]
    idx = np.arange(len(pool)) if cand is None else cand
    ciks = pool["cik"].to_numpy()[idx]
    purch = pool["purch"].to_numpy()[idx]
    ssic = ctx["sic2"].get(sup, 0)
    row = ctx["ind"].get(ssic, {})
    tot = sum(row.values()) + 100.0
    csic = pool["sic2"].to_numpy()[idx]
    log_ind = np.log((np.array([row.get(c, 0) for c in csic]) + 1.0) / tot)
    cap = np.log1p(est_rev) - np.log1p(purch)
    hist = np.array([(sup, c) in ctx["hist"] for c in ciks], dtype=float)
    cid = np.array([ctx["cus_ids"].get(c, -1) for c in ciks])
    deg = np.where(cid >= 0, ctx["deg"][np.maximum(cid, 0)], 0.0)
    # 같은 산업 공급사들로부터의 고객 차수, 2-hop 협업필터링 점수
    A, sup_ids = ctx["A"], ctx["sup_ids"]
    same_ind = [sup_ids[c] for c in ctx["prior"].loc[ctx["prior"]["ssic"] == ssic, "cik"].unique() if c != sup]
    deg_ind = np.asarray(A[same_ind].sum(0)).ravel() if same_ind else np.zeros(A.shape[1])
    if sup in sup_ids:
        r = A[sup_ids[sup]]
        sim = (A @ r.T).toarray().ravel()
        sim[sup_ids[sup]] = 0
        cf = np.asarray(A.T @ sim).ravel()
    else:
        cf = np.zeros(A.shape[1])
    take = lambda v: np.where(cid >= 0, v[np.maximum(cid, 0)], 0.0)  # noqa: E731
    X = np.c_[log_ind, cap, hist, np.log1p(deg), np.log1p(take(deg_ind)), np.log1p(take(cf)), np.log1p(purch)]
    ok = (purch * THETA >= est_rev) & (ciks != sup)
    return X.astype(np.float32), ok, ciks


def queries(kind: str, s: int) -> pd.DataFrame:
    """평가/학습 질의: kind='gt3' (실명·연결, 해당 연도) / 'gt1' (비교공시 정답 익명)."""
    _, _, named, gt1 = base_tables()
    q = named[named["year"] == s] if kind == "gt3" else gt1[gt1["year"] == s]
    q = q.drop_duplicates(["cik", "cust"])
    cols = ["cik", "cust", "year", "est_cust_rev"] + (["direction"] if kind == "gt1" else [])
    return q[cols].reset_index(drop=True)


def size_tercile_map():
    fund, _, named, _ = base_tables()
    f = fund[["cik", "year", "log_at"]]
    return f


# ───────────────────────── 제품 유사도(사업개요 텍스트) 그래프 ─────────────────────────
TEXT_PAIR_COLS = ["text_sim", "peer_cf", "peer_cf_n"]
BOILER = ("company companies inc corporation corp llc ltd subsidiaries subsidiary consolidated financial statements basis "
          "presentation accounting principles generally accepted gaap united states note notes organization nature "
          "operations business description fiscal year years december june september march incorporated delaware "
          "nevada headquartered located principal accompanying include includes including wholly owned "
          "going concern reverse merger shell stockholders stockholder shareholders shares share common stock preferred "
          "liquidity capital net loss losses losses incurred deficit accumulated raise additional funding management plans "
          "substantial doubt ability continue acquisition acquired agreement exchange issued issuance private placement "
          "inception development stage formerly name changed effective date reporting period interim unaudited "
          "estimates assumptions reclassifications certain prior amounts conformity significant policies").split()
K_TEXT = 20


@lru_cache(maxsize=1)
def text_table():
    t = pd.concat([pd.read_parquet(p) for p in sorted((INT / "text").glob("*.parquet"))], ignore_index=True)
    t = t.dropna(subset=["value"]).drop_duplicates("adsh")
    t["cik"] = t["cik"].astype(int)
    t["fy"] = fyear(t["period"])
    return t.sort_values("filed").drop_duplicates(["cik", "fy"], keep="last")


@lru_cache(maxsize=32)
def text_context(s: int):
    """연도 s 질의용: s-1 회계연도 10-K 사업개요 문장 임베딩(embed_text.py 산출) → 기업별 상위 20 유사기업."""
    z = np.load(INT / "emb" / f"fy{s - 1}.npz")
    ciks, M = z["ciks"], z["E"]
    S = M @ M.T
    np.fill_diagonal(S, -1.0)
    top = np.argpartition(-S, K_TEXT, axis=1)[:, :K_TEXT]
    nbr = {}
    for i, c in enumerate(ciks):
        o = top[i][np.argsort(-S[i, top[i]])]
        nbr[c] = (ciks[o], S[i, o])
    return {"ciks": ciks, "idx": {c: i for i, c in enumerate(ciks)}, "M": M, "nbr": nbr}


def text_pair_features(ctx, tctx, sup: int, cand_ciks: np.ndarray) -> np.ndarray:
    """text_sim: 공급사-후보 사업개요 유사도 / peer_cf: 제품이 비슷한 공급사들(상위 20)의 과거 실명 고객에 후보가 있는 정도."""
    M, idx = tctx["M"], tctx["idx"]
    ci = np.array([idx.get(c, -1) for c in cand_ciks])
    if sup in idx:
        v = M[idx[sup]]
        sims = M[np.maximum(ci, 0)] @ v
        tsim = np.where(ci >= 0, sims, 0.0)
    else:
        tsim = np.zeros(len(cand_ciks))
    pcf, pn = np.zeros(len(cand_ciks)), np.zeros(len(cand_ciks))
    if sup in tctx["nbr"]:
        peers, w = tctx["nbr"][sup]
        A, sup_ids, cus_ids = ctx["A"], ctx["sup_ids"], ctx["cus_ids"]
        rows = [(sup_ids[p], ww) for p, ww in zip(peers, w) if p in sup_ids and p != sup]
        if rows:
            vec = np.zeros(A.shape[0])
            for r, ww in rows:
                vec[r] = ww
            sc = np.asarray(A.T @ vec).ravel()
            cnt = np.asarray(A.T @ (vec > 0).astype(float)).ravel()
            cid = np.array([cus_ids.get(c, -1) for c in cand_ciks])
            pcf = np.where(cid >= 0, sc[np.maximum(cid, 0)], 0.0)
            pn = np.where(cid >= 0, cnt[np.maximum(cid, 0)], 0.0)
    return np.c_[tsim, pcf, np.log1p(pn)].astype(np.float32)
