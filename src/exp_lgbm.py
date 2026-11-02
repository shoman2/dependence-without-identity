"""기준선(B1~B4) + LightGBM 랭커(그래프 휴리스틱 특성). torch 미사용 프로세스.
평가 연도 t: 학습 = t 이전 연도들의 GT3 질의(각자 자기 시점 맥락), 평가 = t 년 GT3·GT1 질의(후보 전체 순위)."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import NODE_COLS, PAIR_COLS, context, pair_features, queries, text_context, text_pair_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
YEARS = range(2015, 2026)
# 절제: 그래프 유래 특성(이력·고객차수·산업내차수·2-hop CF) 제거 → "그래프 정보 없음"
NOGRAPH = "--nograph" in sys.argv
TEXT = "--text" in sys.argv  # 제품 유사도(사업개요) 특성 추가
GRAPH_COLS = [PAIR_COLS.index(c) for c in ["history", "cust_deg", "cust_deg_ind", "cf2hop"]]
N_NEG = 99
_train_cache = {}


def node_block(ctx, idx):
    return ctx["pool"][NODE_COLS].to_numpy(np.float32)[idx]


def train_groups(s):
    """연도 s 의 학습 그룹: 질의마다 정답 1 + 제약 통과 후보 중 무작위 음성 99."""
    if s in _train_cache:
        return _train_cache[s]
    ctx, rng = context(s), np.random.default_rng(s)
    pool_ciks = ctx["pool"]["cik"].to_numpy()
    X, y, g = [], [], []
    for q in queries("gt3", s).itertuples():
        pos = np.where(pool_ciks == q.cust)[0]
        if len(pos) == 0:
            continue
        F, ok, _ = pair_features(ctx, q.cik, q.est_cust_rev)
        negs = np.setdiff1d(np.where(ok)[0], pos)
        if len(negs) < 10:
            continue
        idx = np.r_[pos, rng.choice(negs, min(N_NEG, len(negs)), replace=False)]
        if TEXT:
            F = F.copy(); Ft = text_pair_features(ctx, text_context(s), q.cik, pool_ciks[idx])
        X.append(np.c_[F[idx], *( [Ft] if TEXT else [] ), node_block(ctx, idx), ctx["pool"]["sic2"].to_numpy()[idx], np.full(len(idx), ctx["sic2"].get(q.cik, 0))])
        y.append(np.r_[1, np.zeros(len(idx) - 1)])
        g.append(len(idx))
    out = (np.vstack(X), np.concatenate(y), g) if X else None
    _train_cache[s] = out
    return out


def rank_of(score, ok, ti):
    s = np.where(ok, score, -np.inf)
    return 1 + int((s > s[ti]).sum()) if np.isfinite(s[ti]) else int(ok.sum()) + 1


def main():
    rows = []
    for t in YEARS:
        parts = [train_groups(s) for s in range(2012, t)]
        parts = [p for p in parts if p is not None]
        Xtr = np.vstack([p[0] for p in parts]); ytr = np.concatenate([p[1] for p in parts])
        gtr = sum((p[2] for p in parts), [])
        ncol = Xtr.shape[1]
        ranker = lgb.LGBMRanker(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        keep_cols = [i for i in range(ncol) if not (NOGRAPH and i in GRAPH_COLS)]
        cat = [keep_cols.index(ncol - 2), keep_cols.index(ncol - 1)]
        ranker.fit(Xtr[:, keep_cols], ytr, group=gtr, categorical_feature=cat)
        ctx = context(t)
        pool_ciks = ctx["pool"]["cik"].to_numpy()
        allidx = np.arange(len(pool_ciks))
        for kind in ["gt3", "gt1"]:
            for q in queries(kind, t).itertuples():
                hit = np.where(pool_ciks == q.cust)[0]
                if len(hit) == 0:
                    continue
                ti = int(hit[0])
                F, ok, _ = pair_features(ctx, q.cik, q.est_cust_rev)
                Ft = [text_pair_features(ctx, text_context(t), q.cik, pool_ciks)] if TEXT else []
                Xte = np.c_[F, *Ft, node_block(ctx, allidx), ctx["pool"]["sic2"].to_numpy(), np.full(len(allidx), ctx["sic2"].get(q.cik, 0))]
                lp, li, hist = F[:, PAIR_COLS.index("cand_log_purch")], F[:, 0], F[:, 2]
                scores = {"B1 size": lp, "B3 industry+size": li + 0.3 * lp, "B4 +history": li + 0.3 * lp + 10 * hist,
                          "LGBM ranker": ranker.predict(Xte[:, keep_cols])}
                if NOGRAPH:
                    scores = {"LGBM ranker (no graph features)": scores["LGBM ranker"]}
                if TEXT:
                    scores = {"LGBM ranker + product similarity": scores["LGBM ranker"]}
                track = "gt3" if kind == "gt3" else ("gt1a" if q.direction == "revealed_later" else "gt1b")
                for m, sc in scores.items():
                    rows.append({"year": t, "track": track, "model": m, "cik": q.cik, "cust": q.cust,
                                 "rank": rank_of(sc, ok, ti), "pool": int(ok.sum()), "history": bool(hist[ti]),
                                 "true_ok": bool(ok[ti])})
        print("lgbm", t, len(gtr), flush=True)
    name = "ranks_lgbm_nograph.parquet" if NOGRAPH else ("ranks_lgbm_text.parquet" if TEXT else "ranks_lgbm.parquet")
    pd.DataFrame(rows).to_parquet(RES / name)


if __name__ == "__main__":
    main()
