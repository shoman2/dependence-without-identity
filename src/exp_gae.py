"""GAE (방향성 GraphSAGE 인코더 + 순위학습 디코더) vs 같은 디코더·메시지패싱 없음. lightgbm 미사용 프로세스.

평가 연도 t:
  그래프 = t 이전 공개 실명 관계(공급사→고객), 노드 특성 = t-1 재무 + SIC2 임베딩
  인코더 = h' = W·h + SAGE(고객 쪽 이웃) + SAGE(공급사 쪽 이웃)  (방향을 구분)
  학습 = t 이전 연도 GT3 질의: 정답 1 + 음성 99 에 대한 softmax 순위 손실.
         질의 쌍의 엣지는 메시지 전달에서 제거(타깃 누수 방지)
  디코더 = MLP([z_s, z_c, z_s⊙z_c, 쌍 특성])   — 쌍 특성은 LGBM 과 동일
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(Path(__file__).parent))
from features import NODE_COLS, base_tables, context, pair_features, queries, text_context, text_pair_features, text_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEXT = "--text" in sys.argv
YEARS = [int(y) for a in sys.argv[1:] if not a.startswith("--") for y in a.split()] or list(range(2015, 2026))
K_EDGE = 10
SEEDS = [0, 1, 2]
N_NEG = 99
torch.set_num_threads(4)


class Model(nn.Module):
    def __init__(self, n_feat, n_pair, d=64, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.sic = nn.Embedding(100, 16)
        self.inp = nn.Linear(n_feat + 16, d)
        self.out_conv = nn.ModuleList(SAGEConv(d, d) for _ in range(2))
        self.in_conv = nn.ModuleList(SAGEConv(d, d) for _ in range(2))
        self.self_lin = nn.ModuleList(nn.Linear(d, d) for _ in range(2))
        self.text_conv = nn.ModuleList(SAGEConv(d, d) for _ in range(2))
        self.city_conv = nn.ModuleList(SAGEConv(d, d) for _ in range(2))
        self.dec = nn.Sequential(nn.Linear(3 * d + n_pair, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def encode(self, x, sic, ei, et=None, ec=None):
        h = torch.relu(self.inp(torch.cat([x, self.sic(sic)], 1)))
        for oc, ic, sl, tc, cc in zip(self.out_conv, self.in_conv, self.self_lin, self.text_conv, self.city_conv):
            if self.use_graph:
                m = sl(h)
                if ei.shape[1] > 0:
                    m = m + oc(h, ei) + ic(h, ei.flip(0))
                if et is not None and et.shape[1] > 0:
                    m = m + tc(h, et)
                if ec is not None and ec.shape[1] > 0:
                    m = m + cc(h, ec)
                h = torch.relu(m)
            else:
                h = torch.relu(sl(h))
        return h

    def score(self, z, s_idx, c_idx, pf):
        zs, zc = z[s_idx], z[c_idx]
        return self.dec(torch.cat([zs, zc, zs * zc, pf], -1)).squeeze(-1)


def build_year(t):
    fund, sic, _, _ = base_tables()
    ctx = context(t)
    f = fund[fund["year"] == t - 1].drop_duplicates("cik").set_index("cik")
    prior = ctx["prior"].drop_duplicates(["cik", "cust"])
    nodes = sorted(set(ctx["pool"]["cik"]) | set(prior["cik"]) | set(prior["cust"]))
    nid = {c: i for i, c in enumerate(nodes)}
    X = f.reindex(nodes)[NODE_COLS].fillna(0.0).to_numpy(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    sic2 = np.array([ctx["sic2"].get(c, 0) for c in nodes]) % 100
    E = np.array([[nid[a], nid[b]] for a, b in zip(prior["cik"], prior["cust"])], dtype=np.int64).T.reshape(2, -1)
    ET = EC = np.zeros((2, 0), dtype=np.int64)
    if TEXT:
        tctx = text_context(t)
        et = [(nid[a], nid[b]) for a, (nb, _) in tctx["nbr"].items() if a in nid for b in nb[:K_EDGE] if b in nid]
        ET = np.array(et, dtype=np.int64).T.reshape(2, -1)
        tt = text_table()
        loc = tt[(tt["fy"] == t - 1) & tt["cik"].isin(nid)].dropna(subset=["cityba"])
        rng = np.random.default_rng(t)
        ec = []
        for _, g in loc.groupby([loc["cityba"].str.upper(), loc["stprba"]]):
            cs = g["cik"].to_numpy()
            for c in cs:
                others = cs[cs != c]
                for o in rng.choice(others, min(K_EDGE, len(others)), replace=False) if len(others) else []:
                    ec.append((nid[c], nid[o]))
        EC = np.array(ec, dtype=np.int64).T.reshape(2, -1)
    return ctx, nid, torch.tensor(X), torch.tensor(sic2), torch.tensor(E), torch.tensor(ET), torch.tensor(EC)


def train_examples(s, nid):
    ctx, rng = context(s), np.random.default_rng(s)
    pool_ciks = ctx["pool"]["cik"].to_numpy()
    ex = []
    for q in queries("gt3", s).itertuples():
        pos = np.where(pool_ciks == q.cust)[0]
        if len(pos) == 0 or q.cik not in nid:
            continue
        F, ok, ciks = pair_features(ctx, q.cik, q.est_cust_rev)
        negs = np.setdiff1d(np.where(ok)[0], pos)
        negs = np.array([n for n in negs if ciks[n] in nid])
        if len(negs) < 10 or ciks[pos[0]] not in nid:
            continue
        idx = np.r_[pos[0], rng.choice(negs, min(N_NEG, len(negs)), replace=False)]
        Fi = F[idx]
        if TEXT:
            Fi = np.c_[Fi, text_pair_features(ctx, text_context(s), q.cik, ciks[idx])]
        ex.append((nid[q.cik], np.array([nid[c] for c in ciks[idx]]), Fi, (q.cik, q.cust)))
    return ex


def fit(t, use_graph, seed, data, ex, pmu, psd):
    ctx, nid, X, S, E, ET, EC = data
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Model(X.shape[1], ex[0][2].shape[1], use_graph=use_graph)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pair_of_edge = list(zip(E[0].tolist(), E[1].tolist()))
    for _ in range(25):
        model.train()
        for b in np.array_split(rng.permutation(len(ex)), max(1, len(ex) // 64)):
            drop = {(ex[i][0], ex[i][1][0]) for i in b}
            keep = torch.tensor([p not in drop for p in pair_of_edge], dtype=torch.bool)
            z = model.encode(X, S, E[:, keep] if len(keep) else E, ET, EC)
            s_idx = torch.tensor(np.concatenate([[ex[i][0]] * len(ex[i][1]) for i in b]))
            c_idx = torch.tensor(np.concatenate([ex[i][1] for i in b]))
            pf = torch.tensor((np.vstack([ex[i][2] for i in b]) - pmu) / psd)
            sc = model.score(z, s_idx, c_idx, pf)
            loss, o = 0.0, 0
            for i in b:
                n = len(ex[i][1])
                loss = loss + torch.nn.functional.cross_entropy(sc[o:o + n].unsqueeze(0), torch.zeros(1, dtype=torch.long))
                o += n
            opt.zero_grad()
            (loss / len(b)).backward()
            opt.step()
    model.eval()
    return model


def main():
    rows = []
    for t in YEARS:
        data = build_year(t)
        ctx, nid, X, S, E, ET, EC = data
        ex = sum((train_examples(s, nid) for s in range(2012, t)), [])
        allpf = np.vstack([e[2] for e in ex])
        pmu, psd = allpf.mean(0), allpf.std(0) + 1e-6
        names = ([("GAE + product-similarity & location graph", True), ("Decoder only + product-similarity features", False)]
                 if TEXT else [("GAE (graph)", True), ("Decoder only (no message passing)", False)])
        models = {name: [fit(t, ug, sd, data, ex, pmu, psd) for sd in SEEDS] for name, ug in names}
        pool_ciks = ctx["pool"]["cik"].to_numpy()
        cand_nodes = torch.tensor([nid[c] for c in pool_ciks])
        with torch.no_grad():
            Z = {m: [mod.encode(X, S, E, ET, EC) for mod in ms] for m, ms in models.items()}
            for kind in ["gt3", "gt1"]:
                for q in queries(kind, t).itertuples():
                    hit = np.where(pool_ciks == q.cust)[0]
                    if len(hit) == 0 or q.cik not in nid:
                        continue
                    ti = int(hit[0])
                    F, ok, _ = pair_features(ctx, q.cik, q.est_cust_rev)
                    Fp = np.c_[F, text_pair_features(ctx, text_context(t), q.cik, pool_ciks)] if TEXT else F
                    pf = torch.tensor((Fp - pmu) / psd)
                    s_idx = torch.full((len(pool_ciks),), nid[q.cik])
                    track = "gt3" if kind == "gt3" else ("gt1a" if q.direction == "revealed_later" else "gt1b")
                    for m, ms in models.items():
                        sc = np.mean([mod.score(z, s_idx, cand_nodes, pf).numpy() for mod, z in zip(ms, Z[m])], 0)
                        s = np.where(ok, sc, -np.inf)
                        rank = 1 + int((s > s[ti]).sum()) if np.isfinite(s[ti]) else int(ok.sum()) + 1
                        rows.append({"year": t, "track": track, "model": m, "cik": q.cik, "cust": q.cust, "rank": rank,
                                     "pool": int(ok.sum()), "history": bool(F[ti, 2]), "true_ok": bool(ok[ti])})
        pd.DataFrame(rows).to_parquet(RES / f"ranks_gae{'_text' if TEXT else ''}_{'_'.join(map(str, YEARS))}.parquet")
        print("gae", t, "train queries", len(ex), flush=True)


if __name__ == "__main__":
    main()
