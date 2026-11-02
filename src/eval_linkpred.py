"""링크 예측 결과 종합: 전체 / 신규 관계(주 결과) / 지속 관계, 공급사 규모 3분위, GT1a·GT1b(+IPW), 쌍대 검정."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
RES, INT = ROOT / "results", ROOT / "data" / "interim"
ORDER = ["B1 size", "B3 industry+size", "B4 +history", "LGBM ranker (no graph features)", "LGBM ranker",
         "LGBM ranker + product similarity", "Decoder only + product-similarity features", "GAE + product-similarity & location graph", "Decoder only (no message passing)", "GAE (graph)"]


def metrics(d):
    r = d["rank"]
    return pd.Series({"n": len(r), "Hit@1": (r <= 1).mean(), "Hit@10": (r <= 10).mean(), "Hit@50": (r <= 50).mean(),
                      "MRR": (1 / r).mean(), "median rank": r.median()})


def main():
    res = pd.concat([pd.read_parquet(p) for p in sorted(RES.glob("ranks_lgbm*.parquet"))] +
                    [pd.read_parquet(p) for p in sorted(RES.glob("ranks_gae_*.parquet"))], ignore_index=True)
    res = res.drop_duplicates(["year", "track", "model", "cik", "cust"])
    models = [m for m in ORDER if m in set(res["model"])]
    # 모든 모형에 공통으로 있는 질의만 (공정 비교)
    key = ["year", "track", "cik", "cust"]
    common = res.groupby(key)["model"].nunique()
    common = common[common == len(models)].reset_index()[key]
    res = res.merge(common, on=key)
    fund = pd.read_parquet(INT / "fundamentals.parquet")[["cik", "year", "at"]]
    fund["year"] += 1  # 예측 시점(t-1) 규모
    res = res.merge(fund, on=["cik", "year"], how="left")
    g3 = res[res["track"] == "gt3"].copy()
    g3["size_terc"] = pd.qcut(np.log(g3["at"].where(g3["at"] > 0)).rank(method="first"), 3, labels=["Small", "Mid", "Large"])

    out = {}
    for name, d in [("GT3 all", g3), ("GT3 NEW relationships (main)", g3[~g3["history"]]),
                    ("GT3 persistent relationships", g3[g3["history"]])]:
        out[name] = d.groupby("model").apply(metrics).loc[models]
    for t in ["Small", "Mid", "Large"]:
        out[f"GT3 new, supplier size {t}"] = g3[(~g3["history"]) & (g3["size_terc"] == t)].groupby("model").apply(metrics).loc[models]
    for tr in ["gt1a", "gt1b"]:
        d = res[res["track"] == tr]
        if len(d):
            out[f"{tr.upper()}"] = d.groupby("model").apply(metrics).reindex(models)

    # IPW: 익명 관계 전체 중 GT1 에 포함될 확률 (규모·ROA·R&D) → GT1 가중 Hit@10
    rel = pd.read_parquet(INT / "relations.parquet")
    an = rel[(rel["cls"] == "anonymous") & rel["current"]].copy()
    an["year"] = an["ddate"].astype(str).str[:4].astype(int) - (an["ddate"].astype(str).str[4:6].astype(int) < 7)
    an["cik"] = an["cik"].astype(int)
    gt1 = pd.read_parquet(INT / "gt1_comparative.parquet")
    gk = set(zip(gt1["adsh"], gt1["member"]))
    an["in_gt1"] = [(a, m) in gk for a, m in zip(an["adsh"], an["member"])]
    f = pd.read_parquet(INT / "fundamentals.parquet")
    f["ln_at"] = np.log(f["at"].where(f["at"] > 0)); f["roa"] = (f["ni"] / f["at"]).clip(-2, 2)
    f["rnd"] = (f["rnd"].fillna(0) / f["rev"]).where(f["rev"] > 0).clip(0, 5)
    an = an.merge(f[["cik", "year", "ln_at", "roa", "rnd"]], on=["cik", "year"], how="left").dropna(subset=["ln_at", "roa", "rnd"])
    lg = smf.logit("in_gt1 ~ ln_at + roa + rnd", an.assign(in_gt1=an["in_gt1"].astype(int))).fit(disp=0)
    d1 = res[res["track"].isin(["gt1a", "gt1b"])].merge(f[["cik", "year", "ln_at", "roa", "rnd"]].assign(year=lambda x: x["year"]),
                                                        on=["cik", "year"], how="left").dropna(subset=["ln_at", "roa", "rnd"])
    p = lg.predict(d1).clip(lower=np.quantile(lg.predict(an), 0.05))
    d1["w"] = 1 / p
    ipw = d1.groupby("model").apply(lambda d: pd.Series({"n": len(d), "Hit@10 raw": (d["rank"] <= 10).mean(),
                                                         "Hit@10 IPW": np.average(d["rank"] <= 10, weights=d["w"]),
                                                         "MRR IPW": np.average(1 / d["rank"], weights=d["w"])})).reindex(models)
    out["GT1 (a+b) IPW-weighted"] = ipw

    # 쌍대 검정 (신규 관계): 역순위 차이, 공급사 군집
    nw = g3[~g3["history"]].pivot_table(index=key, columns="model", values="rank").reset_index()
    tests = []
    for a, b in [("GAE (graph)", "Decoder only (no message passing)"), ("GAE (graph)", "LGBM ranker"),
                 ("GAE (graph)", "B3 industry+size"), ("LGBM ranker", "B3 industry+size"),
                 ("LGBM ranker", "LGBM ranker (no graph features)"), ("Decoder only (no message passing)", "LGBM ranker (no graph features)"),
                 ("LGBM ranker + product similarity", "LGBM ranker"),
                 ("GAE + product-similarity & location graph", "Decoder only + product-similarity features"),
                 ("GAE + product-similarity & location graph", "LGBM ranker + product similarity"),
                 ("GAE + product-similarity & location graph", "GAE (graph)")]:
        d = nw.assign(diff=1 / nw[a] - 1 / nw[b]).dropna(subset=["diff"])
        fit = smf.ols("diff ~ 1", d).fit(cov_type="cluster", cov_kwds={"groups": d["cik"]})
        tests.append({"A": a, "B": b, "mean RR(A)-RR(B)": fit.params.iloc[0], "t (supplier-clustered)": fit.tvalues.iloc[0], "n": len(d)})
    tests = pd.DataFrame(tests)

    with pd.ExcelWriter(RES / "linkpred_results.xlsx") as xw:
        for k, v in out.items():
            v.round(4).to_excel(xw, sheet_name=k[:31])
        tests.round(4).to_excel(xw, sheet_name="paired tests", index=False)
    pd.set_option("display.width", 200)
    for k, v in out.items():
        print(f"\n== {k} ==\n{v.round(3).to_string()}")
    print("\n== 쌍대 검정 (GT3 신규 관계, 역순위) ==\n", tests.round(4).to_string(index=False))
    print("\nIPW 로짓:", lg.params.round(3).to_dict())


if __name__ == "__main__":
    main()
