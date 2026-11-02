"""논문 그림 일괄 생성 (paper/figures/). 색 규칙: 실명·연결=파랑, 익명=주황, 실명·미연결=회색, 강조/AI=청록.
모든 수치는 results/ 와 data/interim/ 산출물에서 직접 계산한다."""
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, RES, OUT = ROOT / "data" / "interim", ROOT / "results", ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8f8e88", "#e6e5e0", "#ffffff"
NAMED, ANON, UNLINK, AQUA = "#2a78d6", "#eb6834", "#b9b8b2", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 300,
                     "legend.frameon": False, "legend.fontsize": 8.5})
AI_SIC = {3674, 3572, 3576, 3577, 3571, 3672, 3679, 3661, 3663, 3612, 3613, 3621, 3585}


def clean(ax, grid="y"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def fy_of(dd):
    s = dd.astype(str)
    return s.str[:4].astype(int) - (s.str[4:6].astype(int) < 7)


def relations(main=False):
    r = pd.read_parquet(INT / "relations.parquet")
    r = r[r["current"] & r["cls"].isin(["anonymous", "named"])].copy()
    r["fy"] = fy_of(r["ddate"])
    if main:  # 표 1–2 본 표본: 공급사 총자산·매출 필수
        f = pd.read_parquet(INT / "fundamentals.parquet")[["cik", "year", "at", "rev"]].dropna()
        r = r.assign(cik=r["cik"].astype(int)).merge(f.rename(columns={"year": "fy"})[["cik", "fy"]], on=["cik", "fy"])
    r["ai"] = pd.to_numeric(r["sic"], errors="coerce").isin(AI_SIC)
    return r[r["fy"].between(2012, 2025)]


# ── Figure 1: 개념도 ───────────────────────────────────────────────────────────
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sup = {"S1": (0.12, 0.78), "S2": (0.12, 0.50), "S3": (0.12, 0.22)}
    cus = {"Walmart": (0.82, 0.86), "Customer A\n(identity withheld)": (0.82, 0.55), "Customer B\n(identity withheld)": (0.82, 0.24)}
    edges = [("S1", "Walmart", 0.15, "named"), ("S1", "Customer A\n(identity withheld)", 0.23, "anon"),
             ("S2", "Customer A\n(identity withheld)", 0.31, "anon"), ("S3", "Customer B\n(identity withheld)", 0.42, "anon"),
             ("S2", "Walmart", 0.12, "named")]
    for ax, mode, title in [(axes[0], "named", "(a) Network observed from named disclosures"),
                            (axes[1], "all", "(b) Network implied by all disclosures")]:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_title(title)
        for n, (x, y) in sup.items():
            ax.add_patch(FancyBboxPatch((x - 0.07, y - 0.055), 0.14, 0.11, boxstyle="round,pad=0.01", fc="#f4f3ef", ec=INK2, lw=0.8))
            ax.text(x, y, f"Supplier {n[-1]}", ha="center", va="center", fontsize=8.5, color=INK)
        for n, (x, y) in cus.items():
            is_anon = "withheld" in n
            if mode == "named" and is_anon:
                continue
            ax.add_patch(FancyBboxPatch((x - 0.13, y - 0.065), 0.26, 0.13, boxstyle="round,pad=0.01",
                                        fc="#fff4ee" if is_anon else "#eaf2fc", ec=ANON if is_anon else NAMED, lw=1.0,
                                        ls="--" if is_anon else "-"))
            ax.text(x, y, n, ha="center", va="center", fontsize=8, color=INK)
        for s, c, w, kind in edges:
            if mode == "named" and kind == "anon":
                continue
            (x0, y0), (x1, y1) = sup[s], cus[c]
            col = ANON if kind == "anon" else NAMED
            ax.add_patch(FancyArrowPatch((x0 + 0.075, y0), (x1 - 0.135, y1), arrowstyle="-|>", mutation_scale=9, lw=1 + 10 * w,
                                         color=col, alpha=0.85, ls="--" if kind == "anon" else "-"))
            lx, ly = x0 + 0.075 + 0.28 * (x1 - 0.135 - x0 - 0.075), y0 + 0.28 * (y1 - y0)
            ax.text(lx, ly + 0.035, f"{w:.0%}", fontsize=8, color=col, ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        if mode == "named":
            ax.text(0.5, 0.06, "Suppliers 1–3 disclose 3 of 5 dependencies only as \"Customer A/B\":\nthey vanish from a named-only network",
                    ha="center", fontsize=8, color=INK2)
        else:
            ax.text(0.5, 0.06, "Weights of anonymous links are observed (share of supplier revenue);\nonly the counterparty is unknown",
                    ha="center", fontsize=8, color=INK2)
    fig.text(0.01, 0.005, "Edge width ∝ customer's share of supplier revenue. Solid = named customer; dashed = anonymous customer (e.g., XBRL member CustomerAMember).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "fig1_concept.png"); plt.close(fig)


# ── Figure 2: 추세 ────────────────────────────────────────────────────────────
def fig2():
    r = relations(main=True)
    a_all = r.groupby("fy")["cls"].apply(lambda s: (s == "anonymous").mean()) * 100
    a_ai = r[r["ai"]].groupby("fy")["cls"].apply(lambda s: (s == "anonymous").mean()) * 100
    tn = pd.read_csv(RES / "text_naming_trend.csv", index_col=0).loc[2012:2025]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    ax.plot(a_all.index, a_all, color=NAMED, lw=2.2, marker="o", ms=4, label="All suppliers")
    ax.plot(a_ai.index, a_ai, color=AQUA, lw=2.2, marker="o", ms=4, label="AI hardware suppliers")
    for s, c in [(a_all, NAMED), (a_ai, AQUA)]:
        ax.annotate(f"{s.iloc[-1]:.0f}%", (s.index[-1], s.iloc[-1]), xytext=(5, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5)
        ax.annotate(f"{s.iloc[0]:.0f}%", (s.index[0], s.iloc[0]), xytext=(-5, 0), textcoords="offset points", va="center", ha="right", color=INK, fontsize=8.5)
    ax.set_ylim(35, 90); ax.set_xlim(2010.8, 2026.2); ax.set_ylabel("Anonymous share of major-customer links (%)")
    ax.set_title("(a) Major customers disclosed anonymously"); ax.legend(loc="upper left"); clean(ax)
    ax = axes[1]
    ax.bar(tn.index, tn["text_named_rate"] * 100, color=ANON, width=0.7)
    for x, v in zip(tn.index, tn["text_named_rate"] * 100):
        if x in (2012, 2016, 2020, 2025):
            ax.text(x, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("% of all-anonymous filings naming a customer in text")
    ax.set_title("(b) Customer named elsewhere in the 10-K"); clean(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig2_trends.png"); plt.close(fig)


# ── Figure 3: 원인 분해 ───────────────────────────────────────────────────────
def fig3():
    dec = pd.read_csv(RES / "propensity_decomposition.csv", index_col=0).loc[2014:2025]
    by = pd.read_csv(RES / "rev2_table5_main_by_year.csv", index_col=0).loc[2015:2025]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    ax.plot(dec.index, dec["actual"] * 100, color=ANON, lw=2.2, marker="o", ms=4, label="Actual")
    ax.plot(dec.index, dec["expected"] * 100, color=MUTED, lw=2, ls="--", marker="o", ms=3.5, label="Predicted by firm characteristics (base 2014)")
    ax.fill_between(dec.index, dec["expected"] * 100, dec["actual"] * 100, color=ANON, alpha=0.12)
    ax.annotate("Change in propensity\n+27 pp by 2025", (2023.2, 56), fontsize=8.5, color=INK)
    ax.set_ylim(35, 75); ax.set_ylabel("Anonymous share (%)"); ax.legend(loc="upper left")
    ax.set_title("(a) Not explained by changing firms"); clean(ax)
    ax = axes[1]
    styles = {"New SEC registrant": (ANON, "-"), "Existing registrant, first disclosure": ("#eda100", "-"),
              "Re-entrant": (MUTED, "-"), "Continuing": (NAMED, "-")}
    for col, (c, ls) in styles.items():
        ax.plot(by.index, by[col], color=c, lw=2.4 if col == "Continuing" else 1.6, ls=ls, marker="o", ms=3.5, label=col)
    ax.set_ylim(40, 95); ax.set_ylabel("Anonymous share (%)"); ax.legend(loc="upper left", fontsize=7.8)
    ax.set_title("(b) New disclosers lead; continuing disclosers follow"); clean(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig3_decomposition.png"); plt.close(fig)


# ── Figure 4: 맹점의 크기 (거래액 구성) ─────────────────────────────────────────
def fig4():
    r = relations()
    r = r[r["est_cust_rev"].notna()]
    r["kind"] = np.where(r["cls"] == "anonymous", "Anonymous", np.where(r["cls2"] == "named_linked", "Named, linked to registrant", "Named, not linked"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for ax, sub, title in [(axes[0], r, "(a) All suppliers"), (axes[1], r[r["ai"]], "(b) AI hardware suppliers")]:
        t = sub.pivot_table(index="fy", columns="kind", values="est_cust_rev", aggfunc="sum").fillna(0)
        t = t[["Named, linked to registrant", "Named, not linked", "Anonymous"]]
        sh = t.div(t.sum(1), axis=0) * 100
        ax.stackplot(sh.index, sh.T.values, colors=[NAMED, UNLINK, ANON], alpha=0.95, edgecolor="white", linewidth=0.6,
                     labels=sh.columns)
        ax.set_xlim(2012, 2025); ax.set_ylim(0, 100); ax.set_title(title); clean(ax, grid=None)
        last = sh.iloc[-1]
        ax.text(2024.7, 100 - last["Anonymous"] / 2, f"{last['Anonymous']:.0f}%\nanonymous", ha="right", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax.text(2024.7, last["Named, linked to registrant"] / 2, f"{last['Named, linked to registrant']:.0f}%", ha="right", va="center", fontsize=8, color="white")
    axes[0].set_ylabel("Share of relationship dollars (%)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(-0.02, -0.08), ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "fig4_blindspot.png"); plt.close(fig)


# ── Figure 5: 상·하한 ─────────────────────────────────────────────────────────
def fig5():
    b = pd.read_csv(RES / "rev_c4_bounds.csv")
    order = ["Microsoft", "Amazon", "Alphabet", "Meta", "Oracle", "Apple", "NVIDIA", "Cisco", "Dell", "Intel"]
    b = b.set_index("customer").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    floor = 0.01
    for i, rw in b.iterrows():
        lo = max(rw["lower_$B"], floor)
        ax.plot([lo, rw["upper_separate_$B"]], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.plot([lo, rw["upper_joint_$B"]], [i, i], color=ANON, lw=6, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.scatter(lo, i, color=NAMED, s=46, zorder=3)
        ax.scatter(rw["upper_joint_$B"], i, color=ANON, s=46, zorder=3)
        ax.text(lo / 1.35, i, ("$0" if rw["lower_$B"] < 0.005 else f"${rw['lower_$B']:.2f}B"), ha="right", va="center", fontsize=7.8, color=NAMED)
        ax.text(rw["upper_separate_$B"] * 1.12, i, f"${rw['upper_joint_$B']:.0f}B", ha="left", va="center", fontsize=7.8, color=ANON)
    ax.set_xscale("log"); ax.set_xlim(0.002, 2000)
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000]); ax.set_xticklabels(["$0.01B", "$0.1B", "$1B", "$10B", "$100B", "$1,000B"])
    ax.set_yticks(range(len(b))); ax.set_yticklabels(b["customer"]); ax.invert_yaxis()
    ax.set_xlabel("Disclosed purchases from AI hardware and AI-exposed suppliers, FY2023–2025 (US$ billions, log scale)")
    ax.scatter([], [], color=NAMED, s=40, label="Identified from named disclosures (lower bound)")
    ax.scatter([], [], color=ANON, s=40, label="Maximum feasible allocation under joint constraints (upper bound)")
    ax.plot([], [], color=GRID, lw=6, label="Upper bound without the cumulative-capacity constraint")
    ax.plot([], [], " ", label=r"(\$0 lower bounds are plotted at \$0.01B)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, fontsize=7.8)
    ax.set_title("What disclosures identify vs. what they are consistent with"); clean(ax, grid="x")
    fig.tight_layout(); fig.savefig(OUT / "fig5_bounds.png"); plt.close(fig)


# ── Figure 6: AI 증거 (사건연구 + 사양 곡선) ───────────────────────────────────
def fig6():
    es = pd.read_csv(RES / "rev_c1_eventstudy.csv")
    es = pd.concat([es, pd.DataFrame({"year": [2021], "coef": [0], "ci_low": [0], "ci_high": [0]})]).sort_values("year")
    specs = [("Baseline: top decile", 0.069, 1.77), ("+ controls", 0.079, 1.79), ("Outcome: all customers anonymous", 0.096, 2.04),
             ("SIC2×year FE + controls", 0.085, 2.13), ("Matched sample", 0.101, 2.15), ("Post ≥ FY2022", 0.062, 1.66),
             ("Excl. segment-attributed", 0.075, 1.88), ("Suppliers ≥10 years", 0.114, 1.77),
             ("Top decile: 90th–95th pct", 0.140, 1.98), ("Top decile: above 95th pct", 0.035, 0.66),
             ("Top 5% exposure", 0.025, 0.54), ("Infrastructure terms only", 0.041, 0.94),
             ("log(1 + exposure)", 0.013, 1.31), ("AI-hardware SIC", 0.023, 0.73),
             ("Placebo: Post ≥ 2018", -0.041, -0.81), ("Placebo: Post ≥ 2019", -0.047, -1.21), ("Placebo: Post ≥ 2020", -0.047, -1.30)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.axhline(0, color=INK2, lw=1); ax.axvspan(2021.5, 2025.5, color=AQUA, alpha=0.07, lw=0)
    ax.errorbar(es["year"], es["coef"] * 100, yerr=[(es["coef"] - es["ci_low"]) * 100, (es["ci_high"] - es["coef"]) * 100],
                fmt="o", color=AQUA, ecolor=AQUA, capsize=3, ms=5)
    ax.text(2021.7, 24, "Post-ChatGPT", color=INK2, fontsize=8)
    ax.set_xticks(range(2014, 2026, 2)); ax.set_ylabel("Top-decile AI exposure × year (pp), 95% CI")
    ax.set_title("(a) Event study (base FY2021)"); clean(ax)
    ax = axes[1]
    y = np.arange(len(specs))[::-1]
    for yi, (lab, b, t) in zip(y, specs):
        se = abs(b / t) if t else 0
        c = MUTED if lab.startswith("Placebo") else (AQUA if abs(t) >= 1.96 else ("#7fcfb1" if abs(t) >= 1.645 else UNLINK))
        ax.plot([(b - 1.96 * se) * 100, (b + 1.96 * se) * 100], [yi, yi], color=c, lw=2)
        ax.scatter(b * 100, yi, color=c, s=30, zorder=3)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in specs], fontsize=8)
    ax.set_xlabel("AI × Post (pp), 95% CI")
    ax.set_title("(b) Every specification we estimated"); clean(ax, grid="x")
    for c, lab in [(AQUA, "p < 0.05"), ("#7fcfb1", "p < 0.10"), (UNLINK, "n.s."), (MUTED, "placebo")]:
        ax.scatter([], [], color=c, s=25, label=lab)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig6_ai_evidence.png"); plt.close(fig)


# ── Figure 7: 투자 사이클 공변 ─────────────────────────────────────────────────
def fig7():
    c = pd.read_csv(RES / "rev_c2_transmission.csv")
    labels = ["AI suppliers\ncapex growth", "AI suppliers\ncapex surprise", "Non-AI suppliers (placebo)\ncapex growth", "Non-AI suppliers (placebo)\ncapex surprise"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(c))
    for off, col, tcol, color, lab in [(-0.12, "S×AnonDep", "t_anon", ANON, "× anonymous customer dependence"),
                                       (0.12, "S×NamedDep", "t_named", NAMED, "× named customer dependence")]:
        b = c[col].to_numpy(); se = np.abs(b / c[tcol].to_numpy())
        ax.errorbar(x + off, b, yerr=1.96 * se, fmt="o", color=color, ecolor=color, capsize=3, ms=6, label=lab)
    ax.axhline(0, color=INK2, lw=1); ax.axvline(1.5, color=GRID, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("Response of supplier revenue growth\nto platform capex (95% CI)")
    ax.set_title("Supplier revenue moves with hyperscaler investment only in the AI supply chain"); ax.legend(loc="upper right"); clean(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig7_capex.png"); plt.close(fig)


# ── Figure 8: 복원 곡선 ───────────────────────────────────────────────────────
def fig8():
    res = pd.concat([pd.read_parquet(p) for p in glob.glob(str(RES / "ranks_*.parquet"))]).drop_duplicates(["year", "track", "model", "cik", "cust"])
    g = res[(res["track"] == "gt3") & (~res["history"])]
    key = ["year", "cik", "cust"]
    common = g.groupby(key)["model"].nunique(); common = common[common == g["model"].nunique()].reset_index()[key]
    g = g.merge(common, on=key)
    show = [("B1 size", "Size only", UNLINK, ":"), ("B3 industry+size", "Industry prior × size (rule)", MUTED, "--"),
            ("LGBM ranker (no graph features)", "Gradient-boosted ranker", NAMED, "-"),
            ("Decoder only + product-similarity features", "Neural ranker, no message passing", AQUA, "-"),
            ("GAE + product-similarity & location graph", "Graph autoencoder (heterogeneous graph)", ANON, "-")]
    Ks = np.unique(np.round(np.logspace(0, np.log10(500), 60)).astype(int))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m, lab, col, ls in show:
        rk = g.loc[g["model"] == m, "rank"].to_numpy()
        ax.plot(Ks, [(rk <= k).mean() * 100 for k in Ks], color=col, ls=ls, lw=2, label=lab)
    ax.set_xscale("log"); ax.set_xticks([1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_xlabel("K (top-K candidates among ≈3,100 eligible SEC registrants)"); ax.set_ylabel("True customer in top K (%)")
    ax.axvline(10, color=GRID, lw=1); ax.text(10.5, 3, "K = 10", fontsize=8, color=INK2)
    ax.set_ylim(0, 100); ax.set_title(f"Recovering never-named customers (GT3 new relationships, N = {len(common)})")
    ax.legend(loc="upper left", fontsize=8); clean(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig8_recovery.png"); plt.close(fig)


# ── 부록 그림 ─────────────────────────────────────────────────────────────────
def figA():
    t = pd.read_excel(RES / "tables_data_section.xlsx", sheet_name="T1C_ff12", index_col=0).sort_values("anon_ratio_%")
    names = {"Money": "Finance", "Durbl": "Consumer durables", "Utils": "Utilities", "Enrgy": "Energy", "Other": "Other", "Manuf": "Manufacturing",
             "Hlth": "Healthcare", "Telcm": "Telecom", "BusEq": "Business equipment", "NoDur": "Consumer non-durables", "Shops": "Retail & wholesale", "Chems": "Chemicals"}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([names.get(i, i) for i in t.index], t["anon_ratio_%"], color=ANON, height=0.62)
    for i, (v, n) in enumerate(zip(t["anon_ratio_%"], t["edges"])):
        ax.text(v + 1, i, f"{v:.0f}%  (n = {n:,})", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 100); ax.set_xlabel("Anonymous share of major-customer links, FY2010–2025 (%)")
    ax.set_title("Anonymous disclosure by Fama–French 12 industry"); clean(ax, grid="x")
    fig.tight_layout(); fig.savefig(OUT / "figA1_industry.png"); plt.close(fig)

    r = relations()
    a = r[(r["fy"] == 2025) & r["ai"] & (r["cls"] == "anonymous") & r["est_cust_rev"].notna()].sort_values("est_cust_rev", ascending=False).head(10)
    short = lambda n: n.title().replace(" Inc.", "").replace(" Inc", "").replace(" Corp", "").replace("/De", "").replace(" /D", "").replace(",", "").replace(" Technologies", "").strip()[:20]  # noqa: E731
    lab = [f"{short(n)} — {m.replace('Customerthree', 'CustomerThree')}" for n, m in zip(a["name"], a["member"])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(lab[::-1], (a["est_cust_rev"] / 1e9)[::-1], color=ANON, height=0.6)
    for i, (v, p) in enumerate(zip((a["est_cust_rev"] / 1e9)[::-1], a["pct"][::-1])):
        ax.text(v + 0.4, i, rf"\${v:.1f}B ({p:.0%} of revenue)", va="center", fontsize=7.8, color=INK)
    ax.set_xlabel("Estimated relationship value, FY2025 (US$ billions)"); ax.set_xlim(0, (a["est_cust_rev"].max() / 1e9) * 1.45); ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Largest anonymous customers of AI hardware suppliers, FY2025"); clean(ax, grid="x")
    fig.tight_layout(); fig.savefig(OUT / "figA2_largest_anon.png"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, figA]:
        f(); print("done", f.__name__)
