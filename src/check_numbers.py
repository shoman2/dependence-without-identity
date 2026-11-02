"""원고 수치 ↔ 결과 파일 자동 대조. 각 항목: 결과 파일에서 다시 계산한 값을 원고 표기 형식으로 만들어 원고에 있는지 확인."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
_m = ROOT / "paper" / "manuscript.md"
if not _m.exists():
    raise SystemExit("paper/manuscript.md not found — the manuscript is not part of the public repository; skipping number check.")
M = _m.read_text()
checks = []


def chk(label, value, fmt):
    s = fmt.format(value)
    checks.append((label, s, s in M))


t1 = pd.read_excel(R / "tables_data_section.xlsx", sheet_name="T1B_year", index_col=0)
chk("총 링크", int(t1.loc["Total", "edges"]), "{:,}")
chk("총 공급사-연도", int(t1.loc["Total", "supplier_years"]), "{:,}")
chk("2012 익명 비중", t1.loc[2012, "anon_ratio_%"], "{:.1f}")
chk("2025 익명 비중", t1.loc[2025, "anon_ratio_%"], "{:.1f}")
tn = pd.read_csv(R / "text_naming_trend.csv", index_col=0)
chk("본문 실명 2012", 100 * tn.loc[2012, "text_named_rate"], "{:.1f}")
chk("본문 실명 2025", 100 * tn.loc[2025, "text_named_rate"], "{:.1f}")
dk = pd.read_csv(R / "dark_network.csv")
chk("AI HW 익명$ 2025", dk.query("group=='AI hardware' and fy==2025")["dark_$_share"].iloc[0], "{:.3f}")
chk("전체 익명$ 2025", dk.query("group=='All' and fy==2025")["dark_$_share"].iloc[0], "{:.3f}")
chk("AI HW 의존도 포착 2025", dk.query("group=='AI hardware' and fy==2025")["dependence_captured_%"].iloc[0], "{:.1f}")
b = pd.read_csv(R / "rev_c4_bounds.csv").set_index("customer")
chk("MSFT 공동 상한", b.loc["Microsoft", "upper_joint_$B"], "{:.1f}")
chk("Meta 공동 상한", b.loc["Meta", "upper_joint_$B"], "{:.1f}")
chk("Amazon 개별 상한", b.loc["Amazon", "upper_separate_$B"], "{:.1f}")
lp = pd.read_csv(R / "rev2_group_lp.csv")
chk("LP 최대 배정", lp["lp_max_assignable_$B"].iloc[0], "{:.1f}")
c1 = pd.read_csv(R / "rev_c1_did.csv").set_index("spec")
chk("SIC2×연도 + 통제 계수", c1.loc["Supplier + SIC2×year FE + controls", "coef"], "{:.3f}")
chk("매칭 계수", c1.loc["Matched sample (3 nearest controls, same SIC1)", "coef"], "{:.3f}")
chk("매칭 t", c1.loc["Matched sample (3 nearest controls, same SIC1)", "t"], "{:.2f}")
pt = pd.read_csv(R / "rev_c1_pretrend.csv")
chk("사전추세 χ²", pt["pretrend_chi2"].iloc[0], "{:.2f}")
mp = pd.read_csv(R / "rev3_matched_pretrend.csv")
chk("매칭 사전추세 χ²", mp["chi2"].iloc[0], "{:.2f}")
bd = pd.read_csv(R / "rev3_bins_diff.csv")
chk("구간 차이(통제) t", bd.query("controls")["t"].iloc[0], "{:.2f}")
c2 = pd.read_csv(R / "rev_c2_transmission.csv")
chk("AI 익명 계수", c2["S×AnonDep"].iloc[0], "{:.3f}")
chk("비AI 익명 계수", c2["S×AnonDep"].iloc[2], "{:.3f}")
chk("차이 CI 하한", c2["diff_ci_low"].iloc[0], "{:.2f}")
t5 = pd.read_csv(R / "rev2_table5_main.csv").set_index("type")
chk("표5 신규 등록 비중", t5.loc["New SEC registrant", "anon_share_%"], "{:.1f}")
chk("표5 계속 공급사-연도", int(t5.loc["Continuing", "supplier_years"]), "{:,}")
fl = pd.read_csv(R / "rev_c5_flow.csv")
chk("평가 흐름 신규", int(fl["n"].iloc[4]), "{:,}")
ci = pd.read_csv(R / "rev_c5_gt1a_ci.csv")
chk("GT1a 최고 MRR CI 상한", ci["ci_high"].max(), "{:.2f}")
ok = sum(c[2] for c in checks)
for label, s, found in checks:
    print(("✓" if found else "✗"), label, s)
print(f"\n{ok}/{len(checks)} 일치")
