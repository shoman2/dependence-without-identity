"""연결 정밀도 감사 결과 반영 민감도: 'Company/Carrier/Account…+ID' 익명 재분류, 접두일치 연결 해제 → 핵심 수치 재계산."""
import re
from pathlib import Path

import pandas as pd

INT = Path(__file__).resolve().parents[1] / "data" / "interim"
AI = {3674, 3572, 3576, 3577, 3571, 3672, 3679, 3661, 3663, 3612, 3613, 3621, 3585}
NEWANON = re.compile(r"^(Company|Carrier|Account|Entity|Vendor|Buyer|Licensee|Merchant)s?(No|Number)?([A-Z]{1,2}|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)$", re.I)
r = pd.read_parquet(INT / "relations.parquet"); r = r[r["current"]].copy()
r["fy"] = r["ddate"].astype(str).str[:4].astype(int) - (r["ddate"].astype(str).str[4:6].astype(int) < 7)
alt = r.copy()
m_anon = alt["cls"].eq("named") & alt["member"].fillna("").str.replace("Member$", "", regex=True).str.match(NEWANON)
m_pref = alt["cls2"].eq("named_linked") & alt["link_method"].eq("prefix")
alt.loc[m_anon, ["cls", "cls2"]] = ["anonymous", "anonymous"]
alt.loc[m_pref & ~m_anon, "cls2"] = "named_unlinked"
for lab, d in [("baseline", r), ("audit-corrected", alt)]:
    x = d[d["cls"].isin(["anonymous", "named"]) & d["fy"].between(2012, 2025)]
    a = x.groupby("fy")["cls"].apply(lambda s: (s == "anonymous").mean())
    y = x[x["est_cust_rev"].notna() & (x["fy"] == 2025)]
    ai = y[pd.to_numeric(y["sic"], errors="coerce").isin(AI)]
    dk = lambda z: z.loc[z["cls"] == "anonymous", "est_cust_rev"].sum() / z["est_cust_rev"].sum()  # noqa: E731
    print(lab, {"anon_2012": round(100 * a[2012], 1), "anon_2025": round(100 * a[2025], 1), "dark_all_2025": round(dk(y), 3),
                "dark_AI_2025": round(dk(ai), 3), "GT3": d[d["cls2"] == "named_linked"].drop_duplicates(["cik", "ddate", "cust_cik"]).shape[0]})
