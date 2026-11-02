"""주요고객 멤버 분류: 익명 / 상장사 실명(EDGAR 매칭) / 기타 실명 / 범주형."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sec_client import get_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROLE = r"(Customer|Client|Tenant|Distributor|Payor|Payer|Purchaser|Lessee|Partner|Wholesaler|Licensee|Operator|Counterparty|Collaborator|Reseller|Shipper)"
_ID = r"([A-Z]{1,2}|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|First|Second|Third|Fourth|Fifth)"
ANON = re.compile(
    rf"^(Major|Significant|Key|Principal|Large|Largest)?{ROLE}s?(No|Number)?{_ID}(ConcentrationRisk|Concentration)?$"
    rf"|^(Largest|Top|Single|One|Major|Significant|Key|Principal|First|Second|Third|Another|Additional)(Major)?{ROLE}\d*$"
    rf"|^(Second|Third|Fourth|Fifth)Largest{ROLE}$", re.I)
GENERIC = re.compile(r"Other|Government|Federal|Commercial|Retail|Wholesale|Medicare|Medicaid|Nonrelated|RelatedPart|Domestic|Foreign|"
                     r"International|Affiliate|^Direct|Indirect|All|Remaining|Total|Individual|Consumer|Segment|Industrial|Residential|"
                     r"Utilit|Military|StateAndLocal|Institutional|Customers?$|Distributors$|Revenue|Program|Contract|Product|Service|"
                     r"Agenc|Department|Ministry|Payors$|Patients|Insurance|Hospital|Pharmac|Lessees$|Tenants$|Operators$|Partners$|Members$|"
                     r"Group$|Largest|^Top|^(Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)[A-Z]|Merchants$|Clients$|^Us(Army|Navy|AirForce)|^Doe$|^Dod$", re.I)
SUFFIX = r"(Inc|Incorporated|Corp|Corporation|Company|Co|LLC|Ltd|Limited|Plc|LP|Holdings?|Group|NationalAssociation|NA|SA|AG|NV|The)\b"


def norm(s: str) -> str:
    s = re.sub(r"Member$", "", s)
    s = re.sub(SUFFIX, "", re.sub(r"([a-z])([A-Z])", r"\1 \2", s))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def listed_names():
    tick = get_json("https://www.sec.gov/files/company_tickers.json", ROOT / "data" / "raw" / "company_tickers.json")
    out = {}
    for v in tick.values():
        n = norm(re.sub(r"[^A-Za-z0-9 ]", " ", v["title"]).title().replace(" ", ""))
        if len(n) >= 4:
            out.setdefault(n, (int(v["cik_str"]), v["title"]))
    return out


def classify(members: pd.Series) -> pd.DataFrame:
    names = listed_names()
    rows = []
    for m in members:
        base = re.sub(r"Member$", "", m) if isinstance(m, str) else ""
        if not base:
            rows.append(("none", None, None))
        elif ANON.search(base):
            rows.append(("anonymous", None, None))
        elif norm(base) in names:
            rows.append(("named_listed", *names[norm(base)]))
        elif GENERIC.search(base):
            rows.append(("generic", None, None))
        else:
            rows.append(("named_other", None, None))
    return pd.DataFrame(rows, columns=["cls", "cust_cik", "cust_name"], index=members.index)
