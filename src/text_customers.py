"""10-K 본문에서 '실명 고객 + 매출비중' 문장 추출.

잡는 구문 (대소문자 무시, 문장 단위):
  A) <이름> (accounted for|represented|comprised|made up) [approximately|about|over|more than] N% of [our] [total|net|consolidated] (revenue|sales|net sales)
  B) (sales|revenue[s]) (to|from) <이름> (accounted for|represented|were|was|comprised) [approximately] N% of ...
이름 후보는 대문자로 시작하는 1~6단어 구. 일반어(고객·정부·지역·제품·부문 등)는 제외.
반환: 문장, 이름 문자열, 비중. EDGAR 이름 마스터와의 연결은 build_gt2 에서.
"""
import re

NUM = r"(?:approximately\s+|about\s+|over\s+|more than\s+|nearly\s+|almost\s+)?(\d{1,2}(?:\.\d{1,2})?)\s?%"
TAIL = r"\s+of\s+(?:the\s+)?(?:our\s+|its\s+|the\s+company'?s\s+)?(?:total\s+|net\s+|consolidated\s+|gross\s+)*(?:net\s+)?(?:revenues?|sales|net\s+sales)"
NAME = r"((?:[A-Z][A-Za-z0-9&'\.\-]*(?:\s+|,\s+)){0,5}[A-Z][A-Za-z0-9&'\.\-]*)"
VERB = r"(?:accounted\s+for|represented|comprised|made\s+up|constituted|totaled)"
PAT_A = re.compile(NAME + r"(?:\s*\([^)]{0,60}\))?\s+" + VERB + r"\s+" + NUM + TAIL)
PAT_B = re.compile(r"(?:sales|revenues?|net\s+sales)\s+(?:to|from)\s+" + NAME + r"(?:\s*\([^)]{0,60}\))?\s*,?\s+(?:" + VERB[3:-1] + r"|were|was)\s+" + NUM + TAIL, re.I)
GENERIC = re.compile(
    r"^(?:the|our|one|two|three|four|five|a|an|each|no|none|such|this|these|those|its|their|other|another|largest|"
    r"single|top|ten|customer|customers|client|clients|distributor|distributors|government|federal|u\.?s\.?|united|"
    r"international|foreign|domestic|segment|product|products|service|services|sales|revenue|revenues|net|total|"
    r"fiscal|during|in|for|as|approximately|company|we|it|which|that|and|or|of|north|south|europe|asia|china|japan|"
    r"americas|emea|apac|canada|mexico|germany|india|korea|taiwan|year|years|quarter|december|june|september|march|"
    r"customer [a-z]|customer \d|item|note|table|see|accordingly|however|additionally|together|collectively|"
    r"medicare|medicaid|department|agency|dod|navy|army|air force|retail|wholesale|oem|oems|channel|direct|indirect)\b",
    re.I)


def clean_html(html: str) -> str:
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&#\d+;|&[a-zA-Z]+;", " ", t)
    return re.sub(r"\s+", " ", t)


def extract(text: str):
    out = []
    for pat, kind in [(PAT_A, "A"), (PAT_B, "B")]:
        for m in pat.finditer(text):
            name = m.group(1).strip(" ,.")
            # 앞쪽 잡음 제거: 문장 시작 접속어·연도 등
            name = re.sub(r"^(?:In|During|For|Fiscal|Year|Ended|As|Sales to|Revenue from)\s+", "", name)
            name = re.sub(r"^(?:\d{4}\s*,?\s*)", "", name).strip(" ,.")
            if len(name) < 3 or GENERIC.search(name) or re.fullmatch(r"[A-Z]", name):
                continue
            if re.search(r"\b(?:Customer|Client)\s+[A-Z0-9]\b", name):
                continue
            s0 = max(0, m.start() - 150)
            out.append({"kind": kind, "name": name[-80:], "pct": float(m.group(2)) / 100,
                        "sentence": text[s0: m.end() + 30]})
    return out


# ── 고객 실명 판정 필터 (FY2021 100건 판정 표본으로 설계; 새 표본으로 재검증 필요)
SUFFIX = re.compile(r"\b(Inc|Incorporated|Corp|Corporation|Company|Co|LLC|L\.L\.C|Ltd|Limited|LP|L\.P|plc|PLC|AG|SA|S\.A|NV|N\.V|"
                    r"GmbH|Group|Holdings?|Technologies|Systems|Electronics|Communications|Health|University|Stores)\b\.?")
CUSTCTX = re.compile(r"\b(customer|customers|distributor|distributors|client|clients|reseller|retailer|charterer|tenant|purchaser|OEM)\b"
                     r"|sales\s+to\b|sales\s+via\b|largest", re.I)
NONCUST = re.compile(r"\b(products?|segment|division|business|brand|royalt|pool|channel|marketplace|operations|region|states|countries|"
                     r"customers$|consumers|users|providers|implants|sensors|antibod|silver|zinc|gold|copper|largest customer|"
                     r"network partner|statements|SPL|Puerto Rico|France|Latin America|Europe|each)\b", re.I)
ALLCAPS_PRODUCT = re.compile(r"^[A-Z]{5,}$")  # 대소문자 구분: GOCOVRI 같은 제품명 (IBM·CRB 같은 약칭은 통과)
ANON_TXT = re.compile(r"\b(?:significant|major|key|large|principal)?\s*customer\s+[A-Z0-9]\b|\bcustomer\s+(?:one|two|three)\b", re.I)


def is_customer(name: str, sentence: str, filer_name: str, linked: bool) -> bool:
    if NONCUST.search(name) or ANON_TXT.search(name) or ALLCAPS_PRODUCT.match(name):
        return False
    ftoks = {t for t in re.findall(r"[A-Za-z]{3,}", filer_name.lower())} - {"inc", "corp", "corporation", "company", "holdings", "group",
                                                                              "the", "international", "technologies", "industries"}
    ntoks = set(re.findall(r"[A-Za-z]{3,}", name.lower()))
    if ftoks & ntoks:  # 공시 회사 자신의 자회사·사업부 (예: Sypris Solutions → Sypris Technologies)
        return False
    pre = sentence[: max(1, sentence.find(name)) + len(name) + 40]
    return bool(linked or SUFFIX.search(name) or CUSTCTX.search(pre))
