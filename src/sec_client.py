"""SEC EDGAR 공개 API 클라이언트 — fair access 정책 준수(연락처 UA, 초당 요청 제한, 디스크 캐시)."""
import json
import os
import threading
import time
from pathlib import Path

import requests

def _user_agent() -> str:
    """SEC fair-access 정책: 연락처가 들어간 User-Agent 필수. 환경변수 SEC_USER_AGENT 또는 저장소 루트의
    (git 미추적) .sec_user_agent 파일에서 읽는다. 예: "Your Name your.email@example.com"."""
    ua = os.environ.get("SEC_USER_AGENT")
    p = Path(__file__).resolve().parents[1] / ".sec_user_agent"
    if not ua and p.exists():
        ua = p.read_text().strip()
    if not ua:
        raise RuntimeError("Set SEC_USER_AGENT (e.g. 'Your Name you@example.com') or create .sec_user_agent — required by SEC.")
    return ua


UA = _user_agent()
MIN_INTERVAL = 0.15  # SEC 한도는 10 req/s, 여유 있게 ~6.6 req/s
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw"

_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
_last = 0.0
_lock = threading.Lock()  # 스레드가 여러 개여도 전체 요청 간격을 MIN_INTERVAL 이상으로 유지


def get(url: str, cache_path: Path | None = None, retries: int = 4) -> bytes | None:
    """GET with disk cache. 404 → None (캐시에 빈 마커 저장)."""
    global _last
    if cache_path is not None:
        if cache_path.exists():
            return cache_path.read_bytes()
        miss = cache_path.with_suffix(cache_path.suffix + ".404")
        if miss.exists():
            return None
    for attempt in range(retries):
        with _lock:
            wait = MIN_INTERVAL - (time.time() - _last)
            if wait > 0:
                time.sleep(wait)
            _last = time.time()
        try:
            r = _session.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(cache_path.suffix + f".tmp{threading.get_ident()}")
                tmp.write_bytes(r.content)
                tmp.replace(cache_path)  # 중단돼도 반쪽 파일이 캐시에 남지 않게
            return r.content
        if r.status_code == 404:
            if cache_path is not None:
                miss.parent.mkdir(parents=True, exist_ok=True)
                miss.touch()
            return None
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(2 ** (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"SEC GET failed after {retries} tries: {url}")


def get_json(url: str, cache_path: Path | None = None):
    b = get(url, cache_path)
    return None if b is None else json.loads(b)
