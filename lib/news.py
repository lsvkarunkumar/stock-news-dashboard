import datetime as dt
import json
import requests

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def _gdelt_request(params: dict):
    r = requests.get(
        GDELT_DOC_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    r.raise_for_status()

    # Sometimes GDELT returns non-JSON (HTML error page, rate limit, etc.)
    try:
        return r.json()
    except Exception:
        # best-effort fallback: try to parse text if it's JSON-looking
        txt = (r.text or "").strip()
        try:
            return json.loads(txt)
        except Exception:
            return None

def gdelt_mentions(query: str, start: dt.date, end: dt.date) -> int:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": (end + dt.timedelta(days=1)).strftime("%Y%m%d%H%M%S"),
        "maxrecords": 1,
        "sort": "datedesc",
    }
    js = _gdelt_request(params)
    if not js:
        return 0
    return int(js.get("totalRecords", 0) or 0)

def gdelt_latest_headline(query: str, start: dt.date, end: dt.date):
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": (end + dt.timedelta(days=1)).strftime("%Y%m%d%H%M%S"),
        "maxrecords": 1,
        "sort": "datedesc",
    }
    js = _gdelt_request(params)
    if not js:
        return None, None
    arts = js.get("articles", []) or []
    if not arts:
        return None, None
    a = arts[0]
    return a.get("title"), a.get("url")

def build_company_query(symbol: str, name: str) -> str:
    safe_name = (name or "").replace('"', "")
    safe_symbol = (symbol or "").replace('"', "")
    if safe_name and safe_symbol:
        return f'"{safe_name}" OR "{safe_symbol}"'
    return f'"{safe_symbol}"' if safe_symbol else f'"{safe_name}"'
