import datetime as dt
import requests
import pandas as pd

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def gdelt_mentions(query: str, start: dt.date, end: dt.date) -> int:
    # Using DOC API with "mode=timelinevolraw" would be ideal, but simplest: count via "format=json" with maxrecords=1
    # We'll use "format=json&maxrecords=1&format=json&mode=artlist" and read "totalrecords" if present.
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": (end + dt.timedelta(days=1)).strftime("%Y%m%d%H%M%S"),
        "maxrecords": 1,
        "sort": "datedesc",
    }
    r = requests.get(GDELT_DOC_URL, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    js = r.json()
    return int(js.get("totalRecords", 0))

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
    r = requests.get(GDELT_DOC_URL, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    js = r.json()
    arts = js.get("articles", []) or []
    if not arts:
        return None, None
    a = arts[0]
    return a.get("title"), a.get("url")

def build_company_query(symbol: str, name: str) -> str:
    # Keep it simple in V1: use company name + symbol
    # You can add synonyms later (eg "Vodafone Idea" OR "Vi")
    safe_name = (name or "").replace('"', "")
    safe_symbol = (symbol or "").replace('"', "")
    if safe_name and safe_symbol:
        return f'"{safe_name}" OR "{safe_symbol}"'
    return f'"{safe_symbol}"' if safe_symbol else f'"{safe_name}"'
