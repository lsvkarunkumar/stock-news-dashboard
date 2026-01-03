import datetime as dt
from urllib.parse import quote_plus

import feedparser


def _google_news_rss_url(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def build_rss_query(company_name: str, symbol: str | None = None) -> str:
    name = (company_name or "").strip()
    sym = (symbol or "").strip().upper()
    if name and sym:
        return f'"{name}" OR "{sym}"'
    if name:
        return f'"{name}"'
    if sym:
        return f'"{sym}"'
    return ""


def fetch_google_news_rss(company_name: str, symbol: str, days: int = 60) -> list[dict]:
    query = build_rss_query(company_name, symbol)
    if not query:
        return []

    url = _google_news_rss_url(query)
    feed = feedparser.parse(url)

    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    items: list[dict] = []

    for e in getattr(feed, "entries", []) or []:
        published_dt = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            try:
                published_dt = dt.datetime(*e.published_parsed[:6])
            except Exception:
                published_dt = None
        if published_dt is None:
            published_dt = dt.datetime.utcnow()

        if published_dt < cutoff:
            continue

        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()

        src = "Google News"
        try:
            if hasattr(e, "source") and isinstance(e.source, dict):
                src = e.source.get("title", src) or src
            elif hasattr(e, "source") and hasattr(e.source, "title"):
                src = e.source.title or src
        except Exception:
            pass

        if title:
            items.append({
                "published": published_dt.date().isoformat(),
                "title": title,
                "url": link,
                "source": str(src).strip() if src else "Google News"
            })

    # Deduplicate within fetch
    seen = set()
    uniq = []
    for it in items:
        k = (it["published"], it["title"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)

    return uniq
