import feedparser
import datetime as dt
from urllib.parse import quote_plus

def google_news_rss_query(company_name: str, symbol: str):
    """
    Builds a Google News RSS query string.
    """
    base = company_name
    if symbol:
        base = f"{company_name} {symbol}"
    return quote_plus(base)

def fetch_google_news_rss(company_name: str, symbol: str, days: int = 60):
    """
    Fetch Google News RSS headlines for last N days.
    Returns list of dicts: published, title, link, source
    """
    q = google_news_rss_query(company_name, symbol)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

    feed = feedparser.parse(url)
    items = []

    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)

    for e in feed.entries:
        if not hasattr(e, "published_parsed"):
            continue

        published = dt.datetime(*e.published_parsed[:6])
        if published < cutoff:
            continue

        items.append({
            "published": published.date().isoformat(),
            "title": e.title,
            "url": e.link,
            "source": e.get("source", {}).get("title", "Google News")
        })

    return items
