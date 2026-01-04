import re
import feedparser
from datetime import date, timedelta

NEG_WORDS = {
    "loss","decline","falls","fall","down","plunge","crash","slump","weak","worst",
    "penalty","fine","fraud","scam","probe","raids","raid","ed","sebi","cbi",
    "lawsuit","case","ban","halt","suspend","default","bankrupt","bankruptcy",
    "debt","demand","notice","tax","gst","agr","downgrade","cut","cuts","resign",
    "warning","negative","selloff","sell-off","bearish","miss","misses"
}

POS_WORDS = {
    "gain","gains","up","surge","jump","rally","beats","beat","record","strong",
    "profit","growth","approval","cleared","order","contract","wins","win",
    "upgrade","buy","bullish","recovery","turnaround","investment","funding",
    "relief","package","deal","partnership","launch","expansion"
}

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def headline_sentiment(title: str) -> dict:
    t = _norm(title)
    if not t:
        return {"label": "neutral", "score": 0}

    toks = set(t.split(" "))
    neg = len([w for w in NEG_WORDS if w in toks or w in t])
    pos = len([w for w in POS_WORDS if w in toks or w in t])

    raw = pos - neg
    if raw >= 2:
        return {"label": "positive", "score": min(5, raw)}
    if raw <= -2:
        return {"label": "negative", "score": max(-5, raw)}
    return {"label": "neutral", "score": raw}

def fetch_google_news_rss(company_name: str, symbol: str, days: int = 60):
    """
    Pull Google News RSS for India. Free, no key.
    Returns list of dict: published, title, url, source, sentiment_label, sentiment_score
    """
    q = company_name if company_name else symbol
    q = q.strip()
    if not q:
        return []

    # Google News RSS query
    # hl=en-IN gl=IN ceid=IN:en gives India edition
    url = f"https://news.google.com/rss/search?q={q}+stock+OR+{symbol}&hl=en-IN&gl=IN&ceid=IN:en"

    feed = feedparser.parse(url)
    out = []
    cutoff = date.today() - timedelta(days=days)

    for e in feed.entries[:200]:
        title = getattr(e, "title", "") or ""
        link = getattr(e, "link", "") or ""
        published = getattr(e, "published", "") or getattr(e, "updated", "") or ""

        # best-effort date parse
        try:
            pd = feedparser._parse_date(published)
            if pd:
                pub_date = date(pd.tm_year, pd.tm_mon, pd.tm_mday)
            else:
                pub_date = date.today()
        except Exception:
            pub_date = date.today()

        if pub_date < cutoff:
            continue

        source = ""
        try:
            source = e.get("source", {}).get("title", "")
        except Exception:
            source = ""

        sent = headline_sentiment(title)
        out.append({
            "published": str(pub_date),
            "title": title,
            "url": link,
            "source": source or "Google News",
            "sentiment_label": sent["label"],
            "sentiment_score": int(sent["score"]),
        })

    return out
