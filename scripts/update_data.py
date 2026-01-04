import json
from datetime import date

import pandas as pd
import yfinance as yf

from lib.db import init_db, db
from lib.prices import to_yahoo_ticker
from lib.news_rss import fetch_google_news_rss, headline_sentiment
from lib.scoring import indicator_features, indicator_score, combined_score


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        try:
            tickers = list(dict.fromkeys([c[1] for c in df.columns if len(c) > 1]))
            if tickers:
                df = df.xs(tickers[0], axis=1, level=1, drop_level=True)
        except Exception:
            pass
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns]
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


def yf_daily_try(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="370d", interval="1d", progress=False, threads=True, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = _flatten_yf_columns(raw.reset_index())
    dcol = "Date" if "Date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.date.astype(str)

    close_s = df["Close"] if "Close" in df.columns else None
    vol_s = df["Volume"] if "Volume" in df.columns else None
    if close_s is None:
        return pd.DataFrame(columns=["date", "close", "volume"])

    if isinstance(close_s, pd.DataFrame):
        close_s = close_s.iloc[:, 0]
    if isinstance(vol_s, pd.DataFrame):
        vol_s = vol_s.iloc[:, 0]

    df["close"] = pd.to_numeric(close_s, errors="coerce")
    df["volume"] = pd.to_numeric(vol_s, errors="coerce") if vol_s is not None else None
    return df[["date", "close", "volume"]].dropna(subset=["date", "close"])


def yf_daily(symbol: str, exchange: str):
    """
    Stronger fetch:
    1) Yahoo mapped ticker (e.g., RCOM.NS)
    2) If empty, try raw symbol (rare Yahoo variants)
    3) If empty, try alternate suffix (.NS/.BO swap)
    """
    t_main = to_yahoo_ticker(symbol, exchange)
    df = yf_daily_try(t_main)
    if not df.empty:
        return df, {"ticker_used": t_main, "fallback": "none", "rows": len(df)}

    # Try raw symbol (sometimes Yahoo uses special forms)
    df2 = yf_daily_try(symbol)
    if not df2.empty:
        return df2, {"ticker_used": symbol, "fallback": "raw_symbol", "rows": len(df2)}

    # Try alternate suffix swap
    if t_main.endswith(".NS"):
        alt = f"{symbol}.BO"
    elif t_main.endswith(".BO"):
        alt = f"{symbol}.NS"
    else:
        alt = t_main

    df3 = yf_daily_try(alt)
    if not df3.empty:
        return df3, {"ticker_used": alt, "fallback": "suffix_swap", "rows": len(df3)}

    return pd.DataFrame(columns=["date", "close", "volume"]), {"ticker_used": t_main, "fallback": "failed", "rows": 0}


def upsert_prices(symbol, exchange, df):
    if df is None or df.empty:
        return 0
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["date"]) or pd.isna(r["close"]):
            continue
        rows.append(
            (symbol, exchange, str(r["date"]), float(r["close"]), float(r["volume"]) if pd.notna(r["volume"]) else None)
        )
    if not rows:
        return 0
    with db() as con:
        con.executemany(
            "INSERT OR REPLACE INTO prices(symbol, exchange, date, close, volume) VALUES (?,?,?,?,?)",
            rows,
        )
        con.commit()
    return len(rows)


def upsert_news(symbol, exchange, name, days=60):
    items = fetch_google_news_rss(name, symbol, days=days) or []
    if not items:
        return 0
    with db() as con:
        con.executemany(
            """
            INSERT OR IGNORE INTO news_items(symbol, exchange, published, title, url, source)
            VALUES (?,?,?,?,?,?)
            """,
            [(symbol, exchange, it["published"], it["title"], it["url"], it["source"]) for it in items],
        )
        con.commit()
    return len(items)


def rss_counts_and_sentiment(symbol, exchange):
    """
    Reads stored news_items and computes:
    - m5, m60 counts
    - sentiment balance from titles (simple, free)
    """
    import datetime as dt
    from datetime import timedelta

    today = dt.date.today()
    with db() as con:
        df = pd.read_sql_query(
            "SELECT published, title, source FROM news_items WHERE symbol=? AND exchange=?",
            con,
            params=(symbol, exchange),
        )

    if df.empty:
        return 0, 0, 999, {"pos": 0, "neg": 0, "neu": 0, "label": "neutral"}

    df["published"] = pd.to_datetime(df["published"], errors="coerce").dt.date
    df = df.dropna(subset=["published"])
    if df.empty:
        return 0, 0, 999, {"pos": 0, "neg": 0, "neu": 0, "label": "neutral"}

    in5 = df[(today - df["published"]) <= timedelta(days=5)]
    in60 = df[(today - df["published"]) <= timedelta(days=60)]
    m5 = int(len(in5))
    m60 = int(len(in60))
    last = df["published"].max()
    recency_days = (today - last).days if last else 999

    # Sentiment from titles (simple keyword model)
    pos = neg = neu = 0
    for t in in60["title"].astype(str).tolist():
        s = headline_sentiment(t)
        if s["label"] == "positive":
            pos += 1
        elif s["label"] == "negative":
            neg += 1
        else:
            neu += 1

    label = "neutral"
    if pos >= neg + 3:
        label = "positive"
    elif neg >= pos + 3:
        label = "negative"

    return m5, m60, recency_days, {"pos": pos, "neg": neg, "neu": neu, "label": label}


def news_score(m5, m60, recency_days, sent_label):
    """
    Intensity/recency score (0..100), then adjust by sentiment label.
    Negative news should NOT become a buy trigger.
    """
    baseline = max(1.0, m60 / 12.0)
    intensity_ratio = m5 / baseline

    # 0..60
    intensity_pts = (intensity_ratio - 0.5) * 30.0
    intensity_pts = float(max(0.0, min(60.0, intensity_pts)))

    # 0..40
    if recency_days <= 2:
        recency_pts = 40
    elif recency_days <= 5:
        recency_pts = 30
    elif recency_days <= 10:
        recency_pts = 20
    elif recency_days <= 20:
        recency_pts = 10
    else:
        recency_pts = 0

    raw = float(min(100.0, intensity_pts + recency_pts))

    # Sentiment adjustment
    if sent_label == "negative":
        # cap hard: a negative spike should not look "great"
        adj = min(raw * 0.35, 35.0)
    elif sent_label == "neutral":
        adj = raw * 0.7
    else:
        adj = raw

    return int(max(0, min(100, round(adj))))


def upsert_signal(symbol, exchange, asof, combined_val, reasons):
    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons) VALUES (?,?,?,?,?)",
            (symbol, exchange, asof, int(combined_val), json.dumps(reasons)),
        )
        con.commit()


def main():
    init_db()
    asof = date.today().isoformat()

    with db() as con:
        wl = pd.read_sql_query(
            """
            SELECT w.symbol, w.exchange, COALESCE(u.name,w.symbol) AS name
            FROM watchlist w
            LEFT JOIN universe u ON u.symbol=w.symbol AND u.exchange=w.exchange
            """,
            con,
        )

    if wl.empty:
        print("Watchlist empty. Nothing to update.")
        return

    for r in wl.itertuples(index=False):
        symbol = str(r.symbol).strip().upper()
        exchange = str(r.exchange).strip().upper()
        name = str(r.name).strip()

        # Prices
        px, px_meta = yf_daily(symbol, exchange)
        inserted = upsert_prices(symbol, exchange, px)

        # News
        fetched = upsert_news(symbol, exchange, name, days=60)
        m5, m60, recency_days, sent = rss_counts_and_sentiment(symbol, exchange)
        nscore = news_score(m5, m60, recency_days, sent["label"])

        # IndicatorScore from DB prices
        with db() as con:
            dbpx = pd.read_sql_query(
                "SELECT date, close, volume FROM prices WHERE symbol=? AND exchange=? ORDER BY date",
                con,
                params=(symbol, exchange),
            )

        feats = indicator_features(dbpx)
        iscore, ibreak = indicator_score(feats)
        cscore = combined_score(iscore, nscore, 0.8, 0.2)

        reasons = {
            "source": "Daily update (Actions)",
            "name": name,
            "scores": {
                "indicator_score": iscore,
                "news_score": nscore,
                "combined_score": cscore,
                "w_ind": 0.8,
                "w_news": 0.2,
            },
            "prices": {
                "yahoo_ticker": px_meta.get("ticker_used"),
                "fallback": px_meta.get("fallback"),
                "rows_fetched": px_meta.get("rows"),
                "rows_inserted": inserted,
                "rows_db": int(len(dbpx)),
            },
            "rss": {
                "fetched": fetched,
                "m5": m5,
                "m60": m60,
                "recency_days": recency_days,
                "sentiment": sent,
            },
            "indicator": ibreak,
        }

        upsert_signal(symbol, exchange, asof, cscore, reasons)
        print(symbol, exchange, "PX_ins", inserted, "DB_px", len(dbpx), "Ind", iscore, "News", nscore, "Comb", cscore)

    print("Done.")


if __name__ == "__main__":
    main()
