import json
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from lib.db import init_db, db, DB_PATH
from lib.prices import to_yahoo_ticker
from lib.news_rss import fetch_google_news_rss
from lib.scoring import indicator_features, indicator_score, combined_score

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHLIST_CSV = REPO_ROOT / "data" / "watchlist.csv"


def _flatten(df):
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
            df.columns = [c[0] for c in df.columns]
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


def yf_daily(symbol, exchange):
    t = to_yahoo_ticker(symbol, exchange)
    raw = yf.download(t, period="370d", interval="1d", progress=False, threads=True, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close", "volume"]), {"ticker": t, "rows": 0}

    df = _flatten(raw.reset_index())
    dcol = "Date" if "Date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.date.astype(str)

    close_s = df["Close"] if "Close" in df.columns else None
    vol_s = df["Volume"] if "Volume" in df.columns else None
    if close_s is None:
        return pd.DataFrame(columns=["date", "close", "volume"]), {"ticker": t, "rows": 0}

    if isinstance(close_s, pd.DataFrame):
        close_s = close_s.iloc[:, 0]
    if isinstance(vol_s, pd.DataFrame):
        vol_s = vol_s.iloc[:, 0]

    df["close"] = pd.to_numeric(close_s, errors="coerce")
    df["volume"] = pd.to_numeric(vol_s, errors="coerce") if vol_s is not None else None

    out = df[["date", "close", "volume"]].dropna(subset=["date", "close"])
    return out, {"ticker": t, "rows": int(len(out))}


def upsert_prices(symbol, exchange, df):
    if df is None or df.empty:
        return 0
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["date"]) or pd.isna(r["close"]):
            continue
        rows.append((symbol, exchange, str(r["date"]), float(r["close"]), float(r["volume"]) if pd.notna(r["volume"]) else None))
    if not rows:
        return 0
    with db() as con:
        con.executemany("INSERT OR REPLACE INTO prices(symbol, exchange, date, close, volume) VALUES (?,?,?,?,?)", rows)
        con.commit()
    return len(rows)


def upsert_news(symbol, exchange, name, days=60):
    items = fetch_google_news_rss(name, symbol, days=days) or []
    if not items:
        return 0
    with db() as con:
        con.executemany(
            "INSERT OR IGNORE INTO news_items(symbol, exchange, published, title, url, source) VALUES (?,?,?,?,?,?)",
            [(symbol, exchange, it["published"], it["title"], it["url"], it["source"]) for it in items],
        )
        con.commit()
    return len(items)


def upsert_signal(symbol, exchange, asof, score, reasons):
    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons) VALUES (?,?,?,?,?)",
            (symbol, exchange, asof, int(score), json.dumps(reasons)),
        )
        con.commit()


def load_watchlist():
    if not WATCHLIST_CSV.exists():
        return pd.DataFrame(columns=["symbol", "exchange"])
    df = pd.read_csv(WATCHLIST_CSV)
    if df.empty:
        return df
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["exchange"] = df["exchange"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["symbol","exchange"])


def main():
    init_db()
    asof = date.today().isoformat()
    wl = load_watchlist()

    print("DB_PATH =", DB_PATH)
    print("watchlist.csv rows =", len(wl))

    if wl.empty:
        print("No watchlist stocks. Nothing to update.")
        return

    # enrich names from universe if available
    with db() as con:
        uni = pd.read_sql_query("SELECT symbol, exchange, name FROM universe", con)

    wl2 = wl.merge(uni, on=["symbol","exchange"], how="left")
    wl2["name"] = wl2["name"].fillna(wl2["symbol"])

    for r in wl2.itertuples(index=False):
        symbol, exchange, name = r.symbol, r.exchange, str(r.name)

        px, meta = yf_daily(symbol, exchange)
        ins_px = upsert_prices(symbol, exchange, px)
        ins_news = upsert_news(symbol, exchange, name, days=60)

        with db() as con:
            dbpx = pd.read_sql_query(
                "SELECT date, close, volume FROM prices WHERE symbol=? AND exchange=? ORDER BY date",
                con,
                params=(symbol, exchange),
            )

        feats = indicator_features(dbpx)
        iscore, ibreak = indicator_score(feats)

        # temporary: keep news score 0 here (we’ll add sentiment next after prices flow)
        nscore = 0
        cscore = combined_score(iscore, nscore, 0.8, 0.2)

        reasons = {
            "source":"Daily update (Actions)",
            "prices":{"yahoo_ticker": meta["ticker"], "rows_fetched": meta["rows"], "rows_inserted": ins_px, "rows_db": int(len(dbpx))},
            "rss":{"rows_inserted": ins_news},
            "indicator": ibreak,
            "scores":{"indicator_score": iscore, "news_score": nscore, "combined_score": cscore, "w_ind":0.8, "w_news":0.2},
        }
        upsert_signal(symbol, exchange, asof, cscore, reasons)
        print(symbol, exchange, "yf_rows", meta["rows"], "ins_px", ins_px, "db_px", len(dbpx), "news_ins", ins_news)

    print("Done.")


if __name__ == "__main__":
    main()
