import json
import math
import datetime as dt
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from lib.db import init_db, db, DB_PATH
from lib.universe import get_universe, fetch_universe, upsert_universe
from lib.prices import to_yahoo_ticker
from lib.news_rss import fetch_google_news_rss
from lib.scoring import indicator_features, indicator_score, combined_score

# Optional GitHub sync (only works if Streamlit secrets are set)
try:
    from lib.github_sync import github_put_file
except Exception:
    github_put_file = None

st.set_page_config(page_title="News × Price Dashboard", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent
WATCHLIST_CSV = REPO_ROOT / "data" / "watchlist.csv"


def load_css():
    try:
        with open("assets/style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


init_db()
load_css()

st.title("🧠 News × Price Dashboard")
st.caption("Auto-updated watchlist with price context + RSS news (educational). Final decision is yours.")

with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Discover & Add", "Watchlist", "Paper Trading"], index=0)
    st.divider()
    debug_on = st.toggle("Debug mode", value=False)


# ---------------- Helpers ----------------
def is_valid_url(x):
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    if not isinstance(x, str):
        return False
    x = x.strip()
    return x.startswith("http://") or x.startswith("https://")


def ensure_universe_loaded():
    uni = get_universe()
    if len(uni) > 0:
        return uni, False
    fresh = fetch_universe()
    upsert_universe(fresh)
    return get_universe(), True


def normalize_df(rows: pd.DataFrame) -> pd.DataFrame:
    clean = rows.copy()
    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()
    clean["exchange"] = clean["exchange"].astype(str).str.strip().str.upper()
    return clean.drop_duplicates(subset=["symbol", "exchange"])


def _extract_scores_from_reasons(reasons_val):
    ind = 0
    news = 0
    comb = 0
    try:
        if reasons_val is None or (isinstance(reasons_val, float) and math.isnan(reasons_val)):
            return ind, news, comb
        obj = reasons_val
        if isinstance(reasons_val, str):
            obj = json.loads(reasons_val)
        scores = obj.get("scores", {})
        ind = int(scores.get("indicator_score", 0) or 0)
        news = int(scores.get("news_score", 0) or 0)
        comb = int(scores.get("combined_score", 0) or 0)
    except Exception:
        pass
    return ind, news, comb


# ---------------- Watchlist (CSV) ----------------
def ensure_watchlist_file():
    WATCHLIST_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not WATCHLIST_CSV.exists():
        WATCHLIST_CSV.write_text("symbol,exchange\n", encoding="utf-8")


def read_watchlist_csv() -> pd.DataFrame:
    ensure_watchlist_file()
    try:
        df = pd.read_csv(WATCHLIST_CSV)
    except Exception:
        df = pd.DataFrame(columns=["symbol", "exchange"])
    if df.empty:
        return pd.DataFrame(columns=["symbol", "exchange"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["exchange"] = df["exchange"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["symbol", "exchange"]).reset_index(drop=True)


def write_watchlist_csv(df: pd.DataFrame):
    ensure_watchlist_file()
    df = normalize_df(df)
    df.to_csv(WATCHLIST_CSV, index=False)


def sync_watchlist_to_github(df: pd.DataFrame, message: str):
    # If secrets not provided, just keep local CSV (manual mode)
    if github_put_file is None:
        return {"ok": False, "error": "github_sync not available"}
    try:
        csv_text = df.to_csv(index=False)
        res = github_put_file("data/watchlist.csv", csv_text, message)
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}


def add_to_watchlist_bulk(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return
    new_rows = normalize_df(rows)[["symbol", "exchange"]]
    wl = read_watchlist_csv()
    wl2 = pd.concat([wl, new_rows], axis=0, ignore_index=True).drop_duplicates(subset=["symbol", "exchange"])
    write_watchlist_csv(wl2)

    # Try auto-sync to GitHub if secrets exist
    if github_put_file is not None:
        sync_watchlist_to_github(wl2, "Update watchlist.csv (add)")


def remove_from_watchlist_bulk(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return
    rem = normalize_df(rows)[["symbol", "exchange"]]
    wl = read_watchlist_csv()
    if wl.empty:
        return
    merged = wl.merge(rem.assign(_rm=1), on=["symbol", "exchange"], how="left")
    wl2 = merged[merged["_rm"].isna()][["symbol", "exchange"]].copy()
    write_watchlist_csv(wl2)

    if github_put_file is not None:
        sync_watchlist_to_github(wl2, "Update watchlist.csv (delete)")


def get_watchlist_df():
    wl = read_watchlist_csv()
    if wl.empty:
        return wl

    uni = get_universe().copy()
    if uni is None or uni.empty:
        uni, _ = ensure_universe_loaded()

    if "sector" not in uni.columns:
        uni["sector"] = "Unknown"
    uni["sector"] = uni["sector"].fillna("Unknown")

    uni["symbol"] = uni["symbol"].astype(str).str.strip().str.upper()
    uni["exchange"] = uni["exchange"].astype(str).str.strip().str.upper()

    df = wl.merge(uni[["symbol", "exchange", "name", "sector"]], on=["symbol", "exchange"], how="left")
    df["name"] = df["name"].fillna(df["symbol"])
    df["sector"] = df["sector"].fillna("Unknown")
    df["added_at"] = ""  # optional; kept for layout compatibility
    return df


# ---------------- Signals/Prices/News in DB ----------------
def latest_signals_per_stock():
    with db() as con:
        return pd.read_sql_query(
            """
            SELECT s.*
            FROM signals s
            JOIN (
                SELECT symbol, exchange, MAX(asof) AS max_asof
                FROM signals
                GROUP BY symbol, exchange
            ) t
            ON s.symbol=t.symbol AND s.exchange=t.exchange AND s.asof=t.max_asof
            """,
            con,
        )


def upsert_signal(symbol, exchange, asof, combined_val, reasons):
    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons) VALUES (?,?,?,?,?)",
            (symbol, exchange, asof, int(combined_val), json.dumps(reasons)),
        )
        con.commit()


def get_prices_from_db(symbol, exchange):
    with db() as con:
        return pd.read_sql_query(
            "SELECT date, close, volume FROM prices WHERE symbol=? AND exchange=? ORDER BY date",
            con,
            params=(symbol, exchange),
        )


# ---------------- Market data fallback (display only) ----------------
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


def yf_daily_fallback(symbol: str, exchange: str) -> pd.DataFrame:
    t = to_yahoo_ticker(symbol, exchange)
    raw = yf.download(t, period="370d", interval="1d", progress=False, threads=True, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = _flatten_yf_columns(raw.reset_index())
    dcol = "Date" if "Date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.date.astype(str)
    close = df["Close"] if "Close" in df.columns else None
    vol = df["Volume"] if "Volume" in df.columns else None
    if close is None:
        return pd.DataFrame(columns=["date", "close", "volume"])
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    df["close"] = pd.to_numeric(close, errors="coerce")
    df["volume"] = pd.to_numeric(vol, errors="coerce") if vol is not None else np.nan
    return df[["date", "close", "volume"]].dropna(subset=["date", "close"])


# --------- Metrics (DB first) ---------
def compute_metrics(prices: pd.DataFrame) -> dict:
    out = {
        "live": np.nan,
        "52w_high": np.nan,
        "52w_low": np.nan,
        "ret_1d": np.nan,
        "ret_1w": np.nan,
        "ret_1m": np.nan,
        "ret_3m": np.nan,
        "ret_6m": np.nan,
        "ret_12m": np.nan,
    }
    if prices is None or prices.empty:
        return out

    df = prices.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date_dt", "close"]).sort_values("date_dt")
    if df.empty:
        return out

    last_close = float(df["close"].iloc[-1])
    out["live"] = last_close

    cutoff = df["date_dt"].iloc[-1] - pd.Timedelta(days=365)
    d52 = df[df["date_dt"] >= cutoff]
    if not d52.empty:
        out["52w_high"] = float(d52["close"].max())
        out["52w_low"] = float(d52["close"].min())

    last_dt = df["date_dt"].iloc[-1]

    def ret(days):
        target = last_dt - pd.Timedelta(days=days)
        base_rows = df[df["date_dt"] <= target]
        if base_rows.empty:
            return np.nan
        base = float(base_rows["close"].iloc[-1])
        if base == 0:
            return np.nan
        return (last_close / base - 1.0) * 100.0

    out["ret_1d"] = ret(1)
    out["ret_1w"] = ret(7)
    out["ret_1m"] = ret(30)
    out["ret_3m"] = ret(90)
    out["ret_6m"] = ret(180)
    out["ret_12m"] = ret(365)
    return out


def enrich_with_metrics_db(df: pd.DataFrame, max_rows: int = 50) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    take = df.head(max_rows).copy()
    metrics = []
    for r in take.itertuples(index=False):
        px = get_prices_from_db(r.symbol, r.exchange)
        if px.empty:
            px = yf_daily_fallback(r.symbol, r.exchange)  # display-only fallback
        metrics.append(compute_metrics(px))
    met = pd.DataFrame(metrics)
    take2 = pd.concat([take.reset_index(drop=True), met], axis=1)
    rest = df.iloc[max_rows:].copy()
    return pd.concat([take2, rest], axis=0, ignore_index=True)


# ---------------- RSS news -> DB ----------------
def upsert_news_items(symbol: str, exchange: str, company_name: str, days: int = 60) -> int:
    items = fetch_google_news_rss(company_name, symbol, days=days)
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


def rss_counts(symbol: str, exchange: str):
    today = dt.date.today()
    with db() as con:
        df = pd.read_sql_query(
            "SELECT published FROM news_items WHERE symbol=? AND exchange=?",
            con,
            params=(symbol, exchange),
        )
    if df.empty:
        return 0, 0, 999
    df["published"] = pd.to_datetime(df["published"], errors="coerce").dt.date
    df = df.dropna(subset=["published"])
    if df.empty:
        return 0, 0, 999
    m5 = int((today - df["published"] <= dt.timedelta(days=5)).sum())
    m60 = int((today - df["published"] <= dt.timedelta(days=60)).sum())
    last = df["published"].max()
    recency_days = (today - last).days if last else 999
    return m5, m60, recency_days


def news_score_from_rss(m5: int, m60: int, recency_days: int) -> int:
    baseline = max(1.0, m60 / 12.0)
    intensity_ratio = m5 / baseline
    intensity_pts = (intensity_ratio - 0.5) * 30.0
    intensity_pts = float(max(0.0, min(60.0, intensity_pts)))

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

    score = int(round(min(100.0, intensity_pts + recency_pts)))
    return max(0, min(100, score))


def latest_rss_headlines(symbol: str, exchange: str, limit: int = 60) -> pd.DataFrame:
    with db() as con:
        return pd.read_sql_query(
            """
            SELECT published, title, url, source
            FROM news_items
            WHERE symbol=? AND exchange=?
            ORDER BY published DESC
            LIMIT ?
            """,
            con,
            params=(symbol, exchange, int(limit)),
        )


# ---------------- Refresh (Indicators + RSS) ----------------
def refresh_one_stock(symbol: str, exchange: str, name: str):
    symbol = str(symbol).strip().upper()
    exchange = str(exchange).strip().upper()
    name = str(name or symbol).strip()
    asof = date.today().isoformat()

    fetched = 0
    try:
        fetched = upsert_news_items(symbol, exchange, name, days=60)
    except Exception:
        fetched = 0

    m5, m60, recency_days = rss_counts(symbol, exchange)
    nscore = news_score_from_rss(m5, m60, recency_days)

    dbpx = get_prices_from_db(symbol, exchange)
    feats = indicator_features(dbpx)
    iscore, ibreak = indicator_score(feats)
    cscore = combined_score(iscore, nscore, 0.8, 0.2)

    reasons = {
        "source": "Manual refresh",
        "scores": {"indicator_score": iscore, "news_score": nscore, "combined_score": cscore, "w_ind": 0.8, "w_news": 0.2},
        "rss": {"fetched": fetched, "m5": m5, "m60": m60, "recency_days": recency_days},
        "indicator": ibreak,
        "yahoo_ticker": to_yahoo_ticker(symbol, exchange),
        "price_rows_db": int(len(dbpx)),
        "watchlist_source": "data/watchlist.csv",
    }
    upsert_signal(symbol, exchange, asof, cscore, reasons)
    return {"ok": True, "indicator_score": iscore, "news_score": nscore, "combined_score": cscore}


# ---------------- Debug panel ----------------
if debug_on:
    with st.expander("🧪 Debug panel", expanded=False):
        st.write("DB path:", str(DB_PATH))
        st.write("Watchlist CSV:", str(WATCHLIST_CSV))
        wl_now = read_watchlist_csv()
        st.write("watchlist.csv rows:", int(len(wl_now)))
        with db() as con:
            c_univ = con.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
            c_px = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            c_news = con.execute("SELECT COUNT(*) FROM news_items").fetchone()[0]
            c_sig = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        st.write({"universe": c_univ, "prices": c_px, "news_items": c_news, "signals": c_sig})


# ========================= PAGES =========================
if page == "Discover & Add":
    st.subheader("Discover & Add")

    uni, _ = ensure_universe_loaded()
    uni = uni.copy()
    if "sector" not in uni.columns:
        uni["sector"] = "Unknown"
    uni["sector"] = uni["sector"].fillna("Unknown")

    uni["symbol"] = uni["symbol"].astype(str).str.strip().str.upper()
    uni["exchange"] = uni["exchange"].astype(str).str.strip().str.upper()

    c1, c2, c3 = st.columns([3, 4, 2])
    with c1:
        q = st.text_input("Search", placeholder="Type… (e.g., RELIANCE)")
    with c2:
        sectors_all = sorted([s for s in uni["sector"].astype(str).unique()])
        sectors = st.multiselect("Sector (multi-select)", options=sectors_all, default=[])
    with c3:
        st.metric("Universe", f"{len(uni):,}")

    show = uni
    if sectors:
        show = show[show["sector"].isin(sectors)]
    if q:
        mask = (
            show["symbol"].str.contains(q, case=False, na=False)
            | show["name"].astype(str).str.contains(q, case=False, na=False)
        )
        show = show[mask]
    show = show.head(250).copy()

    sig = latest_signals_per_stock()
    if not sig.empty:
        show = show.merge(sig[["symbol", "exchange", "score", "reasons"]], on=["symbol", "exchange"], how="left")
    else:
        show["score"] = 0
        show["reasons"] = None

    show["combined_score"] = show["score"].fillna(0).astype(int)
    inds, news = [], []
    for rv in show["reasons"].tolist():
        i, n, c = _extract_scores_from_reasons(rv)
        inds.append(i)
        news.append(n)
    show["indicator_score"] = inds
    show["news_score"] = news

    show.insert(0, "add", False)
    show = enrich_with_metrics_db(show, max_rows=50)

    add_btn = st.button("➕ Add selected to Watchlist", use_container_width=True)

    edited = st.data_editor(
        show[["add","exchange","symbol","name","sector","indicator_score","news_score","combined_score",
              "live","52w_high","52w_low","ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"]],
        use_container_width=True,
        height=560,
        hide_index=True,
        column_config={
            "add": st.column_config.CheckboxColumn("Add"),
            "indicator_score": st.column_config.NumberColumn("Indicator", format="%d"),
            "news_score": st.column_config.NumberColumn("News", format="%d"),
            "combined_score": st.column_config.NumberColumn("Combined", format="%d"),
            "live": st.column_config.NumberColumn("Live", format="%.2f"),
            "52w_high": st.column_config.NumberColumn("52W High", format="%.2f"),
            "52w_low": st.column_config.NumberColumn("52W Low", format="%.2f"),
            "ret_1d": st.column_config.NumberColumn("1D %", format="%.2f"),
            "ret_1w": st.column_config.NumberColumn("1W %", format="%.2f"),
            "ret_1m": st.column_config.NumberColumn("1M %", format="%.2f"),
            "ret_3m": st.column_config.NumberColumn("3M %", format="%.2f"),
            "ret_6m": st.column_config.NumberColumn("6M %", format="%.2f"),
            "ret_12m": st.column_config.NumberColumn("12M %", format="%.2f"),
        },
    )

    selected = edited[edited["add"] == True][["symbol", "exchange"]].drop_duplicates()
    st.info(f"Selected: **{len(selected)}**")

    if add_btn:
        add_to_watchlist_bulk(selected)
        st.success("Added to watchlist (watchlist.csv). If GH secrets are set, it auto-syncs to GitHub.")
        st.rerun()


elif page == "Watchlist":
    st.subheader("Watchlist (live stats + delete + story)")

    wl = get_watchlist_df()
    if wl.empty:
        st.info("Watchlist is empty. Add from Discover.")
        st.stop()

    sig = latest_signals_per_stock()
    if not sig.empty:
        df = wl.merge(sig[["symbol", "exchange", "score", "reasons", "asof"]], on=["symbol", "exchange"], how="left")
    else:
        df = wl.copy()
        df["score"] = 0
        df["reasons"] = None
        df["asof"] = None

    df["combined_score"] = df["score"].fillna(0).astype(int)
    inds, news = [], []
    for rv in df["reasons"].tolist():
        i, n, c = _extract_scores_from_reasons(rv)
        inds.append(i)
        news.append(n)
    df["indicator_score"] = inds
    df["news_score"] = news

    df = enrich_with_metrics_db(df, max_rows=50)
    df = df.sort_values(["combined_score"], ascending=False).copy()
    df.insert(0, "delete", False)

    del_btn = st.button("🗑️ Delete selected", use_container_width=True)

    edited = st.data_editor(
        df[["delete","exchange","symbol","name","sector","indicator_score","news_score","combined_score",
            "live","52w_high","52w_low","ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"]],
        use_container_width=True,
        height=520,
        hide_index=True,
        column_config={"delete": st.column_config.CheckboxColumn("Del")},
    )

    to_del = edited[edited["delete"] == True][["symbol", "exchange"]].drop_duplicates()
    if del_btn:
        remove_from_watchlist_bulk(to_del)
        st.success(f"Deleted {len(to_del)} item(s) from watchlist.csv.")
        st.rerun()

    st.divider()
    st.subheader("Stock story")

    options = edited.apply(lambda r: f'{r["symbol"]} ({r["exchange"]}) — {r["name"]}', axis=1).tolist()
    pick = st.selectbox("Select stock", options)

    sel_sym = pick.split(" ")[0].strip().upper()
    sel_ex = pick.split("(")[1].split(")")[0].strip().upper()
    row = df[(df.symbol == sel_sym) & (df.exchange == sel_ex)].iloc[0]

    st.markdown(f"### **{row['name']}** — {sel_sym} ({sel_ex})")
    st.caption("Combined = 0.8×Indicator + 0.2×News")

    if st.button("🔄 Refresh scores now (Indicators + RSS)", use_container_width=True):
        res = refresh_one_stock(sel_sym, sel_ex, row["name"])
        st.success(f"Refreshed: Indicator={res['indicator_score']}  News={res['news_score']}  Combined={res['combined_score']}")
        st.rerun()

    st.markdown("#### 🧠 Why these scores?")
    if row.get("reasons"):
        try:
            st.json(json.loads(row["reasons"]))
        except Exception:
            st.info("Score breakdown not available yet. Run Actions once more.")

    st.markdown("#### 📰 Latest RSS headlines (grouped by date)")
    h = latest_rss_headlines(sel_sym, sel_ex, limit=60)
    if h is None or h.empty:
        st.info("No RSS headlines stored yet. Click refresh once.")
    else:
        h = h.copy()
        h["published"] = pd.to_datetime(h["published"], errors="coerce").dt.date
        h = h.dropna(subset=["published"])
        if h.empty:
            st.info("No valid-dated headlines available.")
        else:
            h["_t"] = h["title"].astype(str).str.strip().str.lower()
            h = h.drop_duplicates(subset=["published", "_t"]).drop(columns=["_t"])
            latest_date = h["published"].max()
            for pub_date in sorted(h["published"].unique(), reverse=True):
                day_items = h[h["published"] == pub_date].copy().sort_values(["source", "title"])
                with st.expander(f"📅 {pub_date} — {len(day_items)} headlines", expanded=(pub_date == latest_date)):
                    for _, rr in day_items.iterrows():
                        st.write(f"• {rr['title']}  ({rr['source']})")
                        if is_valid_url(rr["url"]):
                            st.link_button("Open", rr["url"])

    st.markdown("#### 📈 Price chart (from DB)")
    px = get_prices_from_db(sel_sym, sel_ex)
    if px.empty:
        st.warning("No DB price rows yet. Run GitHub Actions after watchlist.csv is synced to GitHub.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name="Price"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


else:
    st.subheader("Paper Trading (demo)")
    with db() as con:
        cash = float(con.execute("SELECT cash FROM paper_wallet WHERE id=1").fetchone()[0])
        trades = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY ts DESC LIMIT 200", con)
    st.metric("Demo cash", round(cash, 2))
    st.dataframe(trades, use_container_width=True, height=420)
