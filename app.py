import json
import math
import datetime as dt
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from lib.db import init_db, db, DB_PATH
from lib.universe import get_universe, fetch_universe, upsert_universe
from lib.prices import to_yahoo_ticker
from lib.news import build_company_query, gdelt_mentions, gdelt_latest_headline
from lib.scoring import compute_score

st.set_page_config(page_title="News × Price Dashboard", layout="wide")

# ---------------- Init ----------------
def load_css():
    try:
        with open("assets/style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

init_db()
load_css()

st.title("🧠 News × Price Dashboard")
st.caption("Auto-updated watchlist with news intensity & price context (educational). Final decision is yours.")

with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Discover & Add", "Watchlist", "Paper Trading"], index=0)
    st.divider()
    debug_on = st.toggle("Debug mode", value=False)
    st.caption("Live here = latest available from free feeds (may be delayed).")

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

def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)

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

# ---------------- DB: watchlist ----------------
def add_to_watchlist_bulk(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return
    clean = normalize_df(rows)
    with db() as con:
        con.executemany(
            "INSERT OR IGNORE INTO watchlist(symbol, exchange, added_at) VALUES (?,?,?)",
            [(r.symbol, r.exchange, date.today().isoformat()) for r in clean.itertuples(index=False)]
        )
        con.commit()

def remove_from_watchlist_bulk(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return
    clean = normalize_df(rows)
    with db() as con:
        con.executemany(
            "DELETE FROM watchlist WHERE symbol=? AND exchange=?",
            [(r.symbol, r.exchange) for r in clean.itertuples(index=False)]
        )
        con.commit()

def get_watchlist_df():
    with db() as con:
        return pd.read_sql_query("""
            SELECT w.symbol, w.exchange, u.name, COALESCE(u.sector,'Unknown') AS sector, COALESCE(u.isin,'') AS isin, w.added_at
            FROM watchlist w
            LEFT JOIN universe u ON u.symbol=w.symbol AND u.exchange=w.exchange
            ORDER BY w.added_at DESC
        """, con)

# ---------------- DB: latest per stock (IMPORTANT FIX) ----------------
def latest_signals_per_stock():
    with db() as con:
        return pd.read_sql_query("""
            SELECT s.*
            FROM signals s
            JOIN (
                SELECT symbol, exchange, MAX(asof) AS max_asof
                FROM signals
                GROUP BY symbol, exchange
            ) t
            ON s.symbol=t.symbol AND s.exchange=t.exchange AND s.asof=t.max_asof
        """, con)

def latest_news_per_stock():
    with db() as con:
        return pd.read_sql_query("""
            SELECT n.*
            FROM news_mentions n
            JOIN (
                SELECT symbol, exchange, MAX(date) AS max_date
                FROM news_mentions
                GROUP BY symbol, exchange
            ) t
            ON n.symbol=t.symbol AND n.exchange=t.exchange AND n.date=t.max_date
        """, con)

def get_prices_from_db(symbol, exchange):
    with db() as con:
        return pd.read_sql_query("""
            SELECT date, close, volume
            FROM prices
            WHERE symbol=? AND exchange=?
            ORDER BY date
        """, con, params=(symbol, exchange))

def upsert_prices(symbol, exchange, df):
    if df is None or df.empty:
        return
    with db() as con:
        con.executemany("""
            INSERT OR REPLACE INTO prices(symbol, exchange, date, close, volume)
            VALUES (?,?,?,?,?)
        """, [
            (symbol, exchange, r.date,
             float(r.close) if pd.notna(r.close) else None,
             float(r.volume) if pd.notna(r.volume) else None)
            for r in df.itertuples(index=False)
        ])
        con.commit()

def upsert_news(symbol, exchange, asof, mentions, headline, url):
    with db() as con:
        con.execute("""
            INSERT OR REPLACE INTO news_mentions(symbol, exchange, date, mentions, sample_headline, sample_url)
            VALUES (?,?,?,?,?,?)
        """, (symbol, exchange, asof, int(mentions), headline, url))
        con.commit()

def upsert_signal(symbol, exchange, asof, score, reasons):
    with db() as con:
        con.execute("""
            INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons)
            VALUES (?,?,?,?,?)
        """, (symbol, exchange, asof, int(score), reasons))
        con.commit()

# ---------------- Market data (live + returns) ----------------
def yf_history_force(symbol: str, exchange: str) -> pd.DataFrame:
    """Try intraday 1m first, fallback daily."""
    t = to_yahoo_ticker(symbol, exchange)

    # Intraday 1m (best free "live-like" option)
    try:
        df = yf.download(t, period="1d", interval="1m", progress=False, threads=True)
        if df is not None and not df.empty:
            df = df.reset_index()
            dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
            df["date"] = pd.to_datetime(df[dt_col]).dt.strftime("%Y-%m-%d %H:%M")
            df = df.rename(columns={"Close": "close", "Volume": "volume"})
            return df[["date", "close", "volume"]]
    except Exception:
        pass

    # Daily fallback
    df = yf.download(t, period="370d", interval="1d", progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    return df[["date", "close", "volume"]]

@st.cache_data(ttl=5*60, show_spinner=False)
def yf_history_cached_daily(symbol: str, exchange: str) -> pd.DataFrame:
    t = to_yahoo_ticker(symbol, exchange)
    df = yf.download(t, period="370d", interval="1d", progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    return df[["date", "close", "volume"]]

def compute_metrics(df: pd.DataFrame) -> dict:
    out = {
        "live": np.nan, "52w_high": np.nan, "52w_low": np.nan,
        "ret_1d": np.nan, "ret_1w": np.nan, "ret_1m": np.nan,
        "ret_3m": np.nan, "ret_6m": np.nan, "ret_12m": np.nan,
    }
    if df is None or df.empty or df["close"].isna().all():
        return out

    d = df.copy()
    d["date_dt"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date_dt", "close"]).sort_values("date_dt")
    if d.empty:
        return out

    last_close = float(d["close"].iloc[-1])
    out["live"] = last_close

    cutoff = d["date_dt"].iloc[-1] - pd.Timedelta(days=365)
    d52 = d[d["date_dt"] >= cutoff]
    if not d52.empty:
        out["52w_high"] = float(d52["close"].max())
        out["52w_low"] = float(d52["close"].min())

    last_dt = d["date_dt"].iloc[-1]

    def ret(days):
        target = last_dt - pd.Timedelta(days=days)
        base_rows = d[d["date_dt"] <= target]
        if base_rows.empty:
            return np.nan
        base = float(base_rows["close"].iloc[-1])
        if base == 0:
            return np.nan
        return (last_close / base - 1.0) * 100.0

    out["ret_1d"]  = ret(1)
    out["ret_1w"]  = ret(7)
    out["ret_1m"]  = ret(30)
    out["ret_3m"]  = ret(90)
    out["ret_6m"]  = ret(180)
    out["ret_12m"] = ret(365)
    return out

def enrich_with_metrics(df: pd.DataFrame, max_rows: int = 30) -> pd.DataFrame:
    """Adds live/52W/returns for top rows (speed-safe)."""
    if df.empty:
        return df
    take = df.head(max_rows).copy()
    metrics = []
    for r in take.itertuples(index=False):
        h = yf_history_cached_daily(r.symbol, r.exchange)
        metrics.append(compute_metrics(h))
    met = pd.DataFrame(metrics)
    out = pd.concat([take.reset_index(drop=True), met], axis=1)
    # merge back
    rest = df.iloc[max_rows:].copy()
    return pd.concat([out, rest], axis=0, ignore_index=True)

# ---------------- Refresh pipeline ----------------
def refresh_one_stock(symbol: str, exchange: str, name: str):
    symbol = str(symbol).strip().upper()
    exchange = str(exchange).strip().upper()
    asof = date.today().isoformat()

    # avoid stale caches
    try:
        st.cache_data.clear()
    except Exception:
        pass

    px = yf_history_force(symbol, exchange)
    upsert_prices(symbol, exchange, px)

    try:
        d5 = date.today() - dt.timedelta(days=5)
        d30 = date.today() - dt.timedelta(days=30)
        d60 = date.today() - dt.timedelta(days=60)

        q = build_company_query(symbol, name)
        m5 = gdelt_mentions(q, d5, date.today())
        m30 = gdelt_mentions(q, d30, date.today())
        m60 = gdelt_mentions(q, d60, date.today())
        headline, url = gdelt_latest_headline(q, d60, date.today())

        upsert_news(symbol, exchange, asof, m60, headline, url)

        score, reasons = compute_score(m5, m30, m60, px)
        upsert_signal(symbol, exchange, asof, score, reasons)
        return {"ok": True}
    except Exception as e:
        # still write a row so UI updates
        score, reasons = compute_score(0, 0, 0, px)
        upsert_signal(symbol, exchange, asof, score, reasons)
        return {"ok": True, "warn": f"News failed: {type(e).__name__}"}

# ---------------- Debug panel ----------------
if debug_on:
    with st.expander("🧪 Debug panel", expanded=False):
        st.write("DB path:", str(DB_PATH))
        with db() as con:
            c_univ = con.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
            c_wl = con.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
            c_px = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            c_news = con.execute("SELECT COUNT(*) FROM news_mentions").fetchone()[0]
            c_sig = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        st.write({"universe": c_univ, "watchlist": c_wl, "prices": c_px, "news_mentions": c_news, "signals": c_sig})

# ========================= PAGES =========================
if page == "Discover & Add":
    st.subheader("Discover & Add")

    uni, _ = ensure_universe_loaded()
    uni = uni.copy()
    uni["sector"] = uni.get("sector", "Unknown")
    if "sector" in uni.columns:
        uni["sector"] = uni["sector"].fillna("Unknown")
    else:
        uni["sector"] = "Unknown"

    uni["symbol"] = uni["symbol"].astype(str).str.strip().str.upper()
    uni["exchange"] = uni["exchange"].astype(str).str.strip().str.upper()

    # Adaptive search (no need to press enter; it re-runs automatically)
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
            show["symbol"].str.contains(q, case=False, na=False) |
            show["name"].astype(str).str.contains(q, case=False, na=False)
        )
        show = show[mask]

    show = show.head(250).copy()

    # Add score into universe (latest per stock)
    sig = latest_signals_per_stock()
    show = show.merge(sig[["symbol","exchange","score"]], on=["symbol","exchange"], how="left")
    show["score"] = show["score"].fillna(0).astype(int)

    show.insert(0, "add", False)

    add_btn = st.button("➕ Add selected to Watchlist", use_container_width=True)

    # Add premium metrics for top rows
    show = enrich_with_metrics(show, max_rows=30)

    edited = st.data_editor(
        show[[
            "add","exchange","symbol","name","sector",
            "score",
            "live","52w_high","52w_low",
            "ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"
        ]],
        use_container_width=True,
        height=560,
        hide_index=True,
        column_config={
            "add": st.column_config.CheckboxColumn("Add"),
            "score": st.column_config.NumberColumn("NewsScore", format="%d"),
            "live": st.column_config.NumberColumn("Live", format="%.2f"),
            "52w_high": st.column_config.NumberColumn("52W High", format="%.2f"),
            "52w_low": st.column_config.NumberColumn("52W Low", format="%.2f"),
            "ret_1d": st.column_config.NumberColumn("1D %", format="%.2f"),
            "ret_1w": st.column_config.NumberColumn("1W %", format="%.2f"),
            "ret_1m": st.column_config.NumberColumn("1M %", format="%.2f"),
            "ret_3m": st.column_config.NumberColumn("3M %", format="%.2f"),
            "ret_6m": st.column_config.NumberColumn("6M %", format="%.2f"),
            "ret_12m": st.column_config.NumberColumn("12M %", format="%.2f"),
        }
    )

    selected = edited[edited["add"] == True][["symbol","exchange"]].drop_duplicates()
    st.info(f"Selected: **{len(selected)}**")

    if add_btn:
        add_to_watchlist_bulk(selected)
        st.success("Added to watchlist.")
        st.rerun()

elif page == "Watchlist":
    st.subheader("Watchlist (live stats + delete + story)")

    wl = get_watchlist_df()
    if wl.empty:
        st.info("Watchlist is empty. Add from Discover.")
        st.stop()

    sectors = sorted([s for s in wl["sector"].astype(str).unique()])
    pick_sectors = st.multiselect("Filter by sector", options=sectors, default=[])
    view = wl.copy()
    if pick_sectors:
        view = view[view["sector"].isin(pick_sectors)].copy()

    sig = latest_signals_per_stock()
    news = latest_news_per_stock()

    df = view.merge(sig[["symbol","exchange","score","reasons","asof"]], on=["symbol","exchange"], how="left") \
             .merge(news[["symbol","exchange","mentions","sample_headline","sample_url","date"]], on=["symbol","exchange"], how="left")

    df["score"] = df["score"].fillna(0).astype(int)
    df["mentions"] = df["mentions"].fillna(0).astype(int)

    # premium metrics for top rows
    df = enrich_with_metrics(df, max_rows=30)

    df = df.sort_values(["score","mentions"], ascending=False).copy()
    df.insert(0, "delete", False)

    c1, c2 = st.columns([2, 4])
    with c1:
        del_btn = st.button("🗑️ Delete selected", use_container_width=True)
    with c2:
        st.caption("Tip: Use Story panel refresh to force-update score/news/prices for that stock.")

    edited = st.data_editor(
        df[[
            "delete","exchange","symbol","name","sector",
            "score","mentions",
            "live","52w_high","52w_low",
            "ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m",
            "sample_headline"
        ]],
        use_container_width=True,
        height=520,
        hide_index=True,
        column_config={
            "delete": st.column_config.CheckboxColumn("Del"),
            "score": st.column_config.NumberColumn("NewsScore", format="%d"),
            "mentions": st.column_config.NumberColumn("Mentions", format="%d"),
            "live": st.column_config.NumberColumn("Live", format="%.2f"),
            "52w_high": st.column_config.NumberColumn("52W High", format="%.2f"),
            "52w_low": st.column_config.NumberColumn("52W Low", format="%.2f"),
            "ret_1d": st.column_config.NumberColumn("1D %", format="%.2f"),
            "ret_1w": st.column_config.NumberColumn("1W %", format="%.2f"),
            "ret_1m": st.column_config.NumberColumn("1M %", format="%.2f"),
            "ret_3m": st.column_config.NumberColumn("3M %", format="%.2f"),
            "ret_6m": st.column_config.NumberColumn("6M %", format="%.2f"),
            "ret_12m": st.column_config.NumberColumn("12M %", format="%.2f"),
        }
    )

    to_del = edited[edited["delete"] == True][["symbol","exchange"]].drop_duplicates()
    if del_btn:
        remove_from_watchlist_bulk(to_del)
        st.success(f"Deleted {len(to_del)} item(s).")
        st.rerun()

    st.divider()
    st.subheader("Stock story")

    options = edited.apply(lambda r: f'{r["symbol"]} ({r["exchange"]}) — {r["name"]}', axis=1).tolist()
    pick = st.selectbox("Select stock", options)

    sel_sym = pick.split(" ")[0].strip().upper()
    sel_ex = pick.split("(")[1].split(")")[0].strip().upper()

    row = df[(df.symbol == sel_sym) & (df.exchange == sel_ex)].iloc[0]
    st.markdown(f"### **{row['name']}** — {sel_sym} ({sel_ex})")

    if st.button("🔄 Refresh this stock now (live)", use_container_width=True):
        res = refresh_one_stock(sel_sym, sel_ex, row["name"])
        st.success("Refreshed.")
        if res.get("warn"):
            st.warning(res["warn"])
        st.rerun()

    headline = safe_str(row.get("sample_headline"))
    url = row.get("sample_url")
    if headline:
        st.write("Latest headline sample:", headline)
    else:
        st.info("No headline stored yet (coverage may be low).")

    if is_valid_url(url):
        st.link_button("Open article", url)

    # Price chart: prefer DB, fallback to forced fetch
    px = get_prices_from_db(sel_sym, sel_ex)
    if px.empty:
        px = yf_history_force(sel_sym, sel_ex)

    if not px.empty and px["close"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name="Price"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data available from free source for this ticker.")

    # show reasons if present
    if row.get("reasons"):
        try:
            st.write("Why this score?")
            st.json(json.loads(row["reasons"]))
        except Exception:
            pass

else:
    st.subheader("Paper Trading (demo)")
    with db() as con:
        cash = float(con.execute("SELECT cash FROM paper_wallet WHERE id=1").fetchone()[0])
        trades = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY ts DESC LIMIT 200", con)
    st.metric("Demo cash", round(cash, 2))
    st.dataframe(trades, use_container_width=True, height=420)
