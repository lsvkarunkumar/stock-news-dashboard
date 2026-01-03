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

# ---------- Init ----------
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
    debug_on = st.toggle("Debug mode", value=True)
    st.caption("Live price here = latest available via free feeds (may be delayed).")

# ---------- Helpers ----------
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
        return uni
    fresh = fetch_universe()
    upsert_universe(fresh)
    return get_universe()

def normalize_df(rows: pd.DataFrame) -> pd.DataFrame:
    clean = rows.copy()
    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()
    clean["exchange"] = clean["exchange"].astype(str).str.strip().str.upper()
    return clean.drop_duplicates(subset=["symbol","exchange"])

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

# ---------- Latest-per-stock (fix for “not updating”) ----------
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

def prices_from_db(symbol, exchange):
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

# ---------- Live price fetch (intraday first) ----------
def yf_history_force(symbol: str, exchange: str) -> pd.DataFrame:
    t = to_yahoo_ticker(symbol, exchange)

    # Try 1-minute intraday first
    try:
        df = yf.download(t, period="1d", interval="1m", progress=False, threads=True)
        if df is not None and not df.empty:
            df = df.reset_index()
            dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
            df["date"] = pd.to_datetime(df[dt_col]).dt.strftime("%Y-%m-%d %H:%M")
            df = df.rename(columns={"Close": "close", "Volume": "volume"})
            return df[["date","close","volume"]]
    except Exception:
        pass

    # Fallback daily
    df = yf.download(t, period="370d", interval="1d", progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","close","volume"])
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df = df.rename(columns={"Close":"close","Volume":"volume"})
    return df[["date","close","volume"]]

def compute_live(px: pd.DataFrame):
    if px is None or px.empty or px["close"].isna().all():
        return np.nan
    return float(px["close"].dropna().iloc[-1])

# ---------- Refresh pipeline (guaranteed DB writes or visible error) ----------
def refresh_one_stock(symbol: str, exchange: str, name: str):
    symbol = str(symbol).strip().upper()
    exchange = str(exchange).strip().upper()
    asof = date.today().isoformat()

    # Clear UI cache so you don’t see stale values after refresh
    try:
        st.cache_data.clear()
    except Exception:
        pass

    # 1) Prices
    px = yf_history_force(symbol, exchange)
    upsert_prices(symbol, exchange, px)

    # 2) News + score (safe)
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
    except Exception as e:
        # still write a signal row even if news fails
        score, reasons = compute_score(0, 0, 0, px)
        upsert_signal(symbol, exchange, asof, score, reasons)
        return {"ok": True, "warn": f"News failed: {type(e).__name__}"}

    return {"ok": True}

# ---------- DEBUG PANEL ----------
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

# ---------- Pages ----------
if page == "Discover & Add":
    st.subheader("Discover & Add")

    uni = ensure_universe_loaded().copy()
    uni["sector"] = uni.get("sector", "Unknown")
    if "sector" in uni.columns:
        uni["sector"] = uni["sector"].fillna("Unknown")
    else:
        uni["sector"] = "Unknown"

    uni["symbol"] = uni["symbol"].astype(str).str.strip().str.upper()
    uni["exchange"] = uni["exchange"].astype(str).str.strip().str.upper()

    # Typeahead-like pick (no Enter needed)
    q = st.text_input("Search", placeholder="Type to filter… (e.g., RELIANCE)")
    sectors_all = sorted([s for s in uni["sector"].astype(str).unique()])
    sectors = st.multiselect("Sector (multi-select)", options=sectors_all, default=[])

    show = uni
    if sectors:
        show = show[show["sector"].isin(sectors)]
    if q:
        m = show["symbol"].str.contains(q, case=False, na=False) | show["name"].astype(str).str.contains(q, case=False, na=False)
        show = show[m]
    show = show.head(200).copy()

    # bring score into universe view (latest-per-stock)
    sig = latest_signals_per_stock()
    show = show.merge(sig[["symbol","exchange","score"]], on=["symbol","exchange"], how="left")
    show["score"] = show["score"].fillna(0).astype(int)

    show.insert(0, "add", False)

    add_selected_btn = st.button("➕ Add selected to watchlist", use_container_width=True)

    edited = st.data_editor(
        show[["add","exchange","symbol","name","sector","score"]],
        use_container_width=True,
        height=540,
        hide_index=True,
        column_config={
            "add": st.column_config.CheckboxColumn("Add"),
            "score": st.column_config.NumberColumn("Score", format="%d"),
        }
    )
    selected = edited[edited["add"] == True][["symbol","exchange"]].drop_duplicates()
    st.info(f"Selected: **{len(selected)}**")

    if add_selected_btn:
        add_to_watchlist_bulk(selected)
        st.success("Added to watchlist.")
        st.rerun()

elif page == "Watchlist":
    st.subheader("Watchlist")

    wl = get_watchlist_df()
    if wl.empty:
        st.info("Watchlist empty. Add from Discover.")
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

    df = df.sort_values(["score","mentions"], ascending=False).copy()
    df.insert(0, "delete", False)

    c1, c2, c3 = st.columns([2,2,6])
    with c1:
        del_selected_btn = st.button("🗑️ Delete selected", use_container_width=True)
    with c2:
        del_all_btn = st.button("🧹 Delete ALL filtered", use_container_width=True)
    with c3:
        st.caption("Refresh any stock from Story panel to force-update score/news/prices.")

    edited = st.data_editor(
        df[["delete","exchange","symbol","name","sector","score","mentions","sample_headline"]],
        use_container_width=True,
        height=520,
        hide_index=True,
        column_config={
            "delete": st.column_config.CheckboxColumn("Del"),
            "score": st.column_config.NumberColumn("Score", format="%d"),
            "mentions": st.column_config.NumberColumn("Mentions", format="%d"),
        }
    )

    to_del = edited[edited["delete"] == True][["symbol","exchange"]].drop_duplicates()
    if del_all_btn:
        remove_from_watchlist_bulk(view[["symbol","exchange"]])
        st.success("Deleted ALL filtered items.")
        st.rerun()
    if del_selected_btn:
        remove_from_watchlist_bulk(to_del)
        st.success(f"Deleted {len(to_del)} selected item(s).")
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
        if res and res.get("warn"):
            st.warning(res["warn"])
        st.rerun()

    # Show latest rows directly from DB (proof of insert)
    with db() as con:
        last_sig = pd.read_sql_query("""
            SELECT * FROM signals WHERE symbol=? AND exchange=? ORDER BY asof DESC LIMIT 1
        """, con, params=(sel_sym, sel_ex))
        last_news = pd.read_sql_query("""
            SELECT * FROM news_mentions WHERE symbol=? AND exchange=? ORDER BY date DESC LIMIT 1
        """, con, params=(sel_sym, sel_ex))

    st.write("Latest signal row:")
    st.dataframe(last_sig, use_container_width=True)

    st.write("Latest news row:")
    st.dataframe(last_news, use_container_width=True)

    headline = safe_str(row.get("sample_headline"))
    url = row.get("sample_url")
    if headline:
        st.write("Latest headline sample:", headline)
    if is_valid_url(url):
        st.link_button("Open article", url)

    px = prices_from_db(sel_sym, sel_ex)
    if px.empty:
        px = yf_history_force(sel_sym, sel_ex)

    if not px.empty and px["close"].notna().any():
        live = compute_live(px)
        st.metric("Live (latest available)", f"{live:.2f}" if not np.isnan(live) else "NA")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name="Price"))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data available from free source for this ticker.")

else:
    st.subheader("Paper Trading (demo)")
    with db() as con:
        cash = float(con.execute("SELECT cash FROM paper_wallet WHERE id=1").fetchone()[0])
        trades = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY ts DESC LIMIT 200", con)
    st.metric("Demo cash", round(cash, 2))
    st.dataframe(trades, use_container_width=True, height=420)
