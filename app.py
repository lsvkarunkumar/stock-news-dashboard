import json
import math
import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from lib.db import init_db, db
from lib.universe import get_universe, fetch_universe, upsert_universe
from lib.prices import to_yahoo_ticker
from lib.news import build_company_query, gdelt_mentions, gdelt_latest_headline
from lib.scoring import compute_score

import yfinance as yf

st.set_page_config(page_title="News × Price Dashboard", layout="wide")

# ---------------- UI ----------------
def load_css():
    try:
        with open("assets/style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

init_db()
load_css()

st.title("🧠 News × Price Watchlist (Auto-updated)")
st.caption("Educational dashboard. Final decision is yours.")

with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Discover & Add", "Watchlist", "Paper Trading"], index=0)
    st.divider()
    st.caption("GitHub Actions still refreshes in background every 6 hours. App also refreshes instantly when you add/view stocks.")

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

def add_to_watchlist_bulk(rows: pd.DataFrame):
    if rows.empty:
        return
    with db() as con:
        con.executemany(
            "INSERT OR IGNORE INTO watchlist(symbol, exchange, added_at) VALUES (?,?,?)",
            [(r.symbol, r.exchange, date.today().isoformat()) for r in rows.itertuples(index=False)]
        )
        con.commit()

def remove_from_watchlist_bulk(rows: pd.DataFrame):
    if rows.empty:
        return
    with db() as con:
        con.executemany(
            "DELETE FROM watchlist WHERE symbol=? AND exchange=?",
            [(r.symbol, r.exchange) for r in rows.itertuples(index=False)]
        )
        con.commit()

def get_watchlist_df():
    with db() as con:
        return pd.read_sql_query("""
            SELECT w.symbol, w.exchange, u.name, COALESCE(u.sector,'Unknown') AS sector, w.added_at
            FROM watchlist w
            LEFT JOIN universe u ON u.symbol=w.symbol AND u.exchange=w.exchange
            ORDER BY w.added_at DESC
        """, con)

def get_prices_from_db(symbol, exchange):
    with db() as con:
        return pd.read_sql_query("""
            SELECT date, close, volume
            FROM prices
            WHERE symbol=? AND exchange=?
            ORDER BY date
        """, con, params=(symbol, exchange))

def upsert_prices(symbol: str, exchange: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    rows = [(symbol, exchange, r.date,
             float(r.close) if pd.notna(r.close) else None,
             float(r.volume) if pd.notna(r.volume) else None)
            for r in df.itertuples(index=False)]
    with db() as con:
        con.executemany("""
            INSERT OR REPLACE INTO prices(symbol, exchange, date, close, volume)
            VALUES (?,?,?,?,?)
        """, rows)
        con.commit()

def upsert_news(symbol: str, exchange: str, asof: str, mentions: int, headline, url):
    with db() as con:
        con.execute("""
            INSERT INTO news_mentions(symbol, exchange, date, mentions, sample_headline, sample_url)
            VALUES (?,?,?,?,?,?)
        """, (symbol, exchange, asof, int(mentions), headline, url))
        con.commit()

def upsert_signal(symbol: str, exchange: str, asof: str, score: int, reasons: str):
    with db() as con:
        con.execute("""
            INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons)
            VALUES (?,?,?,?,?)
        """, (symbol, exchange, asof, score, reasons))
        con.commit()

def get_latest_signals_mentions():
    with db() as con:
        sig = pd.read_sql_query("""
            SELECT s.symbol, s.exchange, s.asof, s.score, s.reasons
            FROM signals s
            WHERE s.asof = (SELECT MAX(asof) FROM signals)
        """, con)
        ment = pd.read_sql_query("""
            SELECT n.symbol, n.exchange, n.date, n.mentions, n.sample_headline, n.sample_url
            FROM news_mentions n
            WHERE n.date = (SELECT MAX(date) FROM news_mentions)
        """, con)
    return sig, ment

# ---------------- Market stats ----------------
def compute_metrics_from_history(df: pd.DataFrame) -> dict:
    out = {
        "live": np.nan, "52w_high": np.nan, "52w_low": np.nan,
        "ret_1d": np.nan, "ret_1w": np.nan, "ret_1m": np.nan,
        "ret_3m": np.nan, "ret_6m": np.nan, "ret_12m": np.nan,
    }
    if df is None or df.empty or df["close"].isna().all():
        return out

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["close"]).sort_values("date")
    if d.empty:
        return out

    last_close = float(d["close"].iloc[-1])
    out["live"] = last_close

    cutoff = d["date"].iloc[-1] - pd.Timedelta(days=365)
    d52 = d[d["date"] >= cutoff]
    if not d52.empty:
        out["52w_high"] = float(d52["close"].max())
        out["52w_low"] = float(d52["close"].min())

    last_date = d["date"].iloc[-1]

    def ret(days):
        target = last_date - pd.Timedelta(days=days)
        base_rows = d[d["date"] <= target]
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

@st.cache_data(ttl=60*60, show_spinner=False)
def yf_history(symbol: str, exchange: str, period_days: int = 370) -> pd.DataFrame:
    t = to_yahoo_ticker(symbol, exchange)
    df = yf.download(t, period=f"{period_days}d", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","close","volume"])
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df = df.rename(columns={"Close":"close", "Volume":"volume"})
    return df[["date","close","volume"]]

def refresh_one_stock(symbol: str, exchange: str, name: str):
    """Immediate refresh: prices + news + signal (no GitHub Actions needed)."""
    asof = date.today().isoformat()

    # Prices
    px = yf_history(symbol, exchange)
    upsert_prices(symbol, exchange, px)

    # News
    d5 = date.today() - dt.timedelta(days=5)
    d30 = date.today() - dt.timedelta(days=30)
    d60 = date.today() - dt.timedelta(days=60)

    q = build_company_query(symbol, name)
    m5 = gdelt_mentions(q, d5, date.today())
    m30 = gdelt_mentions(q, d30, date.today())
    m60 = gdelt_mentions(q, d60, date.today())
    headline, url = gdelt_latest_headline(q, d60, date.today())
    upsert_news(symbol, exchange, asof, m60, headline, url)

    # Signal
    score, reasons = compute_score(m5, m30, m60, px)
    upsert_signal(symbol, exchange, asof, score, reasons)

def refresh_bulk(selected: pd.DataFrame, uni: pd.DataFrame):
    """Refresh multiple stocks quickly (limited to avoid heavy load)."""
    if selected.empty:
        return
    # safety: refresh max 25 at a time
    sel = selected.head(25)
    for r in sel.itertuples(index=False):
        row = uni[(uni.symbol == r.symbol) & (uni.exchange == r.exchange)]
        name = row.iloc[0]["name"] if not row.empty else r.symbol
        try:
            refresh_one_stock(r.symbol, r.exchange, name)
        except Exception:
            # keep going even if one fails
            continue

# ---------------- PAGES ----------------
if page == "Discover & Add":
    st.subheader("🔎 Discover (checkbox add + live stats)")

    uni, _ = ensure_universe_loaded()
    uni = uni.copy()
    uni["sector"] = uni["sector"].fillna("Unknown")

    # Top action bar (no scrolling to add button)
    top = st.container()
    with top:
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        with c1:
            q = st.text_input("Search", placeholder="RELIANCE / TCS / VODAFONE / 500325")
        with c2:
            sectors_all = sorted([s for s in uni["sector"].unique() if isinstance(s, str)])
            sectors = st.multiselect("Sector (multi)", options=sectors_all, default=[])
        with c3:
            st.metric("Universe", f"{len(uni):,}")
        with c4:
            st.write("")  # spacer

    show = uni
    if sectors:
        show = show[show["sector"].isin(sectors)]
    if q:
        mask = (
            show["symbol"].astype(str).str.contains(q, case=False, na=False) |
            show["name"].astype(str).str.contains(q, case=False, na=False)
        )
        show = show[mask]

    show = show.head(250).copy()
    show.insert(0, "add", False)

    # enrich top 40 for premium feel
    enrich_rows = show.head(40).copy()
    metrics = []
    for r in enrich_rows.itertuples(index=False):
        h = yf_history(r.symbol, r.exchange)
        m = compute_metrics_from_history(h)
        metrics.append(m)
    met_df = pd.DataFrame(metrics)
    if not met_df.empty:
        enrich_rows = pd.concat([enrich_rows.reset_index(drop=True), met_df], axis=1)
        show = show.merge(enrich_rows[["exchange","symbol","live","52w_high","52w_low","ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"]],
                          on=["exchange","symbol"], how="left")

    # Add button ABOVE table
    selected_placeholder = st.empty()
    add_btn_col1, add_btn_col2 = st.columns([3, 2])
    with add_btn_col1:
        st.caption("Tick ✅ Add → then click the button (button stays here).")
    with add_btn_col2:
        add_now = st.button("➕ Add selected to Watchlist (and refresh)", use_container_width=True)

    edited = st.data_editor(
        show[[
            "add","exchange","symbol","name","sector",
            "live","52w_high","52w_low",
            "ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"
        ]],
        use_container_width=True,
        height=560,
        hide_index=True,
        column_config={
            "add": st.column_config.CheckboxColumn("Add"),
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
    selected_placeholder.info(f"Selected: **{len(selected)}**")

    if add_now:
        add_to_watchlist_bulk(selected)
        with st.spinner("Refreshing news + prices + score for selected stocks..."):
            refresh_bulk(selected, uni)
        st.success(f"Added & refreshed {min(len(selected),25)} stock(s). (If you selected more than 25, add next batch.)")
        st.rerun()

elif page == "Watchlist":
    st.subheader("📌 Watchlist (live stats + delete + story)")

    wl = get_watchlist_df()
    if wl.empty:
        st.info("Watchlist empty. Add from Discover.")
        st.stop()

    # filters
    sectors = sorted([s for s in wl["sector"].unique() if isinstance(s, str)])
    pick_sectors = st.multiselect("Filter by sector", options=sectors, default=[])
    view = wl.copy()
    if pick_sectors:
        view = view[view["sector"].isin(pick_sectors)].copy()

    sig, ment = get_latest_signals_mentions()
    df = view.merge(sig[["symbol","exchange","score","reasons"]], on=["symbol","exchange"], how="left") \
             .merge(ment[["symbol","exchange","mentions","sample_headline","sample_url"]], on=["symbol","exchange"], how="left")

    df["score"] = df["score"].fillna(0).astype(int)
    df["mentions"] = df["mentions"].fillna(0).astype(int)

    # live metrics (cached)
    metrics = []
    for r in df.itertuples(index=False):
        h = yf_history(r.symbol, r.exchange)
        metrics.append(compute_metrics_from_history(h))
    met_df = pd.DataFrame(metrics)
    df = pd.concat([df.reset_index(drop=True), met_df], axis=1)

    # sort by latest signal proxy
    df = df.sort_values(["score","mentions"], ascending=False).copy()
    df.insert(0, "delete", False)

    # top actions
    del_col1, del_col2 = st.columns([3, 2])
    with del_col1:
        st.caption("Tick Del ✅ and click delete (button stays here).")
    with del_col2:
        delete_now = st.button("🗑️ Delete selected", use_container_width=True)

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
    st.info(f"Selected for delete: **{len(to_del)}**")

    if delete_now:
        remove_from_watchlist_bulk(to_del)
        st.success(f"Deleted {len(to_del)} stock(s).")
        st.rerun()

    st.divider()
    st.subheader("🧾 Story panel")

    options = edited.apply(lambda r: f'{r["symbol"]} ({r["exchange"]}) — {r["name"]}', axis=1).tolist()
    pick = st.selectbox("Select stock to view story", options)

    sel_sym = pick.split(" ")[0]
    sel_ex = pick.split("(")[1].split(")")[0]

    row = df[(df.symbol == sel_sym) & (df.exchange == sel_ex)].iloc[0]
    st.markdown(f"### **{row['name']}** — {sel_sym} ({sel_ex})")

    # If missing headline/price in DB, refresh immediately
    headline = safe_str(row.get("sample_headline"))
    url = row.get("sample_url")

    if not headline:
        with st.spinner("Fetching latest story & updating…"):
            refresh_one_stock(sel_sym, sel_ex, row["name"])
        # reload latest
        sig2, ment2 = get_latest_signals_mentions()
        mrow = ment2[(ment2.symbol == sel_sym) & (ment2.exchange == sel_ex)]
        if not mrow.empty:
            headline = safe_str(mrow.iloc[0].get("sample_headline"))
            url = mrow.iloc[0].get("sample_url")

    if headline:
        st.write("Latest headline sample:", headline)
    else:
        st.info("No headline found yet (may be low coverage).")

    if is_valid_url(url):
        st.link_button("Open article", url)

    px = get_prices_from_db(sel_sym, sel_ex)
    if px.empty:
        px = yf_history(sel_sym, sel_ex)

    if not px.empty and px["close"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name="Close"))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data available.")

    if row.get("reasons"):
        try:
            st.write("**Why this score?**")
            st.json(json.loads(row["reasons"]))
        except Exception:
            pass

else:
    st.subheader("🧪 Paper Trading (demo coins)")
    with db() as con:
        cash = float(con.execute("SELECT cash FROM paper_wallet WHERE id=1").fetchone()[0])
        trades = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY ts DESC LIMIT 200", con)
    st.metric("Demo cash", round(cash, 2))
    st.dataframe(trades, use_container_width=True, height=420)
