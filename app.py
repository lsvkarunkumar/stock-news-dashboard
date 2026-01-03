import json
import math
import datetime as dt
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from lib.db import init_db, db
from lib.universe import get_universe, fetch_universe, upsert_universe
from lib.prices import to_yahoo_ticker
from lib.news import build_company_query, gdelt_mentions, gdelt_latest_headline
from lib.scoring import compute_score

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

st.title("🧠 News × Price Dashboard")
st.caption("Auto-updated watchlist with news intensity & price context (educational). Final decision is yours.")

with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Discover & Add", "Watchlist", "Paper Trading"], index=0)
    st.divider()
    st.caption("Live price here = latest available from free feeds (may be delayed).")

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

# ---------------- Watchlist DB ops (A: normalize + avoid dup) ----------------
def add_to_watchlist_bulk(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return

    clean = rows.copy()
    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()
    clean["exchange"] = clean["exchange"].astype(str).str.strip().str.upper()
    clean = clean.drop_duplicates(subset=["symbol", "exchange"])

    with db() as con:
        con.executemany(
            "INSERT OR IGNORE INTO watchlist(symbol, exchange, added_at) VALUES (?,?,?)",
            [(r.symbol, r.exchange, date.today().isoformat()) for r in clean.itertuples(index=False)]
        )
        con.commit()

def remove_from_watchlist_bulk(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return
    clean = rows.copy()
    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()
    clean["exchange"] = clean["exchange"].astype(str).str.strip().str.upper()
    clean = clean.drop_duplicates(subset=["symbol", "exchange"])

    with db() as con:
        con.executemany(
            "DELETE FROM watchlist WHERE symbol=? AND exchange=?",
            [(r.symbol, r.exchange) for r in clean.itertuples(index=False)]
        )
        con.commit()

def get_watchlist_df():
    with db() as con:
        return pd.read_sql_query("""
            SELECT w.symbol, w.exchange, u.name, COALESCE(u.sector,'Unknown') AS sector, u.isin, w.added_at
            FROM watchlist w
            LEFT JOIN universe u ON u.symbol=w.symbol AND u.exchange=w.exchange
            ORDER BY w.added_at DESC
        """, con)

# ---------------- Market data ----------------
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

# E: improved "live" using intraday first
def yf_history_force(symbol: str, exchange: str) -> pd.DataFrame:
    t = to_yahoo_ticker(symbol, exchange)

    # Try intraday first (near-live last candle)
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

    # Fallback daily
    df = yf.download(t, period="370d", interval="1d", progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    return df[["date", "close", "volume"]]

@st.cache_data(ttl=5 * 60, show_spinner=False)
def yf_history_cached(symbol: str, exchange: str) -> pd.DataFrame:
    # cached daily for table speed
    t = to_yahoo_ticker(symbol, exchange)
    df = yf.download(t, period="370d", interval="1d", progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    return df[["date", "close", "volume"]]

def compute_metrics_from_history(df: pd.DataFrame) -> dict:
    out = {
        "live": np.nan, "52w_high": np.nan, "52w_low": np.nan,
        "ret_1d": np.nan, "ret_1w": np.nan, "ret_1m": np.nan,
        "ret_3m": np.nan, "ret_6m": np.nan, "ret_12m": np.nan,
    }
    if df is None or df.empty or df["close"].isna().all():
        return out

    d = df.copy()
    # intraday has time strings; daily has date strings
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

    out["ret_1d"] = ret(1)
    out["ret_1w"] = ret(7)
    out["ret_1m"] = ret(30)
    out["ret_3m"] = ret(90)
    out["ret_6m"] = ret(180)
    out["ret_12m"] = ret(365)
    return out

# ---------------- News + score ----------------
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

def refresh_one_stock(symbol: str, exchange: str, name: str):
    """Immediate refresh: prices + news + score; never crash if news fails."""
    symbol = str(symbol).strip().upper()
    exchange = str(exchange).strip().upper()
    asof = date.today().isoformat()

    # Prices (force near-live)
    px = yf_history_force(symbol, exchange)
    upsert_prices(symbol, exchange, px)

    # News (safe)
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
    except Exception:
        score, reasons = compute_score(0, 0, 0, px)
        upsert_signal(symbol, exchange, asof, score, reasons)

def refresh_bulk(rows: pd.DataFrame, uni: pd.DataFrame, limit: int = 15):
    """Refresh a small batch so Streamlit doesn't get slow."""
    if rows is None or rows.empty:
        return
    sel = rows.head(limit).copy()
    for r in sel.itertuples(index=False):
        row = uni[(uni.symbol == r.symbol) & (uni.exchange == r.exchange)]
        nm = row.iloc[0]["name"] if not row.empty else r.symbol
        try:
            refresh_one_stock(r.symbol, r.exchange, nm)
        except Exception:
            continue

# ---------------- PAGES ----------------
if page == "Discover & Add":
    st.subheader("Discover & Add")

    uni, _ = ensure_universe_loaded()
    uni = uni.copy()
    uni["sector"] = uni["sector"].fillna("Unknown")
    uni["symbol"] = uni["symbol"].astype(str).str.strip().str.upper()
    uni["exchange"] = uni["exchange"].astype(str).str.strip().str.upper()

    c1, c2, c3 = st.columns([3, 4, 2])
    with c1:
        q = st.text_input("Search", placeholder="RELIANCE / TCS / VODAFONE / 500325")
    with c2:
        sectors_all = sorted([s for s in uni["sector"].unique() if isinstance(s, str)])
        sectors = st.multiselect("Sector (multi-select)", options=sectors_all, default=[])
    with c3:
        st.metric("Universe", f"{len(uni):,}")

    show = uni
    if sectors:
        show = show[show["sector"].isin(sectors)]
    if q:
        mask = (
            show["symbol"].astype(str).str.contains(q, case=False, na=False) |
            show["name"].astype(str).str.contains(q, case=False, na=False)
        )
        show = show[mask]

    # Keep fast
    show = show.head(250).copy()
    show.insert(0, "add", False)

    # (B) ONLY "Add selected" - NO add-250 button
    b1, b2 = st.columns([2, 6])
    with b1:
        add_selected_btn = st.button("➕ Add selected", use_container_width=True)
    with b2:
        st.caption("Tick ✅ Add, then click. (Watchlist will refresh automatically for a small batch.)")

    # Enrich top rows with cached daily metrics (for display)
    enrich_rows = show.head(40).copy()
    metrics = []
    for r in enrich_rows.itertuples(index=False):
        h = yf_history_cached(r.symbol, r.exchange)
        metrics.append(compute_metrics_from_history(h))
    met_df = pd.DataFrame(metrics)
    if not met_df.empty:
        enrich_rows = pd.concat([enrich_rows.reset_index(drop=True), met_df], axis=1)
        show = show.merge(
            enrich_rows[["exchange","symbol","live","52w_high","52w_low","ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"]],
            on=["exchange","symbol"], how="left"
        )

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
    st.info(f"Selected: **{len(selected)}**")

    if add_selected_btn:
        add_to_watchlist_bulk(selected)
        with st.spinner("Refreshing prices/news/score for a few selected stocks..."):
            refresh_bulk(selected, uni, limit=10)
        st.success("Added. (Remaining updates happen via scheduled refreshes / manual refresh buttons.)")
        st.rerun()

elif page == "Watchlist":
    st.subheader("Watchlist")

    wl = get_watchlist_df()
    if wl.empty:
        st.info("Watchlist empty. Add from Discover.")
        st.stop()

    # Sector filter
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

    # Cached daily metrics for table
    metrics = []
    for r in df.itertuples(index=False):
        h = yf_history_cached(r.symbol, r.exchange)
        metrics.append(compute_metrics_from_history(h))
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)

    # Sort by score + mentions
    df = df.sort_values(["score","mentions"], ascending=False).copy()

    # -------- (C) Auto-Update Watchlist rules --------
    st.markdown("### ⚙️ Auto-Update Watchlist (rules)")
    c1, c2, c3 = st.columns([2,2,2])
    with c1:
        min_score = st.slider("Min score", 0, 100, 80, 1)
    with c2:
        add_top_n = st.number_input("Add top N (total)", min_value=1, value=20, step=1)
    with c3:
        prefer_exchange = st.selectbox("If both NSE & BSE exist", ["Prefer NSE", "Prefer BSE"], index=0)

    run_auto = st.button("✨ Auto-Add candidates now", use_container_width=True)

    if run_auto:
        with db() as con:
            uni_df = pd.read_sql_query("""
                SELECT symbol, exchange, name, COALESCE(sector,'Unknown') as sector, COALESCE(isin,'') as isin
                FROM universe
            """, con)
            wl_cur = pd.read_sql_query("SELECT symbol, exchange FROM watchlist", con)

        cand = sig.merge(uni_df, on=["symbol","exchange"], how="left").merge(
            ment[["symbol","exchange","mentions"]], on=["symbol","exchange"], how="left"
        )
        cand["mentions"] = cand["mentions"].fillna(0).astype(int)
        cand = cand[cand["score"] >= int(min_score)].copy()

        # Apply current sector filter
        if pick_sectors:
            cand = cand[cand["sector"].isin(pick_sectors)]

        # Remove already in watchlist
        cand = cand.merge(wl_cur, on=["symbol","exchange"], how="left", indicator=True)
        cand = cand[cand["_merge"] == "left_only"].drop(columns=["_merge"])

        # Dedupe by ISIN (prevents NSE+BSE duplicates if you want only one)
        cand["isin"] = cand["isin"].fillna("")
        if prefer_exchange == "Prefer NSE":
            cand["ex_rank"] = (cand["exchange"] != "NSE").astype(int)
        else:
            cand["ex_rank"] = (cand["exchange"] != "BSE").astype(int)

        cand = cand.sort_values(["isin","ex_rank","score","mentions"], ascending=[True,True,False,False])
        cand = cand.drop_duplicates(subset=["isin"], keep="first")
        cand = cand.drop(columns=["ex_rank"], errors="ignore")

        # Take top N
        cand = cand.sort_values(["score","mentions"], ascending=False).head(int(add_top_n))
        to_add = cand[["symbol","exchange"]]

        add_to_watchlist_bulk(to_add)
        st.success(f"Auto-added {len(to_add)} stock(s).")

        # Refresh a small batch now so you see values immediately
        with st.spinner("Refreshing a few newly added stocks..."):
            refresh_bulk(to_add, uni_df.rename(columns={"symbol":"symbol","exchange":"exchange"}), limit=10)

        st.rerun()

    st.divider()

    # -------- (D) Delete controls --------
    df.insert(0, "delete", False)

    a1, a2, a3 = st.columns([2,2,6])
    with a1:
        del_selected_btn = st.button("🗑️ Delete selected", use_container_width=True)
    with a2:
        del_all_btn = st.button("🧹 Delete ALL filtered", use_container_width=True)
    with a3:
        st.caption("Delete ALL filtered clears the sector-filtered list in one click.")

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

    # Force refresh now (E: intraday first)
    if st.button("🔄 Refresh this stock now (live)", use_container_width=True):
        refresh_one_stock(sel_sym, sel_ex, row["name"])
        st.success("Refreshed.")
        st.rerun()

    headline = safe_str(row.get("sample_headline"))
    url = row.get("sample_url")

    if headline:
        st.write("Latest headline sample:", headline)
    else:
        st.info("No headline stored yet (coverage may be low).")

    if is_valid_url(url):
        st.link_button("Open article", url)

    px = get_prices_from_db(sel_sym, sel_ex)
    if px.empty:
        px = yf_history_force(sel_sym, sel_ex)

    if not px.empty and px["close"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name="Price"))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data available from free source for this ticker.")

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
