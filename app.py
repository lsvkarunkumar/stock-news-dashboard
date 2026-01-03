import json
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime, timedelta

from lib.db import init_db, db
from lib.universe import get_universe, fetch_universe, upsert_universe
from lib.prices import to_yahoo_ticker  # uses .NS / .BO
import yfinance as yf

st.set_page_config(page_title="News × Price Dashboard", layout="wide")

# -------------------- UI / STYLE --------------------
def load_css():
    try:
        with open("assets/style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

init_db()
load_css()

st.title("🧠 News × Price Watchlist (Auto-updated)")
st.caption("Educational dashboard. Final investment decision is always yours.")

with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Discover & Add", "Watchlist", "Paper Trading"], index=0)
    st.divider()
    st.caption("If you ran GitHub Actions and data doesn’t appear, use Streamlit → Reboot app.")

# -------------------- DB HELPERS --------------------
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
            SELECT w.symbol, w.exchange, u.name, u.sector, w.added_at
            FROM watchlist w
            LEFT JOIN universe u ON u.symbol=w.symbol AND u.exchange=w.exchange
            ORDER BY w.added_at DESC
        """, con)

def get_latest_signals():
    with db() as con:
        return pd.read_sql_query("""
            SELECT s.symbol, s.exchange, s.asof, s.score, s.reasons, u.name
            FROM signals s
            LEFT JOIN universe u ON u.symbol=s.symbol AND u.exchange=s.exchange
            WHERE s.asof = (SELECT MAX(asof) FROM signals)
        """, con)

def get_latest_mentions():
    with db() as con:
        return pd.read_sql_query("""
            SELECT n.symbol, n.exchange, n.date, n.mentions, n.sample_headline, n.sample_url, u.name
            FROM news_mentions n
            LEFT JOIN universe u ON u.symbol=n.symbol AND u.exchange=n.exchange
            WHERE n.date = (SELECT MAX(date) FROM news_mentions)
        """, con)

def get_prices_from_db(symbol, exchange):
    with db() as con:
        return pd.read_sql_query("""
            SELECT date, close, volume
            FROM prices
            WHERE symbol=? AND exchange=?
            ORDER BY date
        """, con, params=(symbol, exchange))

def ensure_universe_loaded():
    uni = get_universe()
    if len(uni) > 0:
        return uni, False

    # fallback live load (so Discover works even if DB hasn't synced)
    fresh = fetch_universe()
    upsert_universe(fresh)
    uni2 = get_universe()
    return uni2, True

# -------------------- DATA QUALITY HELPERS --------------------
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

# -------------------- PRICE METRICS --------------------
def _nearest_date_index(dates: pd.Series, target: pd.Timestamp):
    # return index of last date <= target
    d = pd.to_datetime(dates)
    mask = d <= target
    if not mask.any():
        return None
    return int(np.where(mask)[0][-1])

def compute_metrics_from_history(df: pd.DataFrame) -> dict:
    """
    df columns: date(str), close(float), volume(float)
    returns: live, 52w high/low, returns 1D/1W/1M/3M/6M/12M
    """
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
    if df is None or df.empty or df["close"].isna().all():
        return out

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2 = df2.dropna(subset=["close"]).sort_values("date")
    if df2.empty:
        return out

    last_close = float(df2["close"].iloc[-1])
    out["live"] = last_close

    # 52 week (approx 365 days)
    cutoff = df2["date"].iloc[-1] - pd.Timedelta(days=365)
    df_52 = df2[df2["date"] >= cutoff]
    if not df_52.empty:
        out["52w_high"] = float(df_52["close"].max())
        out["52w_low"] = float(df_52["close"].min())

    # returns: use closest prior trading day to target
    last_date = df2["date"].iloc[-1]
    def ret(days):
        idx = _nearest_date_index(df2["date"], last_date - pd.Timedelta(days=days))
        if idx is None:
            return np.nan
        base = float(df2["close"].iloc[idx])
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

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_history_yf(symbols_exchanges: list[tuple[str, str]], period_days: int = 370) -> dict:
    """
    Returns dict {(symbol, exchange): df(date, close, volume)}
    Uses yfinance batch download for speed.
    """
    tickers = [to_yahoo_ticker(s, ex) for s, ex in symbols_exchanges]
    # Batch download
    data = yf.download(
        tickers=tickers,
        period=f"{period_days}d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    out = {}
    # Single ticker returns columns like ('Close') etc. Multi returns MultiIndex.
    if isinstance(data.columns, pd.MultiIndex):
        for (sym, ex), t in zip(symbols_exchanges, tickers):
            try:
                sub = data[t].dropna(how="all")
                if sub.empty:
                    out[(sym, ex)] = pd.DataFrame(columns=["date", "close", "volume"])
                    continue
                sub = sub.reset_index()
                sub["date"] = pd.to_datetime(sub["Date"]).dt.date.astype(str)
                out[(sym, ex)] = pd.DataFrame({
                    "date": sub["date"],
                    "close": sub.get("Close"),
                    "volume": sub.get("Volume")
                })
            except Exception:
                out[(sym, ex)] = pd.DataFrame(columns=["date", "close", "volume"])
    else:
        # single ticker
        (sym, ex) = symbols_exchanges[0]
        sub = data.dropna(how="all")
        if sub.empty:
            out[(sym, ex)] = pd.DataFrame(columns=["date", "close", "volume"])
        else:
            sub = sub.reset_index()
            sub["date"] = pd.to_datetime(sub["Date"]).dt.date.astype(str)
            out[(sym, ex)] = pd.DataFrame({
                "date": sub["date"],
                "close": sub.get("Close"),
                "volume": sub.get("Volume")
            })
    return out

def enrich_with_metrics(df: pd.DataFrame, max_rows: int = 40) -> pd.DataFrame:
    """
    Adds live, 52w high/low, returns columns by fetching yfinance history for top N rows (performance safe).
    """
    if df.empty:
        return df

    take = df.head(max_rows).copy()
    pairs = [(r.symbol, r.exchange) for r in take.itertuples(index=False)]
    histories = fetch_history_yf(pairs)

    metrics_rows = []
    for r in take.itertuples(index=False):
        hist = histories.get((r.symbol, r.exchange), pd.DataFrame())
        m = compute_metrics_from_history(hist)
        metrics_rows.append(m)

    met = pd.DataFrame(metrics_rows)
    out = pd.concat([take.reset_index(drop=True), met], axis=1)

    # format returns
    return out

# -------------------- PAGES --------------------
if page == "Discover & Add":
    st.subheader("🔎 Discover stocks (NSE + BSE) and add with checkboxes")

    uni, did_refresh = ensure_universe_loaded()
    if did_refresh:
        st.success("Universe was empty in this runtime — reloaded NSE+BSE universe now.")

    # Sector filter (works when sector is populated; otherwise show Unknown)
    uni2 = uni.copy()
    uni2["sector"] = uni2["sector"].fillna("Unknown")
    all_sectors = sorted([s for s in uni2["sector"].unique() if isinstance(s, str)])

    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        q = st.text_input("Search (name or symbol)", placeholder="e.g., RELIANCE / TCS / VODAFONE / 500325")
    with c2:
        sectors = st.multiselect("Sector filter (multi-select)", options=all_sectors, default=[])
    with c3:
        st.info(f"Universe: **{len(uni2):,}**", icon="📦")

    show = uni2
    if sectors:
        show = show[show["sector"].isin(sectors)]
    if q:
        mask = (
            show["symbol"].astype(str).str.contains(q, case=False, na=False) |
            show["name"].astype(str).str.contains(q, case=False, na=False)
        )
        show = show[mask]

    # Keep responsive
    show = show.head(400).copy()

    # Add checkbox column
    show.insert(0, "add", False)

    # Enrich top rows with price metrics (live/52w/returns) for premium feel
    enriched = enrich_with_metrics(show.drop(columns=["add"]).rename(columns={"symbol":"symbol","exchange":"exchange"}), max_rows=40)
    # Merge back (only for top 40)
    if not enriched.empty:
        enriched_key = enriched[["exchange","symbol","live","52w_high","52w_low","ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"]]
        show = show.merge(enriched_key, on=["exchange","symbol"], how="left")

    st.caption("Tip: Tick ✅ Add and click **Add selected to Watchlist**. (Price stats are fetched for top rows only for speed.)")

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
            "add": st.column_config.CheckboxColumn("Add", help="Tick to add to watchlist"),
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

    selected = edited[edited["add"] == True][["symbol", "exchange"]].drop_duplicates()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"Selected: **{len(selected)}**")
    with col2:
        if st.button("➕ Add selected to Watchlist", use_container_width=True):
            add_to_watchlist_bulk(selected)
            st.success(f"Added {len(selected)} stock(s) to watchlist.")
            st.rerun()

    st.divider()
    st.write("✅ Next step: Run GitHub Actions → **Update Data** to fetch news + prices + scores for your watchlist.")

elif page == "Watchlist":
    st.subheader("📌 Your Watchlist (premium view + delete + story)")

    wl = get_watchlist_df()
    if wl.empty:
        st.info("Watchlist is empty. Go to **Discover & Add**, tick stocks, and add them.")
        st.stop()

    sig = get_latest_signals()
    ment = get_latest_mentions()

    df = wl.merge(sig[["symbol","exchange","score","reasons"]], on=["symbol","exchange"], how="left") \
           .merge(ment[["symbol","exchange","mentions","sample_headline","sample_url"]], on=["symbol","exchange"], how="left")

    df["score"] = df["score"].fillna(0).astype(int)
    df["mentions"] = df["mentions"].fillna(0).astype(int)
    df["sector"] = df["sector"].fillna("Unknown")

    # Sector filter in watchlist
    sectors = sorted([s for s in df["sector"].unique() if isinstance(s, str)])
    pick_sectors = st.multiselect("Filter watchlist by sector", options=sectors, default=[])
    if pick_sectors:
        df = df[df["sector"].isin(pick_sectors)].copy()

    # Price metrics for top watchlist rows (fast, cached)
    df_show = df.copy()
    df_show.insert(0, "delete", False)

    enriched = enrich_with_metrics(df_show.drop(columns=["delete"]).rename(columns={"symbol":"symbol","exchange":"exchange"}), max_rows=min(60, len(df_show)))
    if not enriched.empty:
        enriched_key = enriched[["exchange","symbol","live","52w_high","52w_low","ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"]]
        df_show = df_show.merge(enriched_key, on=["exchange","symbol"], how="left")

    # Sort by: score + mentions (proxy for “latest updates / news impact”)
    df_show = df_show.sort_values(["score","mentions"], ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Watchlist size", int(len(df_show)))
    c2.metric("Top score", int(df_show["score"].max()) if len(df_show) else 0)
    c3.metric("Total mentions (latest)", int(df_show["mentions"].sum()) if len(df_show) else 0)
    c4.caption("Sorted by score + mention intensity (latest).")

    edited = st.data_editor(
        df_show[[
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
            "delete": st.column_config.CheckboxColumn("Del", help="Tick rows to delete from watchlist"),
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
    if st.button("🗑️ Delete selected from Watchlist", use_container_width=True):
        remove_from_watchlist_bulk(to_del)
        st.success(f"Deleted {len(to_del)} stock(s) from watchlist.")
        st.rerun()

    st.divider()
    st.subheader("🧾 Story panel")

    # Story dropdown
    options = edited.apply(lambda r: f'{r["symbol"]} ({r["exchange"]}) — {r["name"]}', axis=1).tolist()
    pick = st.selectbox("Select stock to view story", options)

    sel_sym = pick.split(" ")[0]
    sel_ex = pick.split("(")[1].split(")")[0]

    row = df_show[(df_show.symbol == sel_sym) & (df_show.exchange == sel_ex)].iloc[0]

    st.markdown(f"### **{row['name']}** — {sel_sym} ({sel_ex})")

    headline = safe_str(row.get("sample_headline"))
    url = row.get("sample_url")

    if headline:
        st.write("Latest headline sample:", headline)
    else:
        st.info("No headline stored yet. Run GitHub Actions again after you add this stock (watchlist).")

    # ✅ BUG FIX: show link button only if URL is valid
    if is_valid_url(url):
        st.link_button("Open article", url)

    # Price chart from DB if exists, else cached yfinance
    px = get_prices_from_db(sel_sym, sel_ex)
    if px.empty:
        histories = fetch_history_yf([(sel_sym, sel_ex)])
        px = histories.get((sel_sym, sel_ex), pd.DataFrame())

    if px is not None and not px.empty and px["close"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name="Close"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data yet. Run GitHub Actions again (it fetches prices for watchlist stocks).")

    # Explain score (if available)
    if row.get("reasons"):
        try:
            reasons = json.loads(row["reasons"])
            st.write("**Why this score?**")
            st.json(reasons)
        except Exception:
            pass

elif page == "Paper Trading":
    st.subheader("🧪 Paper Trading (demo coins)")

    with db() as con:
        cash = float(con.execute("SELECT cash FROM paper_wallet WHERE id=1").fetchone()[0])
        trades = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY ts DESC LIMIT 200", con)

    st.metric("Demo cash", round(cash, 2))

    wl = get_watchlist_df()
    if wl.empty:
        st.info("Add stocks to watchlist first.")
        st.stop()

    pick = st.selectbox("Trade symbol", wl.apply(lambda r: f'{r["symbol"]} ({r["exchange"]}) — {r["name"]}', axis=1).tolist())
    sel_sym = pick.split(" ")[0]
    sel_ex = pick.split("(")[1].split(")")[0]

    # prefer DB price, else yfinance
    px = get_prices_from_db(sel_sym, sel_ex)
    last_price = float(px["close"].dropna().iloc[-1]) if not px.empty and px["close"].notna().any() else np.nan
    if np.isnan(last_price):
        histories = fetch_history_yf([(sel_sym, sel_ex)])
        px2 = histories.get((sel_sym, sel_ex), pd.DataFrame())
        if px2 is not None and not px2.empty and px2["close"].notna().any():
            last_price = float(px2["close"].dropna().iloc[-1])
        else:
            last_price = 0.0

    c1, c2, c3 = st.columns(3)
    side = c1.selectbox("Side", ["BUY", "SELL"])
    qty = c2.number_input("Qty", min_value=1.0, value=1.0, step=1.0)
    price = c3.number_input("Price", min_value=0.0, value=float(last_price), step=0.5)

    notes = st.text_input("Notes (optional)", placeholder="e.g., Score>=90 auto rule / Manual conviction / News spike")

    if st.button("✅ Place paper trade", use_container_width=True):
        ts = pd.Timestamp.utcnow().isoformat()
        cost = qty * price
        with db() as con:
            if side == "BUY" and cost > cash:
                st.error("Not enough demo cash.")
            else:
                con.execute(
                    "INSERT INTO paper_trades(ts,symbol,exchange,side,qty,price,notes) VALUES (?,?,?,?,?,?,?)",
                    (ts, sel_sym, sel_ex, side, float(qty), float(price), notes)
                )
                new_cash = cash - cost if side == "BUY" else cash + cost
                con.execute("UPDATE paper_wallet SET cash=? WHERE id=1", (new_cash,))
                con.commit()
                st.success(f"Paper trade placed: {side} {qty} @ {price}")
                st.rerun()

    st.write("Recent trades")
    st.dataframe(trades, use_container_width=True, height=320)
