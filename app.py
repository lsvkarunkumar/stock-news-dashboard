import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from lib.db import init_db, db
from lib.universe import get_universe, fetch_universe, upsert_universe

st.set_page_config(page_title="News × Price Dashboard", layout="wide")

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

# Sidebar
with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Discover & Add", "Watchlist", "Paper Trading"], index=0)
    st.divider()
    st.caption("Tip: If you just ran GitHub Actions, use Streamlit 'Reboot app' if data doesn't appear.")

# Helpers
def add_to_watchlist_bulk(rows: pd.DataFrame):
    if rows.empty:
        return
    with db() as con:
        con.executemany(
            "INSERT OR IGNORE INTO watchlist(symbol, exchange, added_at) VALUES (?,?,?)",
            [(r.symbol, r.exchange, date.today().isoformat()) for r in rows.itertuples(index=False)]
        )
        con.commit()

def remove_from_watchlist(symbol: str, exchange: str):
    with db() as con:
        con.execute("DELETE FROM watchlist WHERE symbol=? AND exchange=?", (symbol, exchange))
        con.commit()

def get_watchlist_df():
    with db() as con:
        return pd.read_sql_query("""
            SELECT w.symbol, w.exchange, u.name, w.added_at
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

def get_prices(symbol, exchange):
    with db() as con:
        return pd.read_sql_query("""
            SELECT date, close, volume
            FROM prices
            WHERE symbol=? AND exchange=?
            ORDER BY date
        """, con, params=(symbol, exchange))

def ensure_universe_loaded():
    """If universe table is empty in this Streamlit runtime, load it once."""
    uni = get_universe()
    if len(uni) > 0:
        return uni, False

    # Fallback: fetch and populate (no keys needed)
    try:
        fresh = fetch_universe()
        upsert_universe(fresh)
        uni2 = get_universe()
        return uni2, True
    except Exception as e:
        return uni, False

# ---------------- PAGES ----------------

if page == "Discover & Add":
    st.subheader("🔎 Discover stocks (NSE + BSE) and add with checkboxes")

    uni, did_refresh = ensure_universe_loaded()
    if did_refresh:
        st.success("Universe was empty in this runtime — reloaded NSE+BSE universe now.")

    st.info(f"Universe loaded: **{len(uni):,}** rows")
    if uni.empty:
        st.error("Universe is still empty. This means Streamlit runtime didn't get the DB update AND live fetch failed.")
        st.stop()

    # Search
    q = st.text_input("Search (name or symbol)", placeholder="e.g., RELIANCE / TCS / VODAFONE / 500325")
    if q:
        mask = (
            uni["symbol"].astype(str).str.contains(q, case=False, na=False) |
            uni["name"].astype(str).str.contains(q, case=False, na=False)
        )
        show = uni[mask].head(500).copy()
    else:
        show = uni.head(300).copy()  # fast preview

    # Checkbox UI
    show.insert(0, "add", False)
    edited = st.data_editor(
        show[["add", "exchange", "symbol", "name", "isin"]],
        use_container_width=True,
        height=520,
        hide_index=True,
        column_config={
            "add": st.column_config.CheckboxColumn("Add", help="Tick to add to watchlist"),
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

    st.caption("Note: Scores/News/Prices will appear after GitHub Actions 'Update Data' runs again (since it processes watchlist).")

elif page == "Watchlist":
    st.subheader("📌 Your Watchlist (ranked by signals)")

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
    df = df.sort_values(["score","mentions"], ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Watchlist size", int(len(df)))
    c2.metric("Top score", int(df["score"].max()))
    c3.metric("Total mentions (latest)", int(df["mentions"].sum()))

    st.dataframe(
        df[["exchange","symbol","name","score","mentions","sample_headline"]],
        use_container_width=True,
        height=420
    )

    pick = st.selectbox(
        "Select stock to view story",
        df.apply(lambda r: f'{r["symbol"]} ({r["exchange"]}) — {r["name"]}', axis=1).tolist()
    )
    sel_sym = pick.split(" ")[0]
    sel_ex = pick.split("(")[1].split(")")[0]

    row = df[(df.symbol == sel_sym) & (df.exchange == sel_ex)].iloc[0]
    st.markdown(f"### 🧾 Story: **{row['name']}** — {sel_sym} ({sel_ex})")

    if row.get("sample_url"):
        st.write("Latest headline sample:", row.get("sample_headline",""))
        st.link_button("Open article", row["sample_url"])
    else:
        st.info("No headline stored yet. Run GitHub Actions again after adding stocks.")

    px = get_prices(sel_sym, sel_ex)
    if not px.empty and px["close"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=px["date"], y=px["close"], mode="lines", name="Close"))
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data yet. Run GitHub Actions again (it fetches prices for watchlist stocks).")

    if row.get("reasons"):
        try:
            reasons = json.loads(row["reasons"])
            st.write("**Why this score?**")
            st.json(reasons)
        except Exception:
            pass

    if st.button("🗑️ Remove from watchlist", use_container_width=True):
        remove_from_watchlist(sel_sym, sel_ex)
        st.rerun()

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

    px = get_prices(sel_sym, sel_ex)
    last_price = float(px["close"].dropna().iloc[-1]) if not px.empty and px["close"].notna().any() else 0.0

    c1, c2, c3 = st.columns(3)
    side = c1.selectbox("Side", ["BUY","SELL"])
    qty = c2.number_input("Qty", min_value=1.0, value=1.0, step=1.0)
    price = c3.number_input("Price", min_value=0.0, value=last_price, step=0.5)

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

    st.write("Recent trades")
    st.dataframe(trades, use_container_width=True, height=320)
