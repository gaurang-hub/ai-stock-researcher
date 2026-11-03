import streamlit as st
import json
import os
import datetime
import time
import re
import logging
import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import yfinance as yf
import pandas as pd
import numpy as np
import random
from openai import OpenAI
import sqlitecloud

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- APP VERSION CONFIG ---
APP_VERSION = "v27.0.0-dynamic-search-chart"
LAST_UPDATED = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

st.set_page_config(page_title="AI Stock Researcher & Decision-Support Platform", layout="wide")
st.title("📈 AI Stock Research & Decision-Support Quantitative Platform")

# ==============================================================================
# SECTION 1: NATIVE QUANTITATIVE & ADVANCED TECHNICAL ENGINE
# ==============================================================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_advanced_technicals(df):
    if df.empty or len(df) < 50:
        return {}
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 200 else ema_50
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean().iloc[-1]
    
    curr_price = close.iloc[-1]
    
    return {
        "ema_20": round(float(ema_20), 2),
        "ema_50": round(float(ema_50), 2),
        "ema_200": round(float(ema_200), 2),
        "atr_14": round(float(atr_14), 2),
        "trend_alignment": "Bullish" if curr_price > ema_50 > ema_200 else ("Bearish" if curr_price < ema_50 < ema_200 else "Neutral")
    }

# ==============================================================================
# SECTION 2: FUNDAMENTAL & VALUATION SCORING ENGINE (0-100)
# ==============================================================================

def compute_fundamental_score(fundamentals, technicals):
    score = 0
    breakdown = {}
    
    # 1. Profitability (Max 25 pts)
    roe = fundamentals.get("roe", 0)
    try: roe_val = float(roe) if roe != 'N/A' else 0.0
    except: roe_val = 0.0
    prof_score = 25 if roe_val > 0.20 else (18 if roe_val > 0.12 else (10 if roe_val > 0.05 else 0))
    score += prof_score
    breakdown["Profitability"] = prof_score

    # 2. Financial Leverage (Max 25 pts)
    dte = fundamentals.get("debt_to_equity", 0)
    try: dte_val = float(dte) if dte != 'N/A' else 0.0
    except: dte_val = 0.5
    health_score = 25 if dte_val < 0.5 else (15 if dte_val < 1.0 else (5 if dte_val < 2.0 else 0))
    score += health_score
    breakdown["Financial Health"] = health_score

    # 3. Valuation (Max 25 pts)
    pe = fundamentals.get("pe_ratio", 0)
    try: pe_val = float(pe) if pe != 'N/A' else 0.0
    except: pe_val = 30.0
    val_score = 25 if 0 < pe_val <= 20 else (18 if 20 < pe_val <= 35 else (10 if 35 < pe_val <= 50 else 5))
    score += val_score
    breakdown["Valuation"] = val_score

    # 4. Technical Momentum (Max 25 pts)
    rsi = technicals.get("rsi_14", 50)
    tech_score = 25 if 50 <= rsi <= 70 else (15 if 40 <= rsi < 50 else (10 if rsi > 70 else 5))
    score += tech_score
    breakdown["Technical Momentum"] = tech_score

    return score, breakdown

# ==============================================================================
# SECTION 3: MARKET REGIME DETECTION ENGINE
# ==============================================================================

def detect_market_regime():
    try:
        nifty = yf.Ticker("^NSEI").history(period="6mo")
        vix_hist = yf.Ticker("^INDIAVIX").history(period="5d")
        
        if nifty.empty:
            return {"regime": "Normal / Sideways", "vix": 15.0, "recommendation": "Maintain standard risk controls."}
            
        curr_nifty = nifty['Close'].iloc[-1]
        ema_50 = nifty['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema_200 = nifty['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        curr_vix = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 15.0
        
        if curr_nifty > ema_50 and ema_50 > ema_200 and curr_vix < 18:
            regime = "Bull Market (Strong Trend)"
            rec = "Prioritize Trend-Following & Growth strategies."
        elif curr_nifty < ema_50 and curr_vix > 22:
            regime = "Bear Market / High Stress"
            rec = "Capital Preservation emphasis. Reduce position sizes."
        else:
            regime = "Sideways / Rangebound"
            rec = "Focus on Mean-Reversion & Dividend Yield setups."
            
        return {"regime": regime, "nifty_close": round(float(curr_nifty), 2), "vix": round(float(curr_vix), 2), "recommendation": rec}
    except Exception:
        return {"regime": "Sideways / Normal", "vix": 15.5, "recommendation": "Standard diversified allocation."}

# ==============================================================================
# SECTION 4: DATABASE & STORAGE ENGINE (SQLITE CLOUD)
# ==============================================================================

def get_db_connection():
    db_url = st.secrets.get("SQLITE_CLOUD_URL", "")
    if not db_url:
        import sqlite3
        conn = sqlite3.connect("stock_lab.db")
        conn.row_factory = sqlite3.Row
        return conn
    conn = sqlitecloud.connect(db_url)
    return conn

def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS recommendations (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, ticker TEXT, cap TEXT, horizon TEXT, entry REAL, target REAL, stop_loss REAL, confidence_score INTEGER, checklist_passed BOOLEAN, reason TEXT, picked TEXT DEFAULT 'No', outcome TEXT DEFAULT 'Pending', status TEXT DEFAULT 'Active', original_date TEXT, change_notes TEXT DEFAULT '')''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT UNIQUE, qty INTEGER, buy_price REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS market_cache (ticker TEXT PRIMARY KEY, timestamp DATETIME, payload_json TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS news_cache (id INTEGER PRIMARY KEY AUTOINCREMENT, source_name TEXT, timestamp DATETIME, headlines_json TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS system_memory (key TEXT PRIMARY KEY, content TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS app_config (key TEXT PRIMARY KEY, value TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS audit_reports (date_str TEXT PRIMARY KEY, report_markdown TEXT)''')
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error initializing DB: {e}")

init_db()

def get_db_config(key, default=""):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM app_config WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else default
    except Exception:
        return default

def set_db_config(key, value):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO app_config (key, value) VALUES (?, ?)", (key, str(value)))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_db_learnings():
    default_text = "# AI Quantitative Learnings\nNo past notes recorded yet."
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM system_memory WHERE key = 'learnings'")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else default_text
    except Exception:
        return default_text

def set_db_learnings(text):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO system_memory (key, content) VALUES ('learnings', ?)", (text,))
        conn.commit()
        conn.close()
    except Exception:
        pass

# ==============================================================================
# SECTION 5: DYNAMIC YFINANCE TELEMETRY & NEWS
# ==============================================================================

def get_cached_or_live_metrics(ticker_symbol):
    ticker_symbol = ticker_symbol.upper().strip()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT payload_json, timestamp FROM market_cache WHERE ticker = ?", (ticker_symbol,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[1]:
            cached_time = datetime.datetime.strptime(str(row[1]), "%Y-%m-%d %H:%M:%S")
            if (datetime.datetime.now() - cached_time).total_seconds() / 3600.0 < 6.0:
                return json.loads(row[0])
    except Exception:
        pass
            
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        if not current_price:
            hist_test = stock.history(period="1d")
            if not hist_test.empty:
                current_price = hist_test['Close'].iloc[-1]
                
        fifty_two_high = info.get('fiftyTwoWeekHigh', 1) or 1
        volume = info.get('volume', 0)
        vol_52w_high_ratio = round(volume / fifty_two_high, 4) if fifty_two_high else 0

        fundamentals = {
            "current_price": current_price,
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "roe": info.get('returnOnEquity', 'N/A'),
            "debt_to_equity": info.get('debtToEquity', 'N/A'),
            "market_cap": info.get('marketCap', 'N/A'),
            "52w_high": fifty_two_high,
            "vol_52w_high_ratio": vol_52w_high_ratio,
            "sector": info.get('sector', 'Unknown'),
            "long_name": info.get('longName', ticker_symbol)
        }
        
        technicals = {}
        hist_3m = stock.history(period="3mo")
        if not hist_3m.empty and len(hist_3m) > 15:
            rsi_series = compute_rsi(hist_3m['Close'])
            macd_line, _, _ = compute_macd(hist_3m['Close'])
            latest_rsi = rsi_series.iloc[-1]
            latest_macd = macd_line.iloc[-1]
            advanced = compute_advanced_technicals(hist_3m)
            
            technicals = {
                "rsi_14": round(float(latest_rsi), 2) if not np.isnan(latest_rsi) else 50,
                "macd": round(float(latest_macd), 2) if not np.isnan(latest_macd) else 0,
                "advanced": advanced
            }
            
        fund_score, fund_breakdown = compute_fundamental_score(fundamentals, technicals)
        
        payload = {
            "ticker": ticker_symbol,
            "status": "SUCCESS",
            "fundamentals": fundamentals,
            "technicals": technicals,
            "fundamental_score": fund_score,
            "fundamental_breakdown": fund_breakdown
        }
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO market_cache (ticker, timestamp, payload_json) VALUES (?, ?, ?)", (ticker_symbol, now_str, json.dumps(payload)))
        conn.commit()
        conn.close()
        return payload
    except Exception as e:
        logging.error(f"Error fetching yfinance for {ticker_symbol}: {e}")
        return {"ticker": ticker_symbol, "status": f"ERROR: {str(e)}", "fundamentals": {"current_price": 0, "sector": "Unknown", "long_name": ticker_symbol}, "technicals": {}, "fundamental_score": 0, "fundamental_breakdown": {}}

def fetch_hybrid_news():
    feeds = [
        {"name": "Economic Times", "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021504.cms"},
        {"name": "Moneycontrol", "url": "https://www.moneycontrol.com/rss/marketreports.xml"}
    ]
    headers = {'User-Agent': 'Mozilla/5.0'}
    master_headlines = []
    for feed in feeds:
        try:
            res = requests.get(feed["url"], headers=headers, timeout=4)
            if res.status_code == 200:
                parsed = feedparser.parse(res.text)
                for entry in parsed.entries[:10]:
                    title = entry.get("title", "").strip()
                    if title: master_headlines.append(f"[{feed['name']}] {title}")
        except Exception:
            pass
    return master_headlines[:40], []

# ==============================================================================
# SECTION 6: STREAMLIT UI & INTERACTIVE TABS
# ==============================================================================

with st.sidebar:
    st.header("🔑 API & Configuration")
    saved_key = get_db_config("openrouter_key", "")
    default_key = st.secrets.get("OPENROUTER_KEY", saved_key)
    input_key = st.text_input("OpenRouter API Key", value=default_key, type="password")
    if input_key != saved_key:
        set_db_config("openrouter_key", input_key)
        st.success("API Key saved!")
        
    st.markdown("---")
    st.header("⚙️ Research Mandate")
    user_initial_context = st.text_area("Initial Context:", value="Focus on capital protection, dividend yield, and breakout momentum.")
    debug_mode = st.toggle("🐛 Debug Mode", value=True)
    st.markdown(f"**Build:** `{APP_VERSION}`")

tab_home, tab_screener, tab_risk, tab_backtest, tab_recs, tab_learnings, tab_db = st.tabs([
    "🚀 Research Engine", 
    "📊 Fundamental Screener & Bar Chart", 
    "🛡️ Position Sizing & Risk",
    "🧪 Strategy Backtesting Lab",
    "📋 Ledger & Portfolio",
    "🧠 AI Learnings",
    "🗄️ System & DB"
])

# --- TAB 1: RESEARCH ENGINE WITH DYNAMIC SEARCH ---
with tab_home:
    st.header("Institutional Agentic Research & Custom Ticker Analysis")
    
    # Dynamic Ticker Search Input
    user_custom_tickers = st.text_input(
        "Enter Stock Tickers (comma-separated for custom scan or single search):", 
        value="RELIANCE.NS, TCS.NS, INFY.NS, AAPL, NVDA"
    )
    ticker_list = [t.strip().upper() for t in user_custom_tickers.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1: user_confidence = st.slider("Pre-Trade Confidence Score (1-10)", 1, 10, 7)
    with col2: checklist_passed = st.checkbox("Pre-Trade Rules Verified", value=True)

    if st.button("🔍 Run Custom Ticker AI Research Scan", type="primary", use_container_width=True):
        openrouter_key = st.secrets.get("OPENROUTER_KEY", get_db_config("openrouter_key", ""))
        if not openrouter_key:
            st.error("Please provide your OpenRouter API Key.")
        else:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            status_text.markdown("📊 Fetching live telemetry for custom ticker universe...")
            progress_bar.progress(30)
            
            market_data = [get_cached_or_live_metrics(t) for t in ticker_list]
            news, _ = fetch_hybrid_news()
            regime = detect_market_regime()
            
            status_text.markdown("🤖 Generating AI Research & Valuation Report...")
            progress_bar.progress(70)
            
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
                prompt = f"""
                Analyze the following custom stock telemetry and news payload:
                Mandate: {user_initial_context}
                Market Regime: {json.dumps(regime)}
                Stock Data: {json.dumps(market_data, indent=2)}
                News: {json.dumps(news[:20])}
                
                Provide a structured professional research report including entry price, target, stop-loss, and investment verdict for these stocks.
                """
                res = client.chat.completions.create(model="openrouter/free", messages=[{"role": "user", "content": prompt}], temperature=0.3)
                progress_bar.progress(100)
                status_text.success("Analysis Complete!")
                st.markdown(res.choices[0].message.content)
            except Exception as e:
                st.error(f"AI Generation Error: {e}")

# --- TAB 2: FUNDAMENTAL SCREENER & INTUITIVE BAR GRAPH ---
with tab_screener:
    st.header("📊 Fundamental Score Breakdown & Intuitive Bar Graph")
    st.write("Select or type any stock ticker to inspect its core fundamental health dimensions visually.")
    
    selected_screener_ticker = st.text_input("Enter Ticker to Inspect (e.g. TCS.NS, AAPL, RELIANCE.NS):", value="TCS.NS").upper().strip()
    
    if selected_screener_ticker:
        with st.spinner(f"Analyzing fundamentals for {selected_screener_ticker}..."):
            metrics = get_cached_or_live_metrics(selected_screener_ticker)
            score = metrics.get("fundamental_score", 0)
            breakdown = metrics.get("fundamental_breakdown", {})
            fund = metrics.get("fundamentals", {})
            tech = metrics.get("technicals", {})
            
            st.subheader(f"📌 {fund.get('long_name', selected_screener_ticker)} ({selected_screener_ticker})")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Overall Fundamental Score", f"{score} / 100")
            with c2: st.metric("Current Price", f"₹{fund.get('current_price', 0):,.2f}" if fund.get('current_price', 0) else "N/A")
            with c3: st.metric("Trailing P/E Ratio", str(fund.get('pe_ratio', 'N/A')))
            with c4: st.metric("14-Day RSI", str(tech.get('rsi_14', 'N/A')))
            
            st.markdown("### 📈 Fundamental Component Breakdown (Bar Chart)")
            if breakdown:
                df_breakdown = pd.DataFrame(list(breakdown.items()), columns=["Dimension", "Score (Max 25)"]).set_index("Dimension")
                # Render clean intuitive bar chart in Streamlit
                st.bar_chart(df_breakdown, height=300)
                
                st.markdown("**Scoring Details:**")
                for dim, val in breakdown.items():
                    st.write(f"- **{dim}:** `{val} / 25`")
            else:
                st.info("No fundamental score breakdown available for this ticker.")

# --- TAB 3: POSITION SIZING & RISK CALCULATOR ---
with tab_risk:
    st.header("🛡️ Position Sizing & Risk Management Calculator")
    with st.form("risk_calculator"):
        c1, c2, c3 = st.columns(3)
        with c1: total_portfolio = st.number_input("Total Portfolio Value (₹)", value=500000, step=10000)
        with c2: risk_pct = st.slider("Max Acceptable Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
        with c3: calc_ticker = st.text_input("Ticker Symbol", value="RELIANCE.NS")
        
        c4, c5 = st.columns(2)
        with c4: entry_p = st.number_input("Planned Entry Price (₹)", value=1000.0, step=10.0)
        with c5: stop_l = st.number_input("Planned Stop-Loss Price (₹)", value=950.0, step=10.0)
        
        calc_btn = st.form_submit_button("Calculate Risk & Position Size")
        
    if calc_btn:
        if stop_l >= entry_p:
            st.error("Stop-Loss must be lower than Entry Price for a Long trade.")
        else:
            max_loss_amount = total_portfolio * (risk_pct / 100.0)
            risk_per_share = entry_p - stop_l
            max_shares = int(max_loss_amount / risk_per_share)
            total_invested = max_shares * entry_p
            alloc_pct = round((total_invested / total_portfolio) * 100, 2)
            
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Max Shares", f"{max_shares} shares")
            with m2: st.metric("Total Investment", f"₹{total_invested:,.2f}")
            with m3: st.metric("Max Dollar Risk", f"₹{max_loss_amount:,.2f}")
            with m4: st.metric("Portfolio Allocation", f"{alloc_pct}%")

# --- TAB 4: HISTORICAL BACKTESTING LAB ---
with tab_backtest:
    st.header("🧪 Strategy Backtesting & Validation Lab")
    bt_ticker = st.text_input("Enter Ticker for 1-Year Backtest:", value="TCS.NS").upper().strip()
    if st.button("🚀 Run Backtest"):
        with st.spinner(f"Running backtest for {bt_ticker}..."):
            try:
                stock = yf.Ticker(bt_ticker)
                df = stock.history(period="1y")
                if not df.empty and len(df) > 50:
                    df['Returns'] = df['Close'].pct_change()
                    cagr = round(float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100), 2)
                    sharpe = round(float((df['Returns'].mean() / (df['Returns'].std() + 1e-6)) * np.sqrt(252)), 2)
                    
                    b1, b2 = st.columns(2)
                    with b1: st.metric("1-Year Return / CAGR", f"{cagr}%")
                    with b2: st.metric("Sharpe Ratio", sharpe)
                else:
                    st.warning("Insufficient historical data for this ticker.")
            except Exception as e:
                st.error(f"Backtest error: {e}")

# --- TAB 5: LEDGER & PORTFOLIO ---
with tab_recs:
    st.header("📋 Active Recommendations Ledger")
    conn = get_db_connection()
    df_recs = pd.read_sql_query("SELECT id, date, ticker, cap, horizon, entry, target, stop_loss, outcome, status FROM recommendations ORDER BY id DESC", conn)
    conn.close()
    if not df_recs.empty:
        st.dataframe(df_recs, use_container_width=True)
    else:
        st.info("No recommendations logged yet.")

# --- TAB 6: AI LEARNINGS MEMORY ---
with tab_learnings:
    st.header("🧠 AI Quantitative Self-Reflection Memory")
    learnings_text = get_db_learnings()
    updated = st.text_area("Cloud Memory Content", value=learnings_text, height=300)
    if st.button("Save Memory"):
        set_db_learnings(updated)
        st.success("Learnings saved to SQLite Cloud!")

# --- TAB 7: DATABASE MANAGEMENT ---
with tab_db:
    st.header("🗄️ SQLite Cloud Table Inspector")
    conn = get_db_connection()
    cursor = conn.cursor()
    tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
    conn.close()
    st.write(f"Connected Tables: `{', '.join(tables)}`")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: grey; font-size: 13px;'>"
    f"📈 Enterprise AI Stock Research Platform | <b>Version:</b> {APP_VERSION} | <b>Last Build:</b> {LAST_UPDATED}"
    f"</div>",
    unsafe_allow_html=True
)
