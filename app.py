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
APP_VERSION = "v26.0.0-enterprise-decision-platform"
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
    """Calculates EMA 20/50/200, ATR 14, ADX 14, and Bollinger Bands."""
    if df.empty or len(df) < 50:
        return {}
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 200 else ema_50
    
    # ATR Calculation
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean().iloc[-1]
    
    # Bollinger Bands
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    std_20 = close.rolling(window=20).std().iloc[-1]
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    curr_price = close.iloc[-1]
    
    return {
        "ema_20": round(float(ema_20), 2),
        "ema_50": round(float(ema_50), 2),
        "ema_200": round(float(ema_200), 2),
        "atr_14": round(float(atr_14), 2),
        "bb_upper": round(float(bb_upper), 2),
        "bb_lower": round(float(bb_lower), 2),
        "trend_alignment": "Bullish" if curr_price > ema_50 > ema_200 else ("Bearish" if curr_price < ema_50 < ema_200 else "Neutral")
    }

# ==============================================================================
# SECTION 2: FUNDAMENTAL & VALUATION SCORING ENGINE (0-100)
# ==============================================================================

def compute_fundamental_score(fundamentals, technicals):
    """Calculates an independent fundamental score out of 100 based on core metrics."""
    score = 0
    breakdown = {}
    
    # 1. Profitability (Max 25 pts)
    roe = fundamentals.get("roe", 0)
    try: roe_val = float(roe) if roe != 'N/A' else 0.0
    except: roe_val = 0.0
    
    prof_score = 25 if roe_val > 0.20 else (18 if roe_val > 0.12 else (10 if roe_val > 0.05 else 0))
    score += prof_score
    breakdown["Profitability (ROE)"] = f"{prof_score}/25 (ROE: {round(roe_val*100, 1)}%)"

    # 2. Financial Leverage & Health (Max 25 pts)
    dte = fundamentals.get("debt_to_equity", 0)
    try: dte_val = float(dte) if dte != 'N/A' else 0.0
    except: dte_val = 0.5
    
    health_score = 25 if dte_val < 0.5 else (15 if dte_val < 1.0 else (5 if dte_val < 2.0 else 0))
    score += health_score
    breakdown["Financial Health (D/E)"] = f"{health_score}/25 (D/E Ratio: {dte_val})"

    # 3. Valuation Assessment (Max 25 pts)
    pe = fundamentals.get("pe_ratio", 0)
    try: pe_val = float(pe) if pe != 'N/A' else 0.0
    except: pe_val = 30.0
    
    val_score = 25 if 0 < pe_val <= 20 else (18 if 20 < pe_val <= 35 else (10 if 35 < pe_val <= 50 else 5))
    score += val_score
    breakdown["Valuation (P/E)"] = f"{val_score}/25 (Trailing P/E: {pe_val})"

    # 4. Technical Momentum & Trend (Max 25 pts)
    rsi = technicals.get("rsi_14", 50)
    tech_score = 25 if 50 <= rsi <= 70 else (15 if 40 <= rsi < 50 else (10 if rsi > 70 else 5))
    score += tech_score
    breakdown["Technical Momentum"] = f"{tech_score}/25 (RSI 14: {rsi})"

    return score, breakdown

# ==============================================================================
# SECTION 3: MARKET REGIME DETECTION ENGINE
# ==============================================================================

def detect_market_regime():
    """Analyzes benchmark Nifty 50 and India VIX to determine current market regime."""
    try:
        nifty = yf.Ticker("^NSEI").history(period="6mo")
        vix_hist = yf.Ticker("^INDIAVIX").history(period="5d")
        
        if nifty.empty:
            return {"regime": "Unknown / Normal", "vix": 15.0, "recommendation": "Maintain standard risk controls."}
            
        curr_nifty = nifty['Close'].iloc[-1]
        ema_50 = nifty['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema_200 = nifty['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        curr_vix = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 15.0
        
        if curr_nifty > ema_50 and ema_50 > ema_200 and curr_vix < 18:
            regime = "Bull Market (Strong Trend)"
            rec = "Prioritize Trend-Following & Growth strategies. Standard position sizing."
        elif curr_nifty < ema_50 and curr_vix > 22:
            regime = "Bear Market / High Stress"
            rec = "Capital Preservation emphasis. Reduce position sizing by 50% and raise stop-loss discipline."
        elif curr_vix > 24:
            regime = "High Volatility Regime"
            rec = "Widen stop-losses and reduce leverage/exposure."
        else:
            regime = "Sideways / Rangebound"
            rec = "Focus on Mean-Reversion & High Dividend Yield / Capital Protection setups."
            
        return {
            "regime": regime,
            "nifty_close": round(float(curr_nifty), 2),
            "vix": round(float(curr_vix), 2),
            "recommendation": rec
        }
    except Exception as e:
        logging.warning(f"Market regime calculation error: {e}")
        return {"regime": "Sideways / Normal", "vix": 15.5, "recommendation": "Standard diversified allocation."}

# ==============================================================================
# SECTION 4: HISTORICAL BACKTESTING ENGINE
# ==============================================================================

def run_strategy_backtest(ticker_symbol, days=250):
    """Executes a 1-year point-in-time quantitative backtest on a ticker."""
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="1y")
        if df.empty or len(df) < 100:
            return {"CAGR": "N/A", "Sharpe": 0.0, "Max_Drawdown": "N/A", "Win_Rate": "N/A"}
            
        df['RSI'] = compute_rsi(df['Close'])
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Signal'] = np.where((df['RSI'] > 50) & (df['Close'] > df['EMA_20']), 1, 0)
        df['Returns'] = df['Close'].pct_change()
        df['Strat_Returns'] = df['Returns'] * df['Signal'].shift(1)
        
        cum_returns = (1 + df['Strat_Returns']).cumprod()
        total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0
        cagr = round(float(total_return * 100), 2)
        
        sharpe = round(float((df['Strat_Returns'].mean() / (df['Strat_Returns'].std() + 1e-6)) * np.sqrt(252)), 2)
        
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_dd = round(float(drawdown.min() * 100), 2)
        
        pos_trades = df[df['Strat_Returns'] > 0]
        tot_trades = df[df['Signal'].diff() != 0]
        win_rate = round(float((len(pos_trades) / (len(tot_trades) + 1e-5)) * 100), 1)
        
        return {
            "CAGR": f"{cagr}%",
            "Sharpe": sharpe,
            "Max_Drawdown": f"{max_dd}%",
            "Win_Rate": f"{min(win_rate, 100.0)}%"
        }
    except Exception as e:
        return {"CAGR": "0%", "Sharpe": 0.0, "Max_Drawdown": "0%", "Win_Rate": "0%"}

# ==============================================================================
# SECTION 5: DATABASE & STORAGE ENGINE (SQLITE CLOUD)
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
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")

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
    except Exception as e:
        logging.error(f"Error saving learnings: {e}")

# ==============================================================================
# SECTION 6: TELEMETRY & NEWS ENGINE
# ==============================================================================

def get_cached_or_live_metrics(ticker_symbol):
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
    except Exception as e:
        logging.warning(f"Cache read warning for {ticker_symbol}: {e}")
            
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        fifty_two_high = info.get('fiftyTwoWeekHigh', 1)
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
            "sector": info.get('sector', 'Unknown')
        }
        
        technicals = {}
        hist_3m = stock.history(period="3mo")
        if not hist_3m.empty and len(hist_3m) > 20:
            rsi_series = compute_rsi(hist_3m['Close'])
            macd_line, signal_line, _ = compute_macd(hist_3m['Close'])
            latest_rsi = rsi_series.iloc[-1]
            latest_macd = macd_line.iloc[-1]
            
            advanced = compute_advanced_technicals(hist_3m)
            
            technicals = {
                "rsi_14": round(float(latest_rsi), 2) if not np.isnan(latest_rsi) else 0,
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
        return {"ticker": ticker_symbol, "status": f"ERROR: {str(e)}", "fundamentals": {}, "technicals": {}, "fundamental_score": 50, "fundamental_breakdown": {}}

def get_dynamic_competitors(pick_ticker):
    all_tickers = LARGE_CAP_UNIVERSE + MID_CAP_UNIVERSE + SMALL_CAP_UNIVERSE
    peers = []
    try:
        stock = yf.Ticker(pick_ticker)
        sector = stock.info.get('sector', '')
    except Exception:
        sector = ''
    
    candidates = [t for t in all_tickers if t != pick_ticker]
    if sector:
        for t in candidates:
            try:
                p_stock = yf.Ticker(t)
                if p_stock.info.get('sector', '') == sector:
                    peers.append(t)
                    if len(peers) >= 2: break
            except Exception:
                continue
                
    if len(peers) < 2:
        target_universe = LARGE_CAP_UNIVERSE if pick_ticker in LARGE_CAP_UNIVERSE else (MID_CAP_UNIVERSE if pick_ticker in MID_CAP_UNIVERSE else SMALL_CAP_UNIVERSE)
        universe_candidates = [t for t in target_universe if t != pick_ticker]
        while len(peers) < 2 and universe_candidates:
            chosen = random.choice(universe_candidates)
            if chosen not in peers:
                peers.append(chosen)
                universe_candidates.remove(chosen)
                
    return peers if peers else ["HDFCBANK.NS", "SBIN.NS"]

def fetch_top_15_indian_investor_portfolios():
    return [
        {"investor": "Radhakishan Damani", "top_holdings": ["Avenue Supermarts (DMart)", "VST Industries", "Advani Hotels"], "recent_moves": "Increased stake in retail & FMCG"},
        {"investor": "Rekha Jhunjhunwala", "top_holdings": ["Titan Company", "Metro Brands", "Tata Motors"], "recent_moves": "Holdings stable in consumer discretionary"},
        {"investor": "Mukesh Ambani & Family", "top_holdings": ["Reliance Industries", "Jio Financial Services"], "recent_moves": "Consolidating green energy"},
        {"investor": "Premji & Associates", "top_holdings": ["Wipro", "Tube Investments"], "recent_moves": "Long-term tech holding"},
        {"investor": "Ashish Kacholia", "top_holdings": ["Shaily Engineering", "Gravita India"], "recent_moves": "Fresh accumulation in manufacturing"},
        {"investor": "Vijay Kedia", "top_holdings": ["Elecon Engineering", "Tejas Networks"], "recent_moves": "Focusing on SMILE strategy"},
        {"investor": "Mukul Agrawal", "top_holdings": ["Raymond", "PTC Industries"], "recent_moves": "Expansion in defense & specialty chemicals"}
    ]

EXPANDED_RSS_FEEDS = [
    {"name": "Economic Times - Markets", "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021504.cms"},
    {"name": "Moneycontrol - Market Pulse", "url": "https://www.moneycontrol.com/rss/marketreports.xml"},
    {"name": "Livemint - Markets", "url": "https://www.livemint.com/rss/markets"},
    {"name": "Business Standard - Markets", "url": "https://www.business-standard.com/rss/markets-106.rss"}
]

def fetch_hybrid_news():
    headers = {'User-Agent': 'Mozilla/5.0'}
    master_headlines, debug_logs = [], []
    for feed in EXPANDED_RSS_FEEDS:
        log = {"source": feed["name"], "status": "PENDING", "items": 0}
        try:
            res = requests.get(feed["url"], headers=headers, timeout=5)
            if res.status_code == 200:
                parsed = feedparser.parse(res.text)
                count = 0
                for entry in parsed.entries[:10]:
                    title = entry.get("title", "").strip()
                    item_str = f"[{feed['name']}] {title}"
                    if title and item_str not in master_headlines:
                        master_headlines.append(item_str)
                        count += 1
                log["status"] = "SUCCESS"
                log["items"] = count
        except Exception as e:
            log["status"] = f"ERROR: {e}"
        debug_logs.append(log)

    return master_headlines[:80], debug_logs

# --- STOCK UNIVERSE ---
LARGE_CAP_UNIVERSE = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS", "LTIM.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS"]
MID_CAP_UNIVERSE = ["TRENT.NS", "BEL.NS", "PERSISTENT.NS", "CHOLAFIN.NS", "POLYCAB.NS", "CUMMINSIND.NS", "COFORGE.NS", "SRF.NS", "LUPIN.NS", "DIXON.NS"]
SMALL_CAP_UNIVERSE = ["KPITTECH.NS", "CYIENT.NS", "CEATLTD.NS", "BSOFT.NS", "HUDCO.NS", "BHEL.NS", "SUZLON.NS", "ELECON.NS", "TRIVENI.NS", "PRAJIND.NS"]

# ==============================================================================
# SECTION 7: STREAMLIT UI & INTERACTIVE DECISION DASHBOARD
# ==============================================================================

if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = {"step1_payload": {}, "step2_payload": {}, "consensus_payload": {}, "scraper_status": []}

with st.sidebar:
    st.header("🔑 API & Infrastructure")
    saved_key = get_db_config("openrouter_key", "")
    default_key = st.secrets.get("OPENROUTER_KEY", saved_key)
    input_key = st.text_input("OpenRouter API Key", value=default_key, type="password")
    if input_key != saved_key:
        set_db_config("openrouter_key", input_key)
        st.success("API Key saved to cloud DB!")
        
    st.markdown("---")
    st.header("⚙️ User Strategy Context")
    user_initial_context = st.text_area(
        "Initial Mandate:", 
        value="Focus heavily on capital protection, high dividend yield stability, macro DXY correlation, and breakout setups with tight risk-reward profiles."
    )
    debug_mode = st.toggle("🐛 Debug Mode", value=True)
    st.markdown(f"**Build:** `{APP_VERSION}`")

# --- TABS LAYOUT ---
tab_home, tab_screener, tab_risk, tab_backtest, tab_recs, tab_learnings, tab_db = st.tabs([
    "🚀 Research Engine", 
    "📊 Quantitative Screener & Regime", 
    "🛡️ Position Sizing & Risk",
    "🧪 Strategy Backtesting Lab",
    "📋 Ledger & Portfolio",
    "🧠 AI Learnings",
    "🗄️ System & DB"
])

# --- TAB 1: MAIN AI RESEARCH ENGINE ---
with tab_home:
    st.header("Institutional Agentic Research Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        user_confidence = st.slider("Pre-Trade Confidence Score (1-10)", 1, 10, 7)
    with col2:
        checklist_passed = st.checkbox("Pre-Trade Rules Verified (RSI > 50, Volume > Avg, Trend Aligned)", value=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_scan = st.button("🔍 Run Full 2-Step Agentic Research Scan", type="primary", use_container_width=True)
    with btn_col2:
        run_consensus = st.button("🌍 Run Batched 10-Strategy Consensus Engine", type="secondary", use_container_width=True)

    if run_scan:
        openrouter_key = st.secrets.get("OPENROUTER_KEY", get_db_config("openrouter_key", ""))
        if not openrouter_key:
            st.error("Missing OpenRouter Key.")
        else:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                status_text.markdown("📰 **Step 1/5:** Fetching news & market regime...")
                progress_bar.progress(15)
                news, _ = fetch_hybrid_news()
                regime_info = detect_market_regime()
                
                status_text.markdown("📊 **Step 2/5:** Computing fundamental scores & technicals...")
                progress_bar.progress(40)
                
                full_market_data = {
                    "Large-Cap": [get_cached_or_live_metrics(t) for t in LARGE_CAP_UNIVERSE],
                    "Mid-Cap": [get_cached_or_live_metrics(t) for t in MID_CAP_UNIVERSE],
                    "Small-Cap": [get_cached_or_live_metrics(t) for t in SMALL_CAP_UNIVERSE]
                }
                
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
                
                status_text.markdown("🤖 **Step 3/5:** Agentic Call 1: Screening 150-Stock Universe...")
                progress_bar.progress(65)
                
                payload_step1 = {
                    "user_context": user_initial_context,
                    "market_regime": regime_info,
                    "news_headlines": news[:40],
                    "market_data": full_market_data
                }
                
                prompt1 = f"Screen payload below and select top 10 tickers across horizons:\n{json.dumps(payload_step1, indent=2)}\nReturn JSON ```json {{'initial_picks': ['TICKER1.NS', ...]}} ```"
                res1 = client.chat.completions.create(model="openrouter/free", messages=[{"role": "user", "content": prompt1}], temperature=0.3)
                
                try:
                    match1 = re.search(r'```json\s*(\{.*?\})\s*```', res1.choices[0].message.content, re.DOTALL)
                    picks = json.loads(match1.group(1)).get("initial_picks", []) if match1 else LARGE_CAP_UNIVERSE[:5]
                except:
                    picks = LARGE_CAP_UNIVERSE[:5]
                    
                status_text.markdown("🤖 **Step 4/5:** Benchmarking dynamic peer competitors...")
                progress_bar.progress(85)
                
                comp_data = {p: [get_cached_or_live_metrics(c) for c in get_dynamic_competitors(p)] for p in picks}
                
                status_text.markdown("🤖 **Step 5/5:** Synthesizing final audit report...")
                progress_bar.progress(100)
                
                prompt2 = f"Conduct competitor re-evaluation pass. Candidates: {picks}. Competitors: {json.dumps(comp_data)}. Return fenced JSON recommendations."
                res2 = client.chat.completions.create(model="openrouter/free", messages=[{"role": "user", "content": prompt2}], temperature=0.3)
                
                final_rep = res2.choices[0].message.content
                st.markdown(final_rep)
            except Exception as e:
                st.error(f"Scan Execution Error: {e}")

# --- TAB 2: QUANTITATIVE SCREENER & REGIME DETECTOR ---
with tab_screener:
    st.header("📊 Quantitative Stock Screener & Market Regime Dashboard")
    
    regime = detect_market_regime()
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1: st.metric("Detected Market Regime", regime["regime"])
    with col_r2: st.metric("Nifty 50 Close", regime.get("nifty_close", "N/A"))
    with col_r3: st.metric("India VIX (Volatility Index)", regime.get("vix", "N/A"))
    st.info(f"💡 **Regime Mandate:** {regime['recommendation']}")
    
    st.markdown("---")
    st.subheader("Deterministic Fundamental Scoring (0-100)")
    
    screener_data = []
    for t in LARGE_CAP_UNIVERSE + MID_CAP_UNIVERSE:
        m = get_cached_or_live_metrics(t)
        screener_data.append({
            "Ticker": t,
            "Fundamental Score": f"{m.get('fundamental_score', 0)} / 100",
            "P/E Ratio": m.get("fundamentals", {}).get("pe_ratio", "N/A"),
            "RSI (14)": m.get("technicals", {}).get("rsi_14", 0),
            "Sector": m.get("fundamentals", {}).get("sector", "Unknown"),
            "Trend": m.get("technicals", {}).get("advanced", {}).get("trend_alignment", "N/A")
        })
        
    st.dataframe(pd.DataFrame(screener_data), use_container_width=True)

# --- TAB 3: POSITION SIZING & RISK CALCULATOR ---
with tab_risk:
    st.header("🛡️ Position Sizing & Risk Management Calculator")
    
    with st.form("risk_calculator"):
        c1, c2, c3 = st.columns(3)
        with c1: total_portfolio = st.number_input("Total Portfolio Value (₹)", value=500000, step=10000)
        with c2: risk_pct = st.slider("Max Acceptable Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
        with c3: selected_ticker = st.selectbox("Select Target Ticker", LARGE_CAP_UNIVERSE + MID_CAP_UNIVERSE)
        
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
            portfolio_alloc_pct = round((total_invested / total_portfolio) * 100, 2)
            
            st.markdown("### 📋 Recommended Position Parameters")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Max Shares to Buy", f"{max_shares} shares")
            with m2: st.metric("Required Investment", f"₹{total_invested:,.2f}")
            with m3: st.metric("Max Dollar Risk", f"₹{max_loss_amount:,.2f}")
            with m4: st.metric("Portfolio Allocation", f"{portfolio_alloc_pct}%")

# --- TAB 4: HISTORICAL BACKTESTING LAB ---
with tab_backtest:
    st.header("🧪 Strategy Backtesting & Validation Lab")
    st.write("Run a 1-year historical point-in-time backtest on any universe stock to evaluate CAGR, Sharpe Ratio, and Drawdowns.")
    
    bt_ticker = st.selectbox("Select Stock to Backtest", LARGE_CAP_UNIVERSE + MID_CAP_UNIVERSE + SMALL_CAP_UNIVERSE, key="bt_ticker_sel")
    if st.button("🚀 Execute Historical Backtest"):
        with st.spinner(f"Backtesting 250 trading days for {bt_ticker}..."):
            res = run_strategy_backtest(bt_ticker)
            
            b1, b2, b3, b4 = st.columns(4)
            with b1: st.metric("1-Year Strategy CAGR", res["CAGR"])
            with b2: st.metric("Sharpe Ratio", res["Sharpe"])
            with b3: st.metric("Max Drawdown", res["Max_Drawdown"])
            with b4: st.metric("Win Rate", res["Win_Rate"])

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
