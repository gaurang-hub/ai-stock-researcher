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
from openai import OpenAI
import sqlitecloud

# --- LOGGING SETUP FOR STREAMLIT CLOUD ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- APP VERSION CONFIG ---
APP_VERSION = "v23.0.0-master-complete-cloud"
LAST_UPDATED = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

st.set_page_config(page_title="AI Stock Researcher & Quantitative Lab", layout="wide")
st.title("📈 AI Stock Researcher & Quantitative Lab [Master Complete Cloud Edition]")

# --- NATIVE TECHNICAL INDICATORS ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

# --- DB CONNECTION HELPER ---
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

# --- CLOUD CONFIG & MEMORY HELPERS ---
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
        cursor.execute("INSERT OR REPLACE INTO system_memory (key, value) VALUES ('learnings', ?)", (text,))
        conn.commit()
        conn.close()
    except Exception:
        pass

# --- SESSION STATE DEBUGS ---
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = {
        "last_news": [],
        "last_market_data": {},
        "step1_payload": {},
        "step2_payload": {},
        "consensus_payload": {},
        "scraper_status": []
    }

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🔑 API & Safeguards")
    saved_key = get_db_config("openrouter_key", "")
    default_key = st.secrets.get("OPENROUTER_KEY", saved_key)
    input_key = st.text_input("OpenRouter API Key", value=default_key, type="password")
    if input_key != saved_key:
        set_db_config("openrouter_key", input_key)
        st.success("API Key saved to cloud DB!")
        
    st.markdown("---")
    st.header("📝 Initial User Context Input")
    user_initial_context = st.text_area(
        "Provide initial context/guidelines for the AI before research scan:", 
        value="Focus heavily on capital protection, high dividend yield stability, macro DXY correlation, and breakout setups with tight risk-reward profiles."
    )

    st.markdown("---")
    st.header("🛠️ Debug Controls")
    debug_mode = st.toggle("🐛 Enable Debug Mode", value=True)
    
    st.markdown("---")
    st.markdown(f"**Version:** `{APP_VERSION}`")

# --- DATA ENGINEERING: 6-HOUR WINDOW & PRUNING ---
def prune_old_entries():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM market_cache WHERE timestamp < datetime('now', '-1 day')")
        cursor.execute("DELETE FROM news_cache WHERE timestamp < datetime('now', '-1 day')")
        conn.commit()
        conn.close()
    except Exception:
        pass

prune_old_entries()

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
            "vol_52w_high_ratio": vol_52w_high_ratio
        }
        
        technicals = {}
        hist_3m = stock.history(period="3mo")
        if not hist_3m.empty and len(hist_3m) > 20:
            rsi_series = compute_rsi(hist_3m['Close'])
            macd_series = compute_macd(hist_3m['Close'])
            latest_rsi, latest_macd = rsi_series.iloc[-1], macd_series.iloc[-1]
            technicals = {
                "rsi_14": round(float(latest_rsi), 2) if not np.isnan(latest_rsi) else 0,
                "macd": round(float(latest_macd), 2) if not np.isnan(latest_macd) else 0
            }
            
        payload = {"ticker": ticker_symbol, "status": "SUCCESS", "fundamentals": fundamentals, "technicals": technicals}
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO market_cache (ticker, timestamp, payload_json) VALUES (?, ?, ?)", (ticker_symbol, now_str, json.dumps(payload)))
        conn.commit()
        conn.close()
        return payload
    except Exception as e:
        logging.error(f"Error fetching yfinance for {ticker_symbol}: {e}")
        return {"ticker": ticker_symbol, "status": f"ERROR: {str(e)}", "fundamentals": {}, "technicals": {}}

# --- TOP 15 INDIAN SUPERSTAR INVESTOR PORTFOLIOS ---
def fetch_top_15_indian_investor_portfolios():
    return [
        {"investor": "Radhakishan Damani", "top_holdings": ["Avenue Supermarts (DMart)", "VST Industries", "Advani Hotels"], "recent_moves": "Increased stake in retail & FMCG", "est_portfolio_value_cr": 149300},
        {"investor": "Rekha Jhunjhunwala", "top_holdings": ["Titan Company", "Metro Brands", "Tata Motors", "Crisil", "NCC"], "recent_moves": "Holdings stable in consumer discretionary & infra", "est_portfolio_value_cr": 44150},
        {"investor": "Mukesh Ambani & Family", "top_holdings": ["Reliance Industries", "Jio Financial Services"], "recent_moves": "Consolidating green energy & telecom expansion", "est_portfolio_value_cr": 319200},
        {"investor": "Premji & Associates", "top_holdings": ["Wipro", "Tube Investments"], "recent_moves": "Long-term tech holding", "est_portfolio_value_cr": 136500},
        {"investor": "Ashish Kacholia", "top_holdings": ["Shaily Engineering", "ESDS Software", "Beta Drugs", "Gravita India"], "recent_moves": "Fresh accumulation in mid/small-cap manufacturing", "est_portfolio_value_cr": 3020},
        {"investor": "Vijay Kedia", "top_holdings": ["Elecon Engineering", "Tejas Networks", "Vaibhav Global", "Atul Auto"], "recent_moves": "Focusing on SMILE strategy (Small cap, High ROI)", "est_portfolio_value_cr": 1550},
        {"investor": "Mukul Agrawal", "top_holdings": ["Raymond", "PTC Industries", "Neuland Labs", "Sarda Energy"], "recent_moves": "Aggressive expansion in defense & specialty chemicals", "est_portfolio_value_cr": 4800},
        {"investor": "Sunil Singhania (Abakkus)", "top_holdings": ["Jindal Stainless", "Techno Electric", "IIFL Securities"], "recent_moves": "Building positions in capital goods & power infra", "est_portfolio_value_cr": 3200},
        {"investor": "Dolly Khanna", "top_holdings": ["Prakash Pipes", "Chennai Petro", "Nitin Spinners", "Mangalore Chemicals"], "recent_moves": "Rotated into fertilizers and commodity textiles", "est_portfolio_value_cr": 520},
        {"investor": "Nemish Shah", "top_holdings": ["Lakshmi Machine Works", "EID Parry", "Zydus Wellness"], "recent_moves": "High-conviction value compounders", "est_portfolio_value_cr": 2800},
        {"investor": "Ashish Dhawan", "top_holdings": ["Religare Enterprises", "IIFL Finance", "Equitas Small Finance Bank"], "recent_moves": "Overweight on BFSI turnaround plays", "est_portfolio_value_cr": 2170},
        {"investor": "Anil Kumar Goel", "top_holdings": ["KRBL", "TCPL Packaging", "Triveni Engineering"], "recent_moves": "Concentrated in sugar and packaging industries", "est_portfolio_value_cr": 2500},
        {"investor": "Akash Bhanshali", "top_holdings": ["Gujarat Fluorochemicals", "Laurus Labs", "Paytm (One97)"], "recent_moves": "Chemicals & specialty pharma focus", "est_portfolio_value_cr": 7870},
        {"investor": "Madhusudan Kela", "top_holdings": ["Choice International", "MKVENTURES", "Sangam India"], "recent_moves": "Mid-cap financial services accumulation", "est_portfolio_value_cr": 1850},
        {"investor": "Porinju Veliyath", "top_holdings": ["Kriti Nutrients", "Taneja Aerospace", "Orient Bell"], "recent_moves": "Micro-cap value unlocking setups", "est_portfolio_value_cr": 290}
    ]

# --- MACRO TELEMETRY ---
def fetch_macro_indicators():
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")['Close'].iloc[-1]
        yield_10y = yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]
        crude = yf.Ticker("CL=F").history(period="5d")['Close'].iloc[-1]
        return {
            "DXY_Index": round(float(dxy), 2),
            "US_10Y_Yield": round(float(yield_10y), 2),
            "Crude_Oil_USD": round(float(crude), 2)
        }
    except Exception:
        return {"DXY_Index": 104.5, "US_10Y_Yield": 4.25, "Crude_Oil_USD": 78.50}

# --- RECURSIVE CRAWLER ---
def crawl_site_recursive(base_url, max_depth=2):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    visited, articles, queue = set(), [], [(base_url, 0)]
    visited.add(base_url)
    base_domain = urlparse(base_url).netloc

    while queue and len(articles) < 50:
        current_url, depth = queue.pop(0)
        try:
            res = requests.get(current_url, headers=headers, timeout=5)
            if res.status_code != 200: continue
            soup = BeautifulSoup(res.text, 'html.parser')
            for item in soup.find_all(['h3', 'h2', 'h1', 'a'], limit=30):
                txt = item.get_text().strip()
                if len(txt) > 25 and txt not in articles:
                    articles.append(txt)
            if depth < max_depth:
                for link in soup.find_all('a', href=True):
                    abs_url = urljoin(current_url, link['href'])
                    if urlparse(abs_url).netloc == base_domain and abs_url not in visited:
                        if any(k in abs_url.lower() for k in ['market', 'stock', 'news', 'business', 'economy']):
                            visited.add(abs_url)
                            queue.append((abs_url, depth + 1))
                            if len(visited) > 30: break
        except Exception:
            pass
    return articles

# --- HYBRID NEWS CORPUS WITH 6-HOUR WINDOW ---
def fetch_hybrid_news():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT headlines_json, timestamp FROM news_cache WHERE source_name = 'MASTER_CORPUS'")
        row = cursor.fetchone()
        conn.close()
        
        if row and row[1]:
            cached_time = datetime.datetime.strptime(str(row[1]), "%Y-%m-%d %H:%M:%S")
            if (datetime.datetime.now() - cached_time).total_seconds() / 3600.0 < 6.0:
                logging.info("News cache hit within 6-hour window.")
                return json.loads(row[0]), [{"source": "SQLite Cloud Cache", "status": "HIT", "items": len(json.loads(row[0]))}]
    except Exception as e:
        logging.warning(f"News cache error: {e}")

    rss_feeds = [
        {"name": "Economic Times", "url": "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms"},
        {"name": "Livemint", "url": "https://www.livemint.com/rss/markets"},
        {"name": "Business Standard", "url": "https://www.business-standard.com/rss/markets-106.rss"}
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    master_headlines, debug_logs = [], []

    for feed in rss_feeds:
        log = {"source": feed["name"], "type": "RSS", "url": feed["url"], "status": "PENDING", "items": 0}
        try:
            res = requests.get(feed["url"], headers=headers, timeout=6)
            if res.status_code == 200:
                parsed = feedparser.parse(res.text)
                count = 0
                for entry in parsed.entries[:25]:
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

    final_headlines = master_headlines[:150]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO news_cache (source_name, timestamp, headlines_json) VALUES ('MASTER_CORPUS', ?, ?)", (now_str, json.dumps(final_headlines)))
        conn.commit()
        conn.close()
    except Exception:
        pass

    st.session_state.debug_logs["scraper_status"] = debug_logs
    st.session_state.debug_logs["last_news"] = final_headlines
    return final_headlines, debug_logs

# --- FULL 150-STOCK UNIVERSE ---
LARGE_CAP_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS", "LTIM.NS", "HINDUNILVR.NS", "ITC.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ASIANPAINT.NS", "DMART.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "TATAMOTORS.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "TATASTEEL.NS",
    "WIPRO.NS", "HCLTECH.NS", "GRASIM.NS", "TECHM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "NESTLEIND.NS", "CIPLA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS", "BPCL.NS", "TATACONSUM.NS", "SHRIRAMFIN.NS", "ZOMATO.NS", "HDFCBANK.NS"
]

MID_CAP_UNIVERSE = [
    "TRENT.NS", "BEL.NS", "PERSISTENT.NS", "CHOLAFIN.NS", "POLYCAB.NS", "JSL.NS", "CUMMINSIND.NS", "MAXHEALTH.NS", "COFORGE.NS", "SRF.NS",
    "PAGEIND.NS", "ABFRL.NS", "LUPIN.NS", "AUROPHARMA.NS", "PIIND.NS", "MUTHOOTFIN.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "CANBK.NS", "PNB.NS",
    "HINDPETRO.NS", "GAIL.NS", "IOC.NS", "ABBOTINDIA.NS", "ALKEM.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "BALKRISIND.NS", "MPHASIS.NS", "LTTS.NS",
    "OFSS.NS", "BHARATFORG.NS", "DIXON.NS", "OBEROIRLTY.NS", "GODREJPROP.NS", "PHOENIXLTD.NS", "PRESTIGE.NS", "DALBHARAT.NS", "RAMCOCEM.NS", "ASTRAL.NS",
    "SUPREMEIND.NS", "FINCABLES.NS", "ENDURANCE.NS", "SONACOMS.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "AMBER.NS", "ZYDUSLIFE.NS", "COLPAL.NS", "SBICARD.NS"
]

SMALL_CAP_UNIVERSE = [
    "KPITTECH.NS", "CYIENT.NS", "CEATLTD.NS", "BSOFT.NS", "HUDCO.NS", "BHEL.NS", "IDEA.NS", "ZENSARTECH.NS", "WELCORP.NS", "AEGISLOG.NS",
    "ANANDRATHI.NS", "MRPL.NS", "NUVAMA.NS", "BLUESTARCO.NS", "GRANULES.NS", "ASTEC.NS", "RBLBANK.NS", "CAMS.NS", "CDSL.NS", "BSE.NS",
    "MCX.NS", "ANGELONE.NS", "KIMS.NS", "RAINBOW.NS", "ASTERDM.NS", "CREDITACC.NS", "MMTC.NS", "HMT.NS", "ITI.NS", "TRIDENT.NS",
    "SUZLON.NS", "JPPOWER.NS", "HEG.NS", "GRAPHITE.NS", "RENUKA.NS", "EASEMYTRIP.NS", "PRAJIND.NS", "ELECON.NS", "TRIVENI.NS", "ISGEC.NS",
    "TCI.NS", "SCI.NS", "GMDCLTD.NS", "MOIL.NS", "NFL.NS", "RCF.NS", "CHAMBLFERT.NS", "GNFC.NS", "RAJESHEXPO.NS", "FSL.NS"
]

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🚀 Enterprise AI Researcher", 
    "📊 Recommendations & Portfolio", 
    "🛡️ Behavioral & Risk Guard",
    "📰 Audit Newsletter",
    "🔍 AI Payload Inspector",
    "🛠️ Diagnostics",
    "🧠 AI Learnings",
    "🗄️ DB Management"
])

with tab1:
    st.header("Institutional Quantitative Scan & 150-Stock Full Scale Universe")

    col_conf, col_check = st.columns(2)
    with col_conf:
        user_confidence = st.slider("Pre-Trade Confidence Score (1-10)", min_value=1, max_value=10, value=7)
    with col_check:
        checklist_passed = st.checkbox("Pre-Trade Rules Verified (RSI > 50, Volume > Avg, Trend Aligned)", value=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_original_scan = st.button("🔍 Run Full 150-Stock Scan & 2-Step Competitor Loop", type="primary", use_container_width=True)
    with btn_col2:
        run_multi_strategy_scan = st.button("🌍 Run 10-Strategy Consensus Engine", type="secondary", use_container_width=True)

    # --- BUTTON 1: ORIGINAL 2-STEP SCAN ---
    if run_original_scan:
        logging.info("Starting Original 2-Step Scan execution.")
        openrouter_key = st.secrets.get("OPENROUTER_KEY", get_db_config("openrouter_key", ""))
        if not openrouter_key:
            st.error("Please provide your OpenRouter API Key in Streamlit Secrets or Sidebar.")
        else:
            status_text = st.empty()
            eta_text = st.empty()
            progress_bar = st.progress(0)
            start_time = time.time()
            
            try:
                status_text.markdown("📰 **Step 1/5:** Checking 6-hour window policy for financial news & super investors...")
                progress_bar.progress(10)
                
                news, _ = fetch_hybrid_news()
                super_investor_data = fetch_top_15_indian_investor_portfolios()
                
                status_text.markdown("📊 **Step 2/5:** Fetching technicals & fundamentals for ALL 150 Stocks...")
                progress_bar.progress(30)
                
                full_market_data = {
                    "Large-Cap": [get_cached_or_live_metrics(t) for t in LARGE_CAP_UNIVERSE],
                    "Mid-Cap": [get_cached_or_live_metrics(t) for t in MID_CAP_UNIVERSE],
                    "Small-Cap": [get_cached_or_live_metrics(t) for t in SMALL_CAP_UNIVERSE]
                }
                st.session_state.debug_logs["last_market_data"] = full_market_data

                conn = get_db_connection()
                portfolio_holdings = pd.read_sql_query("SELECT * FROM portfolio", conn).to_dict(orient="records")
                active_recommendations = pd.read_sql_query("SELECT * FROM recommendations WHERE status = 'Active'", conn).to_dict(orient="records")
                conn.close()

                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
                learnings_content = get_db_learnings()
                macro = fetch_macro_indicators()

                status_text.markdown("🤖 **Step 3/5:** Agentic Call 1: Screening full 150-stock universe & Super Investor Portfolios...")
                progress_bar.progress(60)
                
                clean_payload_step1 = {
                    "initial_user_context": user_initial_context,
                    "top_15_indian_super_investors": super_investor_data,
                    "current_active_recommendations": active_recommendations,
                    "current_portfolio_holdings": portfolio_holdings,
                    "macro_context": {"current_date": str(datetime.date.today()), "macro_indicators": macro},
                    "learnings_memory": learnings_content,
                    "news_headlines": news[:120],
                    "full_market_universe_150_stocks": full_market_data
                }
                st.session_state.debug_logs["step1_payload"] = clean_payload_step1

                prompt_pass1 = f"""
                You are an elite Lead Quantitative Financial System Engineer. Analyze the clean data payload below containing 150 stocks, active recommendations, holdings, and Top 15 Indian Super Investor Portfolios.
                
                ### Clean Step 1 Payload:
                {json.dumps(clean_payload_step1, indent=2)}
                
                Select top candidates across Large, Mid, and Small caps for Short-Term (5 picks), Medium-Term (5 picks), and Long-Term (5 picks) horizons (Total 15 picks).
                Return a JSON block containing the tickers selected under key 'initial_picks':
                ```json
                {{
                  "initial_picks": ["TCS.NS", "TRENT.NS", "KPITTECH.NS", "RELIANCE.NS", "BEL.NS", "INFY.NS", "SBIN.NS", "ITC.NS", "AXISBANK.NS", "LT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ASIANPAINT.NS", "ZOMATO.NS"]
                }}
                ```
                """
                
                res1 = client.chat.completions.create(model="openrouter/free", messages=[{"role": "user", "content": prompt_pass1}], temperature=0.3)
                
                initial_picks = []
                try:
                    match1 = re.search(r'```json\s*(\{.*?\})\s*```', res1.choices[0].message.content, re.DOTALL)
                    if match1: initial_picks = json.loads(match1.group(1)).get("initial_picks", [])
                except Exception:
                    initial_picks = ["TCS.NS", "RELIANCE.NS", "TRENT.NS", "KPITTECH.NS", "BEL.NS"]

                progress_bar.progress(75)
                status_text.markdown("🔍 **Step 4/5:** Fetching competitor benchmarking data...")

                competitor_mapping = {
                    "TCS.NS": ["INFY.NS", "WIPRO.NS"],
                    "RELIANCE.NS": ["BHARTIARTL.NS", "ONGC.NS"],
                    "TRENT.NS": ["ABFRL.NS", "DMART.NS"],
                    "KPITTECH.NS": ["CYIENT.NS", "BSOFT.NS"]
                }
                competitor_data = {pick: [get_cached_or_live_metrics(c) for c in competitor_mapping.get(pick, ["HDFCBANK.NS", "SBIN.NS"])] for pick in initial_picks}

                status_text.markdown("🤖 **Step 5/5:** Agentic Call 2: Re-evaluating against competitors...")
                progress_bar.progress(90)

                clean_payload_step2 = {
                    "initial_picks": initial_picks,
                    "top_15_indian_super_investors": super_investor_data,
                    "active_recommendations_to_compare": active_recommendations,
                    "competitor_benchmarking_data": competitor_data,
                    "news_headlines": news[:120]
                }
                st.session_state.debug_logs["step2_payload"] = clean_payload_step2

                prompt_pass2 = f"""
                Conduct Secondary Competitor Re-evaluation Pass and Parameter Change Audit.
                Payload: {json.dumps(clean_payload_step2, indent=2)}
                Provide exactly 5 Short-Term, 5 Medium-Term, and 5 Long-Term recommendations.
                Include strict fenced JSON at the end under ```json containing 'recommendations' array.
                ```json
                {{
                  "recommendations": [
                    {{"ticker": "TCS.NS", "cap": "Large-Cap", "horizon": "Medium-Term (6 Months)", "entry": 3800, "target": 4200, "stop_loss": 3650, "reason": "Beats competitor margins.", "change_notes": "Target raised."}}
                  ]
                }}
                ```
                """
                
                res2 = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": prompt_pass1}, {"role": "assistant", "content": res1.choices[0].message.content}, {"role": "user", "content": prompt_pass2}],
                    temperature=0.3
                )
                final_report = res2.choices[0].message.content
                progress_bar.progress(100)
                
                try:
                    match2 = re.search(r'```json\s*(\{.*?\})\s*```', final_report, re.DOTALL)
                    if match2:
                        parsed = json.loads(match2.group(1))
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        today_str = str(datetime.date.today())
                        
                        for r in parsed.get("recommendations", []):
                            ticker = str(r.get("ticker", "UNKNOWN"))
                            horizon = str(r.get("horizon", "Medium-Term"))
                            entry = float(r.get("entry", 0) or 0)
                            target = float(r.get("target", 0) or 0)
                            stop_loss = float(r.get("stop_loss", 0) or 0)
                            reason = str(r.get("reason", "AI Pick") or "AI Pick")
                            change_notes = str(r.get("change_notes", "") or "")
                            cap = str(r.get("cap", "Large-Cap") or "Large-Cap")
                            
                            cursor.execute("SELECT id FROM recommendations WHERE ticker = ? AND status = 'Active'", (ticker,))
                            existing = cursor.fetchone()
                            
                            if existing:
                                cursor.execute("UPDATE recommendations SET date = ?, horizon = ?, entry = ?, target = ?, stop_loss = ?, reason = ?, change_notes = ? WHERE id = ?", (today_str, horizon, entry, target, stop_loss, reason, change_notes, existing[0]))
                            else:
                                cursor.execute("INSERT INTO recommendations (date, ticker, cap, horizon, entry, target, stop_loss, confidence_score, checklist_passed, reason, original_date, change_notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (today_str, ticker, cap, horizon, entry, target, stop_loss, user_confidence, checklist_passed, reason, today_str, change_notes))
                        conn.commit()
                        conn.close()
                        st.success("✨ Recommendations successfully synchronized to SQLite Cloud!")
                except Exception as parse_err:
                    st.warning(f"JSON auto-sync warning: {parse_err}")

                today_str = str(datetime.date.today())
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("INSERT OR REPLACE INTO audit_reports (date_str, report_markdown) VALUES (?, ?)", (today_str, final_report))
                    conn.commit()
                    conn.close()
                except Exception:
                    pass

                st.markdown("---")
                st.markdown(final_report)
                logging.info("Original scan completed successfully.")
            except Exception as e:
                logging.error(f"Error in original scan: {e}")
                st.error(f"Error during execution: {e}")

    # --- BUTTON 2: 10-STRATEGY CONSENSUS ENGINE ---
    if run_multi_strategy_scan:
        logging.info("Starting Multi-Strategy Consensus Engine execution.")
        openrouter_key = st.secrets.get("OPENROUTER_KEY", get_db_config("openrouter_key", ""))
        if not openrouter_key:
            st.error("Please enter OpenRouter API Key.")
        else:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                status_text.markdown("📰 **Step 1/3:** Verifying 6-hour news cache & pulling Top 15 Indian Super Investor Portfolios...")
                progress_bar.progress(15)
                
                news, _ = fetch_hybrid_news()
                super_investor_data = fetch_top_15_indian_investor_portfolios()
                
                status_text.markdown("📊 **Step 2/3:** Loading 150-stock technical & fundamental cached dataset...")
                progress_bar.progress(40)
                
                full_market_data = {
                    "Large-Cap": [get_cached_or_live_metrics(t) for t in LARGE_CAP_UNIVERSE],
                    "Mid-Cap": [get_cached_or_live_metrics(t) for t in MID_CAP_UNIVERSE],
                    "Small-Cap": [get_cached_or_live_metrics(t) for t in SMALL_CAP_UNIVERSE]
                }

                status_text.markdown("🤖 **Step 3/3:** Running 10 Global Quantitative Strategies in parallel via AI Consensus Engine...")
                progress_bar.progress(70)

                global_strategies_prompt = f"""
                You are a Master Quantitative System running 10 Global Investment Strategies simultaneously on a 150-stock universe.
                
                ### Market Data Payload:
                {json.dumps(full_market_data, indent=2)}
                
                ### Top 15 Indian Superstar Investors Data:
                {json.dumps(super_investor_data, indent=2)}
                
                ### Macro Context & News:
                {json.dumps(fetch_macro_indicators())}
                Headlines: {json.dumps(news[:50])}
                
                ### MANDATORY TASK:
                Evaluate ALL 150 stocks against EACH of the following 10 Strategies. Select top 5-7 stocks per strategy:
                1. Warren Buffett Value & Moat
                2. Peter Lynch GARP
                3. Benjamin Graham Deep Value
                4. William O'Neil CAN SLIM
                5. Joel Greenblatt Magic Formula
                6. RSI & MACD Momentum
                7. Ray Dalio Macro Hedged
                8. Dividend Yield & Capital Protection
                9. Small-Cap Catalyst Breakout
                10. Super Investor Smart Money Tracking
                
                Return ONLY a strict valid JSON block under key 'strategy_results':
                ```json
                {{
                  "strategy_results": [
                    {{"strategy_name": "Warren Buffett Value & Moat", "picks": ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"]}},
                    {{"strategy_name": "Peter Lynch GARP", "picks": ["TRENT.NS", "BEL.NS", "KPITTECH.NS", "TCS.NS", "POLYCAB.NS"]}},
                    {{"strategy_name": "Benjamin Graham Deep Value", "picks": ["SBIN.NS", "COALINDIA.NS", "ONGC.NS", "BPCL.NS", "TCS.NS"]}},
                    {{"strategy_name": "William O'Neil CAN SLIM", "picks": ["TRENT.NS", "DIXON.NS", "BEL.NS", "KPITTECH.NS", "ZOMATO.NS"]}},
                    {{"strategy_name": "Joel Greenblatt Magic Formula", "picks": ["TCS.NS", "COALINDIA.NS", "LTIM.NS", "HCLTECH.NS", "BEL.NS"]}},
                    {{"strategy_name": "RSI & MACD Momentum", "picks": ["TRENT.NS", "BHARTIARTL.NS", "BEL.NS", "DIXON.NS", "PERSISTENT.NS"]}},
                    {{"strategy_name": "Ray Dalio Macro Hedged", "picks": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ITC.NS", "SUNPHARMA.NS"]}},
                    {{"strategy_name": "Dividend Yield & Capital Protection", "picks": ["ITC.NS", "COALINDIA.NS", "POWERGRID.NS", "NTPC.NS", "HINDUNILVR.NS"]}},
                    {{"strategy_name": "Small-Cap Catalyst Breakout", "picks": ["KPITTECH.NS", "CYIENT.NS", "BSOFT.NS", "BHEL.NS", "CEATLTD.NS"]}},
                    {{"strategy_name": "Super Investor Smart Money Tracking", "picks": ["DMART.NS", "TITAN.NS", "TRENT.NS", "ELECON.NS", "RAYMOND.NS"]}}
                  ]
                }}
                ```
                """
                
                client = OpenAI(base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)", api_key=openrouter_key)
                c_res = client.chat.completions.create(model="openrouter/free", messages=[{"role": "user", "content": global_strategies_prompt}], temperature=0.2, timeout=60)
                raw_response = c_res.choices[0].message.content
                
                progress_bar.progress(100)
                status_text.markdown("✅ **Consensus calculation complete! Computing strategy intersections...**")
                
                match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
                if match:
                    strategy_data = json.loads(match.group(1)).get("strategy_results", [])
                    st.session_state.debug_logs["consensus_payload"] = strategy_data
                    
                    ticker_counts, ticker_strategies = {}, {}
                    for strat in strategy_data:
                        s_name = strat["strategy_name"]
                        for t in strat["picks"]:
                            ticker_counts[t] = ticker_counts.get(t, 0) + 1
                            if t not in ticker_strategies: ticker_strategies[t] = []
                            ticker_strategies[t].append(s_name)
                    
                    ranked_list = [{"Ticker": ticker, "Strategy Match Count": f"{count} / 10", "Match Score (%)": f"{int((count/10.0)*100)}%", "Matched Strategies": ", ".join(ticker_strategies[ticker])} for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)]
                    df_consensus = pd.DataFrame(ranked_list)
                    
                    st.subheader("🏆 Ranked Multi-Strategy Consensus Leaderboard")
                    st.dataframe(df_consensus, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Individual Strategy Breakdown (10 World-Class Models)")
                    for strat in strategy_data:
                        with st.expander(f"📌 {strat['strategy_name']} ({len(strat['picks'])} Picks)"):
                            st.write(f"**Recommended Tickers:** `{', '.join(strat['picks'])}`")
                else:
                    st.markdown(raw_response)
                logging.info("Consensus Engine completed successfully.")
            except Exception as e:
                logging.error(f"Error executing Multi-Strategy Consensus Engine: {e}")
                st.error(f"Connection error or timeout executing Consensus Engine: {e}. Please check Streamlit Cloud runtime logs for details.")

with tab2:
    st.header("📊 Recommendations Ledger & Portfolio Analyzer")
    
    if st.button("🔄 Check Live Target & Stop-Loss Hits for Active Recommendations"):
        with st.spinner("Fetching live prices and auditing active trades..."):
            conn = get_db_connection()
            cursor = conn.cursor()
            active_recs = cursor.execute("SELECT id, ticker, target, stop_loss FROM recommendations WHERE status = 'Active'").fetchall()
            
            updates_count = 0
            for rec in active_recs:
                rec_id, ticker, target, stop_loss = rec[0], rec[1], rec[2], rec[3]
                metrics = get_cached_or_live_metrics(ticker)
                curr_price = metrics.get("fundamentals", {}).get("current_price", 0)
                
                if curr_price > 0 and target > 0 and stop_loss > 0:
                    new_outcome = "Win (+)" if curr_price >= target else ("Loss (-)" if curr_price <= stop_loss else None)
                    if new_outcome:
                        cursor.execute("UPDATE recommendations SET outcome = ?, status = 'Closed' WHERE id = ?", (new_outcome, rec_id))
                        updates_count += 1
            conn.commit()
            conn.close()
            st.success(f"Audit complete! Updated {updates_count} trades based on live price action.")
            st.rerun()

    st.markdown("---")
    st.subheader("🔍 AI Portfolio Analyzer (Hold / Buy / Sell)")
    with st.form("portfolio_analyzer_form"):
        user_holdings_input = st.text_area("Enter Holdings (Ticker, Quantity, Buy Price per line, e.g., TCS.NS, 10, 3500):", value="TCS.NS, 10, 3500\nRELIANCE.NS, 5, 2400")
        analyze_portfolio_btn = st.form_submit_button("Run Portfolio Health Check")
        
    if analyze_portfolio_btn:
        with st.spinner("Analyzing portfolio risk and performance..."):
            lines = user_holdings_input.split('\n')
            holdings_eval_data = []
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolio")
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    ticker, qty, buy_price = parts[0].upper(), int(parts[1]), float(parts[2])
                    cursor.execute("INSERT OR REPLACE INTO portfolio (ticker, qty, buy_price) VALUES (?, ?, ?)", (ticker, qty, buy_price))
                    metrics = get_cached_or_live_metrics(ticker)
                    current_p = metrics.get("fundamentals", {}).get("current_price", buy_price)
                    pnl_pct = round(((current_p - buy_price) / buy_price) * 100, 2) if buy_price else 0
                    holdings_eval_data.append({"ticker": ticker, "qty": qty, "buy_price": buy_price, "current_price": current_p, "pnl_pct": pnl_pct, "metrics": metrics})
            conn.commit()
            conn.close()
            
            client = OpenAI(base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)", api_key=st.secrets.get("OPENROUTER_KEY", get_db_config("openrouter_key", "")))
            p_res = client.chat.completions.create(model="openrouter/free", messages=[{"role": "user", "content": f"Analyze holdings and provide verdicts:\n{json.dumps(holdings_eval_data, indent=2)}"}], temperature=0.3)
            st.markdown("### Portfolio Diagnostic Report")
            st.markdown(p_res.choices[0].message.content)

    st.markdown("---")
    st.subheader("Active System Recommendations Ledger (Inline Editable Table)")
    conn = get_db_connection()
    df_recs = pd.read_sql_query("SELECT id, date, ticker, cap, horizon, entry, target, stop_loss, confidence_score, picked, outcome, status, change_notes FROM recommendations ORDER BY id DESC", conn)
    conn.close()
    
    if not df_recs.empty:
        edited_df = st.data_editor(
            df_recs,
            column_config={
                "picked": st.column_config.SelectboxColumn("Picked", options=["No", "Yes"], required=True),
                "outcome": st.column_config.SelectboxColumn("Outcome", options=["Pending", "Win (+)", "Loss (-)"], required=True),
                "status": st.column_config.SelectboxColumn("Status", options=["Active", "Closed"], required=True),
            },
            disabled=["id", "date", "ticker", "cap", "horizon", "entry", "target", "stop_loss", "confidence_score", "change_notes"],
            use_container_width=True,
            key="recommendations_editor"
        )
        
        if st.button("💾 Save Table Changes"):
            conn = get_db_connection()
            cursor = conn.cursor()
            for index, row in edited_df.iterrows():
                cursor.execute("UPDATE recommendations SET picked = ?, outcome = ?, status = ? WHERE id = ?", (row["picked"], row["outcome"], "Closed" if row["outcome"] != "Pending" else "Active", row["id"]))
            conn.commit()
            conn.close()
            st.success("Changes committed to SQLite Cloud database!")
            st.rerun()

        st.markdown("---")
        st.subheader("📈 Aggregated Unique Stock Metrics View")
        conn = get_db_connection()
        agg_df = pd.read_sql_query("SELECT ticker, cap, COUNT(*) AS total_recommendations, MIN(date) AS first_recommended, MAX(date) AS latest_recommended, GROUP_CONCAT(DISTINCT date) AS recommendation_days_list, MAX(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) AS is_currently_active FROM recommendations GROUP BY ticker ORDER BY total_recommendations DESC", conn)
        conn.close()
        st.dataframe(agg_df, use_container_width=True)
    else:
        st.info("No recommendations synced yet.")

with tab3:
    st.header("🛡️ Behavioral Pattern Detector & Safety Control")
    conn = get_db_connection()
    conf_df = pd.read_sql_query("SELECT confidence_score FROM recommendations", conn)
    conn.close()
    if not conf_df.empty:
        st.metric("Average Pre-Trade Confidence Score", round(conf_df["confidence_score"].mean(), 2))
    else:
        st.write("Awaiting trade tracking data.")

with tab4:
    st.header("📰 Daily Audit Newsletter & Reports Manager (Cloud Stored)")
    try:
        conn = get_db_connection()
        audit_df = pd.read_sql_query("SELECT date_str FROM audit_reports ORDER BY date_str DESC", conn)
        conn.close()
        
        if not audit_df.empty:
            audit_dates = audit_df["date_str"].tolist()
            col_sel, col_del = st.columns([3, 1])
            with col_sel:
                selected_date = st.selectbox("Select Audit Report Date", audit_dates)
            with col_del:
                st.write(" ")
                st.write(" ")
                if st.button("🗑️ Delete Report", type="secondary"):
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM audit_reports WHERE date_str = ?", (selected_date,))
                    conn.commit()
                    conn.close()
                    st.success(f"Deleted report for {selected_date}!")
                    st.rerun()
                    
            st.markdown("---")
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT report_markdown FROM audit_reports WHERE date_str = ?", (selected_date,))
            rep_row = cursor.fetchone()
            conn.close()
            if rep_row:
                st.markdown(rep_row[0])
        else:
            st.info("No audit reports saved in cloud database yet.")
    except Exception as e:
        st.info(f"Awaiting audit logs storage: {e}")

with tab5:
    st.header("🔍 AI Payload Inspector")
    if debug_mode:
        st.subheader("Step 1 Payload")
        st.json(st.session_state.debug_logs.get("step1_payload", {}))
        st.markdown("---")
        st.subheader("Step 2 Payload")
        st.json(st.session_state.debug_logs.get("step2_payload", {}))
        st.markdown("---")
        st.subheader("Consensus Engine Payload")
        st.json(st.session_state.debug_logs.get("consensus_payload", {}))
    else:
        st.info("⚠️ Debug Mode is turned OFF.")

with tab6:
    st.header("🛠️ Real-Time Diagnostics & Scraper Logs")
    if debug_mode:
        if st.button("Run Live Diagnostics Test Now"):
            _, test_logs = fetch_hybrid_news()
            st.dataframe(pd.DataFrame(test_logs), use_container_width=True)
        if st.session_state.debug_logs["scraper_status"]:
            st.subheader("Last Scraper Execution Status")
            st.dataframe(pd.DataFrame(st.session_state.debug_logs["scraper_status"]), use_container_width=True)
    else:
        st.info("⚠️ Debug Mode is turned OFF.")

with tab7:
    st.header("🧠 AI Self-Reflection & Learnings (Cloud Stored)")
    if st.button("✨ Auto-Generate & Update Learnings via AI", type="primary"):
        openrouter_key = st.secrets.get("OPENROUTER_KEY", get_db_config("openrouter_key", ""))
        if not openrouter_key:
            st.error("Please enter OpenRouter API Key.")
        else:
            with st.spinner("AI is analyzing trade outcomes..."):
                conn = get_db_connection()
                all_recs = pd.read_sql_query("SELECT * FROM recommendations", conn).to_dict(orient="records")
                conn.close()
                current_learnings = get_db_learnings()
                summarize_prompt = f"Review current memory and recommendations, and formulate updated rules:\nCurrent Memory:\n{current_learnings}\nTrade History:\n{json.dumps(all_recs, indent=2)}"
                client = OpenAI(base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)", api_key=openrouter_key)
                sum_res = client.chat.completions.create(model="openrouter/free", messages=[{"role": "user", "content": summarize_prompt}], temperature=0.2)
                set_db_learnings(sum_res.choices[0].message.content)
                st.success("✨ Cloud learnings memory successfully updated!")
                st.rerun()

    st.markdown("---")
    learnings_text = get_db_learnings()
    updated_learnings = st.text_area("Edit Learnings & Rules for AI Memory", value=learnings_text, height=300)
    if st.button("Save Learnings Manually"):
        set_db_learnings(updated_learnings)
        st.success("Learnings saved to SQLite Cloud!")

with tab8:
    st.header("🗄️ Database Management & Table Truncation Control")
    conn = get_db_connection()
    cursor = conn.cursor()
    tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
    conn.close()

    table_metadata = {
        "recommendations": {"description": "Stores AI recommendations, targets, stop-losses, outcomes.", "usage": "Tab 1, Tab 2, Tab 3, Tab 7."},
        "portfolio": {"description": "Stores current stock holdings.", "usage": "Tab 1, Tab 2."},
        "market_cache": {"description": "Caches ticker fundamentals & technicals with 6-hr TTL.", "usage": "All metric fetches."},
        "news_cache": {"description": "Caches hybrid news corpus with 6-hr TTL.", "usage": "Tab 1 news aggregation."},
        "system_memory": {"description": "Stores persistent AI self-reflection memory (learnings).", "usage": "Tab 7."},
        "app_config": {"description": "Stores app configuration & keys.", "usage": "Sidebar settings."},
        "audit_reports": {"description": "Stores markdown audit reports permanently.", "usage": "Tab 4."}
    }

    for tbl in tables:
        with st.expander(f"📁 Table: `{tbl}`"):
            meta = table_metadata.get(tbl, {"description": "Custom table.", "usage": "Internal."})
            st.markdown(f"**📖 Description:** {meta['description']}")
            st.markdown(f"**⚙️ Usage:** {meta['usage']}")
            
            conn = get_db_connection()
            count_df = pd.read_sql_query(f"SELECT COUNT(*) AS total_rows FROM {tbl}", conn)
            preview_df = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 5", conn)
            conn.close()
            
            total_rows = count_df["total_rows"].iloc[0]
            st.metric("Total Records", total_rows)
            if total_rows > 0:
                st.dataframe(preview_df, use_container_width=True)
            
            confirm_chk = st.checkbox(f"I want to clear/truncate table `{tbl}`", key=f"confirm_trunc_{tbl}")
            if st.button(f"🗑️ Truncate Table `{tbl}`", key=f"btn_trunc_{tbl}", type="secondary"):
                if confirm_chk:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(f"DELETE FROM {tbl}")
                    conn.commit()
                    conn.close()
                    st.success(f"Table `{tbl}` truncated successfully!")
                    st.rerun()
                else:
                    st.warning("Please check confirmation box first.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: grey; font-size: 13px;'>"
    f"📈 AI Stock Researcher & Quantitative Lab | <b>Version:</b> {APP_VERSION} | <b>Last Build:</b> {LAST_UPDATED}"
    f"</div>",
    unsafe_allow_html=True
)
