import streamlit as st
import json
import os
import datetime
import time
import re
import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import sqlitecloud

# --- APP VERSION CONFIG ---
APP_VERSION = "v19.0.0-native-indicators"
LAST_UPDATED = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# --- STORAGE PATHS ---
STORAGE_DIR = '.'
LEARNINGS_FILE = os.path.join(STORAGE_DIR, 'learnings.md')
CONFIG_FILE = os.path.join(STORAGE_DIR, 'config.json')
AUDIT_DIR = os.path.join(STORAGE_DIR, 'audit_logs')

os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

st.set_page_config(page_title="AI Stock Researcher & Quantitative Lab", layout="wide")
st.title("📈 AI Stock Researcher & Quantitative Lab [Streamlit Cloud Edition]")

# --- NATIVE TECHNICAL INDICATORS (NO PANDAS-TA DEPENDENCY) ---
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
    return macd_line

# --- DB CONNECTION HELPER (SQLITE CLOUD FOR 24/7 PERSISTENCE) ---
def get_db_connection():
    db_url = st.secrets.get("SQLITE_CLOUD_URL", "")
    if not db_url:
        st.error("Missing `SQLITE_CLOUD_URL` in Streamlit Secrets!")
        st.stop()
    conn = sqlitecloud.connect(db_url)
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            cap TEXT,
            horizon TEXT,
            entry REAL,
            target REAL,
            stop_loss REAL,
            confidence_score INTEGER,
            checklist_passed BOOLEAN,
            reason TEXT,
            picked TEXT DEFAULT 'No',
            outcome TEXT DEFAULT 'Pending',
            status TEXT DEFAULT 'Active',
            original_date TEXT,
            change_notes TEXT DEFAULT ''
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE,
            qty INTEGER,
            buy_price REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_cache (
            ticker TEXT PRIMARY KEY,
            timestamp DATETIME,
            payload_json TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT,
            timestamp DATETIME,
            headlines_json TEXT
        )
    ''')
    
    cursor.execute("PRAGMA table_info(recommendations)")
    recs_cols = [col[1] for col in cursor.fetchall()]
    
    missing_columns = {
        "picked": "TEXT DEFAULT 'No'",
        "outcome": "TEXT DEFAULT 'Pending'",
        "status": "TEXT DEFAULT 'Active'",
        "original_date": "TEXT",
        "change_notes": "TEXT DEFAULT ''"
    }
    
    for col_name, col_type in missing_columns.items():
        if col_name not in recs_cols:
            cursor.execute(f"ALTER TABLE recommendations ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()

init_db()

# --- SESSION STATE DEBUGS ---
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = {
        "last_news": [],
        "last_market_data": {},
        "step1_payload": {},
        "step2_payload": {},
        "scraper_status": []
    }

# --- CONFIG HELPERS ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

config = load_config()

# --- SIDEBAR CONTROLS & USER CONTEXT ---
with st.sidebar:
    st.header("🔑 API & Safeguards")
    default_key = st.secrets.get("OPENROUTER_KEY", config.get("openrouter_key", ""))
    input_key = st.text_input("OpenRouter API Key", value=default_key, type="password")
    if input_key != config.get("openrouter_key", ""):
        config["openrouter_key"] = input_key
        save_config(config)
        st.success("API Key updated for session!")
        
    st.markdown("---")
    st.header("📝 Initial User Context Input")
    user_initial_context = st.text_area(
        "Provide initial context/guidelines for the AI before research scan:", 
        value="Focus heavily on capital protection, high dividend yield stability, macro DXY correlation, and breakout setups with tight risk-reward profiles."
    )

    st.markdown("---")
    st.header("🛠️ Debug Controls")
    debug_mode = st.toggle("🐛 Enable Debug Mode", value=True, help="Toggle granular diagnostics on or off")
    
    st.markdown("---")
    st.markdown(f"**Version:** `{APP_VERSION}`")

# --- DATA ENGINEERING: 6-HOUR WINDOW & 1-DAY PRUNING ---
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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT payload_json, timestamp FROM market_cache WHERE ticker = ?", (ticker_symbol,))
    row = cursor.fetchone()
    
    if row and row[1]:
        try:
            cached_time = datetime.datetime.strptime(str(row[1]), "%Y-%m-%d %H:%M:%S")
            hours_elapsed = (datetime.datetime.now() - cached_time).total_seconds() / 3600.0
            if hours_elapsed < 6.0:
                conn.close()
                return json.loads(row[0])
        except Exception:
            pass
            
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
            latest_rsi = rsi_series.iloc[-1]
            latest_macd = macd_series.iloc[-1]
            
            technicals = {
                "rsi_14": round(float(latest_rsi), 2) if not np.isnan(latest_rsi) else 0,
                "macd": round(float(latest_macd), 2) if not np.isnan(latest_macd) else 0
            }
            
        payload = {"ticker": ticker_symbol, "status": "SUCCESS", "fundamentals": fundamentals, "technicals": technicals}
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("INSERT OR REPLACE INTO market_cache (ticker, timestamp, payload_json) VALUES (?, ?, ?)",
                       (ticker_symbol, now_str, json.dumps(payload)))
        conn.commit()
        conn.close()
        return payload
    except Exception as e:
        conn.close()
        return {"ticker": ticker_symbol, "status": f"ERROR: {str(e)}", "fundamentals": {}, "technicals": {}}

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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    visited = set()
    articles = []
    queue = [(base_url, 0)]
    visited.add(base_url)
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    while queue and len(articles) < 50:
        current_url, depth = queue.pop(0)
        try:
            res = requests.get(current_url, headers=headers, timeout=5)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, 'html.parser')
            for item in soup.find_all(['h3', 'h2', 'h1', 'a'], limit=30):
                txt = item.get_text().strip()
                if len(txt) > 25 and txt not in articles:
                    articles.append(txt)
            
            if depth < max_depth:
                for link in soup.find_all('a', href=True):
                    abs_url = urljoin(current_url, link['href'])
                    parsed_link = urlparse(abs_url)
                    if parsed_link.netloc == base_domain and abs_url not in visited:
                        if any(k in abs_url.lower() for k in ['market', 'stock', 'news', 'business', 'economy', 'company', 'share']):
                            visited.add(abs_url)
                            queue.append((abs_url, depth + 1))
                            if len(visited) > 30:
                                break
        except Exception:
            pass
    return articles

# --- HYBRID NEWS CORPUS WITH 6-HOUR WINDOW ---
def fetch_hybrid_news():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT headlines_json, timestamp FROM news_cache WHERE source_name = 'MASTER_CORPUS'")
    row = cursor.fetchone()
    
    if row and row[1]:
        try:
            cached_time = datetime.datetime.strptime(str(row[1]), "%Y-%m-%d %H:%M:%S")
            hours_elapsed = (datetime.datetime.now() - cached_time).total_seconds() / 3600.0
            if hours_elapsed < 6.0:
                conn.close()
                return json.loads(row[0]), [{"source": "SQLite Cache", "status": "HIT (Within 6-Hour Window)", "items": len(json.loads(row[0]))}]
        except Exception:
            pass

    rss_feeds = [
        {"name": "Economic Times", "url": "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms"},
        {"name": "Livemint", "url": "https://www.livemint.com/rss/markets"},
        {"name": "Business Standard", "url": "https://www.business-standard.com/rss/markets-106.rss"},
        {"name": "Financial Express", "url": "https://www.financialexpress.com/feed/"}
    ]
    seed_crawls = [
        {"name": "Economic Times", "url": "https://economictimes.indiatimes.com/markets/stocks/news"},
        {"name": "Business Standard", "url": "https://www.business-standard.com/markets/news"}
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    master_headlines = []
    debug_logs = []

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
                log["status"] = "SUCCESS (Fresh Fetch)"
                log["items"] = count
            else:
                log["status"] = f"HTTP {res.status_code}"
        except Exception as e:
            log["status"] = f"ERROR: {e}"
        debug_logs.append(log)

    for seed in seed_crawls:
        log = {"source": seed["name"], "type": "Crawler", "url": seed["url"], "status": "PENDING", "items": 0}
        try:
            crawled = crawl_site_recursive(seed["url"], max_depth=2)
            count = 0
            for art in crawled:
                item_str = f"[{seed['name']}] {art}"
                if item_str not in master_headlines:
                    master_headlines.append(item_str)
                    count += 1
            log["status"] = "SUCCESS (Fresh Fetch)"
            log["items"] = count
        except Exception as e:
            log["status"] = f"ERROR: {e}"
        debug_logs.append(log)

    final_headlines = master_headlines[:150]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("INSERT OR REPLACE INTO news_cache (source_name, timestamp, headlines_json) VALUES ('MASTER_CORPUS', ?, ?)",
                   (now_str, json.dumps(final_headlines)))
    conn.commit()
    conn.close()

    st.session_state.debug_logs["scraper_status"] = debug_logs
    st.session_state.debug_logs["last_news"] = final_headlines
    return final_headlines, debug_logs

# --- FULL 50+50+50 = 150 STOCKS UNIVERSE ---
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

    if st.button("🔍 Run Full 150-Stock Scan & 2-Step Competitor Loop", type="primary"):
        openrouter_key = st.secrets.get("OPENROUTER_KEY", config.get("openrouter_key", ""))
        if not openrouter_key:
            st.error("Please provide your OpenRouter API Key in Streamlit Secrets or Sidebar.")
        else:
            status_text = st.empty()
            eta_text = st.empty()
            progress_bar = st.progress(0)
            start_time = time.time()
            
            try:
                status_text.markdown("📰 **Step 1/5:** Checking 6-hour window policy for financial news corpus...")
                progress_bar.progress(10)
                eta_text.text("Estimated time remaining: ~45 seconds (Processing 150 Stocks)")
                
                news, scraper_debugs = fetch_hybrid_news()
                
                status_text.markdown("📊 **Step 2/5:** Fetching technicals & fundamentals for ALL 150 Stocks (50 Large, 50 Mid, 50 Small)...")
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
                
                learnings_content = load_config().get("learnings", "# AI Quantitative Learnings\nNo past notes.")
                if os.path.exists(LEARNINGS_FILE):
                    with open(LEARNINGS_FILE, 'r') as f:
                        learnings_content = f.read()

                macro = fetch_macro_indicators()
                macro_context = {
                    "current_date": str(datetime.date.today()),
                    "macro_indicators": macro,
                    "macro_factors": "Indian markets tracking Q-end corporate earnings, RBI monetary policy stance, and FII/DII flows."
                }

                status_text.markdown("🤖 **Step 3/5:** Agentic Call 1: Screening full 150-stock universe & active recommendations...")
                progress_bar.progress(60)
                
                clean_payload_step1 = {
                    "initial_user_context": user_initial_context,
                    "current_active_recommendations": active_recommendations,
                    "current_portfolio_holdings": portfolio_holdings,
                    "macro_context": macro_context,
                    "learnings_memory": learnings_content,
                    "news_headlines": news[:120],
                    "full_market_universe_150_stocks": full_market_data
                }
                st.session_state.debug_logs["step1_payload"] = clean_payload_step1

                prompt_pass1 = f"""
                You are an elite Lead Quantitative Financial System Engineer. Analyze the clean data payload below containing 150 stocks, active recommendations, and holdings.
                
                ### Clean Step 1 Payload:
                {json.dumps(clean_payload_step1, indent=2)}
                
                ### Task:
                Select top candidates across Large, Mid, and Small caps for Short-Term (5 picks), Medium-Term (5 picks), and Long-Term (5 picks) horizons (Total 15 picks). If any stock here was previously active in `current_active_recommendations`, evaluate whether its entry, target, or stop-loss needs updating based on latest market conditions.
                Return a JSON block containing the tickers selected under key 'initial_picks'.
                ```json
                {{
                  "initial_picks": ["TCS.NS", "TRENT.NS", "KPITTECH.NS", "RELIANCE.NS", "BEL.NS", "INFY.NS", "SBIN.NS", "ITC.NS", "AXISBANK.NS", "LT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ASIANPAINT.NS", "ZOMATO.NS"]
                }}
                ```
                """
                
                res1 = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": prompt_pass1}],
                    temperature=0.3
                )
                
                initial_picks = []
                try:
                    match1 = re.search(r'```json\s*(\{.*?\})\s*```', res1.choices[0].message.content, re.DOTALL)
                    if match1:
                        initial_picks = json.loads(match1.group(1)).get("initial_picks", [])
                except Exception:
                    initial_picks = ["TCS.NS", "RELIANCE.NS", "TRENT.NS", "KPITTECH.NS", "BEL.NS"]

                progress_bar.progress(75)
                status_text.markdown(f"🔍 **Step 4/5:** Initial picks found: `{initial_picks}`. Fetching competitor benchmarking data...")

                competitor_mapping = {
                    "TCS.NS": ["INFY.NS", "WIPRO.NS"],
                    "RELIANCE.NS": ["BHARTIARTL.NS", "ONGC.NS"],
                    "TRENT.NS": ["ABFRL.NS", "DMART.NS"],
                    "KPITTECH.NS": ["CYIENT.NS", "BSOFT.NS"]
                }
                
                competitor_data = {}
                for pick in initial_picks:
                    comps = competitor_mapping.get(pick, ["HDFCBANK.NS", "SBIN.NS"])
                    competitor_data[pick] = [get_cached_or_live_metrics(c) for c in comps]

                status_text.markdown("🤖 **Step 5/5:** Agentic Call 2: Re-evaluating against active recommendations & competitors, explaining any parameter shifts...")
                progress_bar.progress(90)

                clean_payload_step2 = {
                    "macro_context": macro_context,
                    "initial_picks": initial_picks,
                    "active_recommendations_to_compare": active_recommendations,
                    "competitor_benchmarking_data": competitor_data,
                    "news_headlines": news[:120]
                }
                st.session_state.debug_logs["step2_payload"] = clean_payload_step2

                prompt_pass2 = f"""
                You are conducting the Secondary Competitor Re-evaluation Pass and Parameter Change Audit.
                
                ### Clean Step 2 Payload:
                {json.dumps(clean_payload_step2, indent=2)}
                
                ### MANDATORY INSTRUCTIONS:
                1. Re-evaluate initial picks against competitors and active recommendations.
                2. **Parameter Change Check:** If a recommended stock already exists in `active_recommendations_to_compare` and its entry, target, or stop loss has changed, explicitly explain WHY it changed and what should be kept in the `change_notes` field.
                3. Provide **EXACTLY 5 recommendations for Short-Term, 5 for Medium-Term, and 5 for Long-Term** (total 15 recommendations).
                4. Required columns: `| Ticker | Cap | Horizon (Short-Term / Medium-Term / Long-Term) | Entry (₹) | Target (₹) | Stop-Loss (₹) | Rationale / Parameter Change Notes |`
                5. At the very end of your response, include a strict fenced JSON block under the tag ```json that contains ONLY the recommendations array so the app can parse it automatically:
                ```json
                {{
                  "recommendations": [
                    {{"ticker": "TCS.NS", "cap": "Large-Cap", "horizon": "Medium-Term (6 Months)", "entry": 3800, "target": 4200, "stop_loss": 3650, "reason": "Beats competitor margins.", "change_notes": "Target raised by 5% due to robust Q3 TCV wins."}}
                  ]
                }}
                ```
                """
                
                res2 = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[
                        {"role": "user", "content": prompt_pass1},
                        {"role": "assistant", "content": res1.choices[0].message.content},
                        {"role": "user", "content": prompt_pass2}
                    ],
                    temperature=0.3
                )
                final_report = res2.choices[0].message.content
                
                progress_bar.progress(100)
                elapsed_time = round(time.time() - start_time, 1)
                eta_text.text(f"✅ 2-Step Agentic Scan completed successfully in {elapsed_time} seconds!")
                
                try:
                    match2 = re.search(r'```json\s*(\{.*?\})\s*```', final_report, re.DOTALL)
                    if match2:
                        parsed = json.loads(match2.group(1))
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        today_str = str(datetime.date.today())
                        
                        for r in parsed.get("recommendations", []):
                            ticker = r.get("ticker", "UNKNOWN")
                            horizon = r.get("horizon", "Medium-Term")
                            entry = r.get("entry", 0)
                            target = r.get("target", 0)
                            stop_loss = r.get("stop_loss", 0)
                            reason = r.get("reason", "AI Pick")
                            change_notes = r.get("change_notes", "")
                            
                            cursor.execute("SELECT id FROM recommendations WHERE ticker = ? AND status = 'Active'", (ticker,))
                            existing = cursor.fetchone()
                            
                            if existing:
                                cursor.execute('''
                                    UPDATE recommendations 
                                    SET date = ?, horizon = ?, entry = ?, target = ?, stop_loss = ?, reason = ?, change_notes = ?
                                    WHERE id = ?
                                ''', (today_str, horizon, entry, target, stop_loss, reason, change_notes, existing[0]))
                            else:
                                cursor.execute('''
                                    INSERT INTO recommendations (date, ticker, cap, horizon, entry, target, stop_loss, confidence_score, checklist_passed, reason, original_date, change_notes)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (today_str, ticker, r.get("cap", "Large-Cap"), horizon, entry, target, stop_loss, user_confidence, checklist_passed, reason, today_str, change_notes))
                        conn.commit()
                        conn.close()
                        st.success("✨ Recommendations successfully parsed and synchronized to SQLite Cloud!")
                except Exception as parse_err:
                    st.warning(f"Report generated, but JSON auto-sync encountered an issue ({parse_err}).")

                today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                audit_filename = os.path.join(AUDIT_DIR, f"audit_{today_str}.md")
                with open(audit_filename, 'w') as f:
                    f.write(final_report)
                
                st.markdown("---")
                st.markdown(final_report)
                
            except Exception as e:
                st.error(f"Error during execution: {e}")

with tab2:
    st.header("📊 Recommendations Ledger & Portfolio Analyzer")
    
    if st.button("🔄 Check Live Target & Stop-Loss Hits for Active Recommendations"):
        with st.spinner("Fetching live prices and auditing active trades against target / stop-loss thresholds..."):
            conn = get_db_connection()
            cursor = conn.cursor()
            active_recs = cursor.execute("SELECT id, ticker, target, stop_loss FROM recommendations WHERE status = 'Active'").fetchall()
            
            updates_count = 0
            for rec in active_recs:
                rec_id = rec[0]
                ticker = rec[1]
                target = rec[2]
                stop_loss = rec[3]
                
                metrics = get_cached_or_live_metrics(ticker)
                curr_price = metrics.get("fundamentals", {}).get("current_price", 0)
                
                if curr_price > 0 and target > 0 and stop_loss > 0:
                    new_outcome = None
                    if curr_price >= target:
                        new_outcome = "Win (+)"
                    elif curr_price <= stop_loss:
                        new_outcome = "Loss (-)"
                        
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
        user_holdings_input = st.text_area(
            "Enter Holdings (Ticker, Quantity, Buy Price per line, e.g., TCS.NS, 10, 3500):",
            value="TCS.NS, 10, 3500\nRELIANCE.NS, 5, 2400"
        )
        analyze_portfolio_btn = st.form_submit_button("Run Portfolio Health Check")
        
    if analyze_portfolio_btn:
        with st.spinner("Analyzing portfolio risk, performance, and generating recommendations..."):
            lines = user_holdings_input.split('\n')
            holdings_eval_data = []
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolio")
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    ticker = parts[0].upper()
                    qty = int(parts[1])
                    buy_price = float(parts[2])
                    cursor.execute("INSERT OR REPLACE INTO portfolio (ticker, qty, buy_price) VALUES (?, ?, ?)", (ticker, qty, buy_price))
                    
                    metrics = get_cached_or_live_metrics(ticker)
                    current_p = metrics.get("fundamentals", {}).get("current_price", buy_price)
                    pnl_pct = round(((current_p - buy_price) / buy_price) * 100, 2) if buy_price else 0
                    holdings_eval_data.append({"ticker": ticker, "qty": qty, "buy_price": buy_price, "current_price": current_p, "pnl_pct": pnl_pct, "metrics": metrics})
            conn.commit()
            conn.close()
            
            client = OpenAI(base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)", api_key=st.secrets.get("OPENROUTER_KEY", config.get("openrouter_key", "")))
            portfolio_prompt = f"""
            You are an expert Portfolio Risk Manager. Analyze the user's current holdings below and provide explicit actionable verdicts: HOLD, BUY MORE, or SELL for each item, with clear reasoning and timeframes.
            
            Holdings Data:
            {json.dumps(holdings_eval_data, indent=2)}
            """
            
            p_res = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": portfolio_prompt}],
                temperature=0.3
            )
            st.markdown("### Portfolio Diagnostic Report")
            st.markdown(p_res.choices[0].message.content)

    st.markdown("---")
    st.subheader("Active System Recommendations Ledger (Inline Editable Table)")
    conn = get_db_connection()
    df_recs = pd.read_sql_query("SELECT id, date, ticker, cap, horizon, entry, target, stop_loss, confidence_score, picked, outcome, status, change_notes FROM recommendations ORDER BY id DESC", conn)
    conn.close()
    
    if not df_recs.empty:
        st.markdown("Edit `picked` or `outcome` directly in the table below using dropdowns and click **Save Table Changes**.")
        
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
                cursor.execute("""
                    UPDATE recommendations 
                    SET picked = ?, outcome = ?, status = ? 
                    WHERE id = ?
                """, (row["picked"], row["outcome"], "Closed" if row["outcome"] != "Pending" else "Active", row["id"]))
            conn.commit()
            conn.close()
            st.success("Table changes successfully committed to SQLite Cloud database!")
            st.rerun()

        st.markdown("---")
        st.subheader("📈 Aggregated Unique Stock Metrics View")
        st.markdown("Deduplicated view showing recurring recommendations, streak counts, and activity dates for each stock.")
        
        conn = get_db_connection()
        agg_df = pd.read_sql_query("""
            SELECT 
                ticker,
                cap,
                COUNT(*) AS total_recommendations,
                MIN(date) AS first_recommended,
                MAX(date) AS latest_recommended,
                GROUP_CONCAT(DISTINCT date) AS recommendation_days_list,
                MAX(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) AS is_currently_active
            FROM recommendations
            GROUP BY ticker
            ORDER BY total_recommendations DESC
        """, conn)
        conn.close()
        st.dataframe(agg_df, use_container_width=True)

    else:
        st.info("No recommendations synced yet. Run the AI Researcher in Tab 1.")

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
    st.header("📰 Daily Audit Newsletter & Reports Manager")
    audit_files = sorted(os.listdir(AUDIT_DIR), reverse=True)
    if audit_files:
        col_sel, col_del = st.columns([3, 1])
        with col_sel:
            selected_audit = st.selectbox("Select Report Date to Preview", audit_files)
        with col_del:
            st.write(" ")
            st.write(" ")
            if st.button("🗑️ Delete Selected Audit Log", type="secondary"):
                file_to_del = os.path.join(AUDIT_DIR, selected_audit)
                if os.path.exists(file_to_del):
                    os.remove(file_to_del)
                    st.success(f"Deleted `{selected_audit}`!")
                    st.rerun()
                    
        st.markdown("---")
        with open(os.path.join(AUDIT_DIR, selected_audit), 'r') as f:
            st.markdown(f.read())
    else:
        st.info("No audit reports generated yet.")

with tab5:
    st.header("🔍 AI Payload Inspector (Step 1 & Step 2 Debug Logs)")
    if debug_mode:
        st.subheader("Step 1 Payload (Initial Screening & Active Recs)")
        st.json(st.session_state.debug_logs.get("step1_payload", {}))
        
        st.markdown("---")
        st.subheader("Step 2 Payload (Competitor Re-evaluation & Parameter Audit)")
        st.json(st.session_state.debug_logs.get("step2_payload", {}))
    else:
        st.info("⚠️ Debug Mode is currently turned OFF. Enable '🐛 Enable Debug Mode' in the sidebar.")

with tab6:
    st.header("🛠️ Real-Time Diagnostics & Scraper Logs")
    if debug_mode:
        if st.button("Run Live Diagnostics Test Now"):
            _, test_logs = fetch_hybrid_news()
            st.subheader("Scraper Status")
            st.dataframe(pd.DataFrame(test_logs), use_container_width=True)
            
        if st.session_state.debug_logs["scraper_status"]:
            st.subheader("Last Scraper Execution Status")
            st.dataframe(pd.DataFrame(st.session_state.debug_logs["scraper_status"]), use_container_width=True)
            
        if st.session_state.debug_logs["last_market_data"]:
            st.subheader("Last Fetched Market Data Snapshot Summary")
            st.write("Cached tickers count / tiers loaded successfully.")
    else:
        st.info("⚠️ Debug Mode is currently turned OFF. Enable '🐛 Enable Debug Mode' in the sidebar.")

with tab7:
    st.header("🧠 AI Self-Reflection & Learnings (`learnings.md`)")
    
    if st.button("✨ Auto-Generate & Update Learnings via AI", type="primary"):
        openrouter_key = st.secrets.get("OPENROUTER_KEY", config.get("openrouter_key", ""))
        if not openrouter_key:
            st.error("Please enter your OpenRouter API Key.")
        else:
            with st.spinner("AI is analyzing all historical trade outcomes and audit reports to generate new quantitative rules..."):
                conn = get_db_connection()
                all_recs = pd.read_sql_query("SELECT * FROM recommendations", conn).to_dict(orient="records")
                conn.close()
                
                current_learnings = "# AI Quantitative Learnings\nNo past notes."
                if os.path.exists(LEARNINGS_FILE):
                    with open(LEARNINGS_FILE, 'r') as f:
                        current_learnings = f.read()
                        
                summarize_prompt = f"""
                You are a Senior AI Quantitative Risk Manager. Review the historical recommendation database and existing memory notes below.
                
                ### Current Learnings Memory:
                {current_learnings}
                
                ### Complete Historical Trade Ledger:
                {json.dumps(all_recs, indent=2)}
                
                ### Task:
                1. Analyze winning trades vs. losing trades vs. pending trades.
                2. Formulate 3-5 concise, high-conviction bullet rules.
                3. Return the COMPLETE updated `learnings.md` text content in Markdown format.
                """
                
                client = OpenAI(base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)", api_key=openrouter_key)
                sum_res = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": summarize_prompt}],
                    temperature=0.2
                )
                
                new_learnings_text = sum_res.choices[0].message.content
                with open(LEARNINGS_FILE, 'w') as f:
                    f.write(new_learnings_text)
                st.success("✨ `learnings.md` successfully updated with new AI insights!")
                st.rerun()

    st.markdown("---")
    
    learnings_text = "# AI Quantitative Learnings\nNo past notes."
    if os.path.exists(LEARNINGS_FILE):
        with open(LEARNINGS_FILE, 'r') as f:
            learnings_text = f.read()
            
    updated_learnings = st.text_area("Edit Learnings & Rules for AI Memory", value=learnings_text, height=300)
    if st.button("Save Learnings Manually"):
        with open(LEARNINGS_FILE, 'w') as f:
            f.write(updated_learnings)
        st.success("Learnings saved!")

with tab8:
    st.header("🗄️ Database Management & Table Truncation Control (SQLite Cloud)")
    st.markdown("Inspect cloud database structure, view record counts, and safely clear or truncate individual tables.")

    conn = get_db_connection()
    cursor = conn.cursor()
    tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
    conn.close()

    table_metadata = {
        "recommendations": {
            "description": "Stores all AI-generated stock recommendations, entry prices, targets, stop-losses, selection status (`picked`), trade outcomes (`Win/Loss`), and parameter change notes.",
            "usage": "Used in Tab 1 (active rec context), Tab 2 (ledger & target/stop-loss checker), Tab 3 (confidence score), and Tab 7 (AI learnings)."
        },
        "portfolio": {
            "description": "Stores current stock holdings (ticker, quantity, buy price).",
            "usage": "Used in Tab 1 (LLM context) and Tab 2 (Portfolio Health Check)."
        },
        "market_cache": {
            "description": "Caches ticker fundamentals, RSI, MACD, and pricing metrics with 6-hour TTL.",
            "usage": "Used across all stock metric fetches to prevent redundant API calls."
        },
        "news_cache": {
            "description": "Caches hybrid RSS and web-crawled financial news corpus with 6-hour TTL.",
            "usage": "Used in Tab 1 news aggregation."
        }
    }

    for tbl in tables:
        with st.expander(f"📁 Table: `{tbl}`"):
            meta = table_metadata.get(tbl, {"description": "Custom table.", "usage": "Internal app usage."})
            st.markdown(f"**📖 What it stores:** {meta['description']}")
            st.markdown(f"**⚙️ Where data is used:** {meta['usage']}")
            
            conn = get_db_connection()
            count_df = pd.read_sql_query(f"SELECT COUNT(*) AS total_rows FROM {tbl}", conn)
            preview_df = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 5", conn)
            conn.close()
            
            total_rows = count_df["total_rows"].iloc[0]
            st.metric("Total Records", total_rows)
            
            if total_rows > 0:
                st.markdown("**Sample Preview (First 5 Rows):**")
                st.dataframe(preview_df, use_container_width=True)
            
            confirm_key = f"confirm_trunc_{tbl}"
            confirm_chk = st.checkbox(f"I want to clear/truncate table `{tbl}`", key=confirm_key)
            
            if st.button(f"🗑️ Truncate Table `{tbl}`", key=f"btn_trunc_{tbl}", type="secondary"):
                if confirm_chk:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(f"DELETE FROM {tbl}")
                    conn.commit()
                    conn.close()
                    st.success(f"Table `{tbl}` has been successfully truncated!")
                    st.rerun()
                else:
                    st.warning(f"Please check the confirmation box above before truncating `{tbl}`.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: grey; font-size: 13px;'>"
    f"📈 AI Stock Researcher & Quantitative Lab | <b>Version:</b> {APP_VERSION} | <b>Last Build:</b> {LAST_UPDATED}"
    f"</div>",
    unsafe_allow_html=True
)
