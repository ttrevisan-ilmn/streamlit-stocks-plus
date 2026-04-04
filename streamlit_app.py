import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from asbury_metrics import get_asbury_6_signals, get_asbury_6_historical
from seaf_model import get_seaf_model, get_top_3_sectors
from gamma_profile import get_gamma_profile
from options_flow import get_daily_flow_snapshot, analyze_flow_sentiment, get_volatility_analysis
from fundamental_metrics import fetch_fundamental_data, format_large_number
from congress_tracker import fetch_congress_members, fetch_stock_disclosures, get_top_traded_tickers, get_active_traders, check_watchlist_overlap
from macro_analysis import fetch_macro_data, get_yield_curve_data, get_asset_performance, render_yield_curve_chart, render_intermarket_chart
from screener_engine import get_screener_universe, fetch_screener_data, apply_strategy
from power_gauge import calculate_power_gauge
from weinstein import get_weinstein_stage
from canslim import get_canslim_metrics
from services.data_fetcher import get_ticker_options, calculate_mphinancial_mechanics
from services.logger import setup_logger
from datetime import datetime
import re
import streamlit.components.v1 as components

logger = setup_logger(__name__)

# --- PERSISTENT TAB STATE ---
# Preserve the active tab across reruns so widget interactions don't jump
# back to tab 0. st.tabs in Streamlit 1.28+ supports `key` which pairs with
# session_state to restore the selection on the next rerun.
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

def get_tv_symbol(ticker):
    """Simple mapping for TradingView symbols."""
    ticker = ticker.upper()
    # Manual Mapping for common ETFs/Indices
    if ticker in ['SPY', 'IWM', 'QQQ', 'DIA', 'GLD', 'SLV', 'TLT']:
        return f"AMEX:{ticker}"
    if ticker in ['VIX']:
        return "CBOE:VIX"
    if ticker in ['BTC-USD']:
        return "COINBASE:BTCUSD"
    # Default assumption for stocks (imperfect but fast)
    return f"NASDAQ:{ticker}"

def render_mini_chart_html(symbol, description):
    """Generates HTML for TradingView Mini Chart widget."""
    return f"""
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
  {{
  "symbol": "{symbol}",
  "width": "100%",
  "height": "100%",
  "locale": "en",
  "dateRange": "12M",
  "colorTheme": "dark",
  "isTransparent": false,
  "autosize": true,
  "largeChartUrl": ""
}}
  </script>
</div>
"""

# --- API CALL COUNTER WITH PERSISTENCE ---
import json
import os

API_STATS_FILE = "api_stats.json"

def load_total_calls():
    """Load total API calls from persistent storage."""
    if os.path.exists(API_STATS_FILE):
        try:
            with open(API_STATS_FILE, 'r') as f:
                return json.load(f).get('total_calls', 0)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading API stats: {e}")
            return 0
    return 0

def save_total_calls(count):
    """Save total API calls to persistent storage."""
    try:
        with open(API_STATS_FILE, 'w') as f:
            json.dump({'total_calls': count}, f)
    except IOError as e:
        print(f"Error saving API stats: {e}")
        pass

if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
    st.session_state.api_reset_time = datetime.now()

if 'total_api_calls' not in st.session_state:
    st.session_state.total_api_calls = load_total_calls()

def track_api_call():
    """Increment API call counters and save persistence."""
    st.session_state.api_calls += 1
    st.session_state.total_api_calls += 1
    save_total_calls(st.session_state.total_api_calls)

# --- THE MPHINANCAL ENGINE ---
# (Moved to services/data_fetcher.py)

# --- UI SETUP ---
st.set_page_config(page_title="Mphinancial Terminal", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for that "Sober Terminal" Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; color: white !important; }
    .stMetric label { color: #9ca3af !important; }
    .stMetric div[data-testid="stMetricValue"] { color: white !important; }
    </style>
    """, unsafe_allow_html=True)


st.sidebar.title("Phinancial Terminal")

# --- TICKER SELECTION ---
custom_input = st.sidebar.text_input("Quick Ticker Search:", help="Type any symbol (e.g. EOSE, TSLL) to override list selection.").upper().strip()
if custom_input:
    custom_input = re.sub(r'[^A-Z0-9-]', '', custom_input)

ticker_options = get_ticker_options()
# Smart Default: Initialize Session State to SPY if not set
if "ticker_selector" not in st.session_state:
    default_idx = 0
    if ticker_options:
        for i, opt in enumerate(ticker_options):
            if opt.startswith("SPY"):
                default_idx = i
                break
        st.session_state.ticker_selector = ticker_options[default_idx]

selected_option = st.sidebar.selectbox(
    "Or Browse S&P 500:", 
    options=ticker_options, 
    key="ticker_selector",
    help="Select from supported S&P 500 stocks and ETFs."
)

if custom_input:
    ticker = custom_input
else:
    ticker = selected_option.split(" - ")[0]

# Sanitize for yfinance (BRK.B -> BRK-B)
ticker = ticker.replace('.', '-')

# --- ANALYSIS STATE MANAGEMENT ---
def initialize_analysis_state():
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {
            'ticker': None,
            'power_gauge': None,
            'weinstein': None,
            'canslim': None,
            'timestamp': None
        }

def run_analysis_pipeline(ticker_symbol):
    """Orchestrates fetching data for all strategy modules."""
    initialize_analysis_state()
    
    # Check if we already have valid data for this ticker
    current_data = st.session_state.analysis_data
    if current_data['ticker'] == ticker_symbol and current_data['power_gauge'] is not None:
        return

    # Create a status container
    # Provide a spinner since st.status is new in Streamlit 1.25.0, assuming support but fallback to plain spinner if needed.
    # We'll use st.status as it's cleaner.
    try:
        with st.status(f"Running Multi-Strategy Analysis for {ticker_symbol}...", expanded=True) as status:
            # 1. Power Gauge
            status.write("⚡ Computing Power Gauge (20-Factor Model)...")
            pg_data = calculate_power_gauge(ticker_symbol)
            if not pg_data: status.write("⚠️ Power Gauge failed.")
            
            # 2. Weinstein Stage
            status.write("📉 Identifying Weinstein Stage...")
            w_data = get_weinstein_stage(ticker_symbol)
            if not w_data: status.write("⚠️ Weinstein Stage failed.")
            
            # 3. CANSLIM
            status.write("🚀 Checking CANSLIM Factors...")
            c_data = get_canslim_metrics(ticker_symbol)
            if not c_data: status.write("⚠️ CANSLIM failed.")
            
            # Update State
            st.session_state.analysis_data = {
                'ticker': ticker_symbol,
                'power_gauge': pg_data,
                'weinstein': w_data,
                'canslim': c_data,
                'timestamp': datetime.now()
            }
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
    except Exception as e:
        st.error(f"Error running analysis pipeline: {e}")

# Run Pipeline on Ticker Change
if ticker:
    run_analysis_pipeline(ticker)

# --- WATCHLIST ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

st.sidebar.divider()
st.sidebar.markdown("### Watchlist")

# Add to watchlist button
if ticker and ticker not in st.session_state.watchlist:
    if st.sidebar.button(f"+ Add {ticker} to Watchlist"):
        st.session_state.watchlist.append(ticker)
        st.rerun()

# Display watchlist
if st.session_state.watchlist:
    for wl_ticker in st.session_state.watchlist:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.markdown(f"**{wl_ticker}**")
        if col2.button("X", key=f"remove_{wl_ticker}"):
            st.session_state.watchlist.remove(wl_ticker)
            st.rerun()
else:
    st.sidebar.caption("No tickers in watchlist")

# --- API CALL COUNTER ---
st.sidebar.divider()
st.sidebar.markdown("### API Usage")

time_since_reset = datetime.now() - st.session_state.api_reset_time
minutes_elapsed = int(time_since_reset.total_seconds() / 60)

# Display counters as styled boxes for better visibility
st.sidebar.markdown(f"""
<div style='background-color: #1e3a5f; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
    <p style='margin: 0; color: #94a3b8; font-size: 0.8em;'>Session Calls ({minutes_elapsed}m)</p>
    <h3 style='margin: 5px 0; color: white;'>{st.session_state.api_calls}</h3>
</div>
<div style='background-color: #1e3a5f; padding: 10px; border-radius: 5px;'>
    <p style='margin: 0; color: #94a3b8; font-size: 0.8em;'>Total (All Time)</p>
    <h3 style='margin: 5px 0; color: white;'>{st.session_state.total_api_calls}</h3>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("Reset Session"):
    st.session_state.api_calls = 0
    st.session_state.api_reset_time = datetime.now()
    st.rerun()

st.sidebar.caption("Rate limit: ~2,000/hour")

# --- SETTINGS ---
st.sidebar.divider()
with st.sidebar.expander("⚙️ Settings"):
    st.caption("Configure API keys for external data sources.")
    user_congress_key = st.text_input("Congress.gov API Key", type="password", key="user_congress_key_input")
    if user_congress_key:
        st.session_state['user_congress_key'] = user_congress_key
        st.success("Key saved!")
    elif 'congress_api_key' in st.secrets:
        st.info("Using key from secrets.toml")
    else:
        st.warning("No API key found.")
        st.markdown("[Get a free key](https://api.congress.gov/sign-up/)")

# --- HEADER: TICKER TAPE REMOVED (Replaced by Grid in Tab 1) ---

# --- TOP SIGNALS TODAY ---
st.markdown("### ⚡ Top Signals Today")
st.caption(f"Data freshness: ~{datetime.now().strftime('%Y-%m-%d %H:%M %Z')}")

try:
    macro_data = fetch_macro_data()
    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
    
    # 1. Market Trend (SP500 > 200 SMA)
    if not macro_data.empty and '^GSPC' in macro_data['Close']:
        spx = macro_data['Close']['^GSPC']
        spx_close = spx.iloc[-1]
        spx_sma200 = spx.rolling(200).mean().iloc[-1]
        trend_color = "#4ade80" if spx_close > spx_sma200 else "#ef4444"
        trend_text = "UPTREND" if spx_close > spx_sma200 else "DOWNTREND"
        sig_col1.markdown(f"**SPX Form:** <span style='color:{trend_color}; font-weight:bold;'>{trend_text}</span>", unsafe_allow_html=True)
    else:
        sig_col1.markdown("**SPX Form:** N/A")
        
    # 2. Yield Spread (10Y-3M)
    yields = get_yield_curve_data(macro_data)
    if yields is not None:
        spread = yields['Spread'].dropna().iloc[-1]
        spread_color = "#ef4444" if spread < 0 else "#4ade80"
        spread_text = "INVERTED" if spread < 0 else "NORMAL"
        sig_col2.markdown(f"**Yield Curve:** <span style='color:{spread_color}; font-weight:bold;'>{spread_text}</span> ({spread:+.0f} bps)", unsafe_allow_html=True)
    else:
        sig_col2.markdown("**Yield Curve:** N/A")

    # 3. Gold/Risk Proxy
    if not macro_data.empty and 'GC=F' in macro_data['Close']:
        gold = macro_data['Close']['GC=F']
        gold_ret = (gold.iloc[-1] / gold.iloc[-20]) - 1 # 1-month return
        gold_color = "#fbbf24"
        sig_col3.markdown(f"**Gold (1M):** <span style='color:{gold_color}; font-weight:bold;'>{gold_ret:+.1%}</span>", unsafe_allow_html=True)
    else:
        sig_col3.markdown("**Gold (1M):** N/A")
        
    # 4. Active Target Context
    sig_col4.markdown(f"**Focus:** <span style='color:#60a5fa; font-weight:bold;'>{ticker}</span>", unsafe_allow_html=True)

except Exception as e:
    st.warning("⚠️ Top signals temporarily unavailable.")

# --- SIDEBAR NAVIGATION (Persists across reruns unlike st.tabs) ---
st.sidebar.divider()
TAB_LABELS = [
    "📊 Market Health",
    "📈 Sector Rotation",
    "🌐 Intermarket",
    "📉 Stock Analysis",
    "🏛️ Congress Trades",
    "🌪️ Options Flow",
    "🔍 Stock Screener",
    "⚡ Power Gauge",
    "📉 Stage Analysis",
    "🚀 CANSLIM",
    "💼 Navellier Grade",
]

if 'active_view' not in st.session_state:
    st.session_state.active_view = TAB_LABELS[0]

active_view = st.sidebar.radio(
    "Navigate",
    TAB_LABELS,
    key='active_view',
)

st.divider()

# --- RENDER ACTIVE VIEW ---
if active_view == "📊 Market Health":
    try:
        from views.market_health import render_market_health
        render_market_health(render_mini_chart_html, track_api_call)
    except Exception as e:
        logger.error(f"Error rendering Market Health: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "📈 Sector Rotation":
    try:
        from views.sector_rotation import render_sector_rotation
        render_sector_rotation(track_api_call)
    except Exception as e:
        logger.error(f"Error rendering Sector Rotation: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "🌐 Intermarket":
    try:
        from views.intermarket import render_intermarket
        render_intermarket(track_api_call)
    except Exception as e:
        logger.error(f"Error rendering Intermarket: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "📉 Stock Analysis":
    try:
        from views.stock_analysis import render_stock_analysis
        render_stock_analysis(ticker, track_api_call, run_analysis_pipeline, calculate_mphinancial_mechanics, get_tv_symbol)
    except Exception as e:
        logger.error(f"Error rendering Stock Analysis: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "🏛️ Congress Trades":
    try:
        from views.congress_trades import render_congress_trades
        render_congress_trades(track_api_call)
    except Exception as e:
        logger.error(f"Error rendering Congress Trades: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "🌪️ Options Flow":
    try:
        from views.options_intelligence import render_options_intelligence
        render_options_intelligence(ticker, track_api_call)
    except Exception as e:
        logger.error(f"Error rendering Options Flow: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "🔍 Stock Screener":
    try:
        from views.screener_tab import render_screener
        render_screener()
    except Exception as e:
        logger.error(f"Error rendering Screener: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "⚡ Power Gauge":
    try:
        from views.power_gauge_tab import render_power_gauge
        render_power_gauge(ticker)
    except Exception as e:
        logger.error(f"Error rendering Power Gauge: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "📉 Stage Analysis":
    try:
        from views.weinstein_tab import render_weinstein
        render_weinstein(ticker)
    except Exception as e:
        logger.error(f"Error rendering Stage Analysis: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "🚀 CANSLIM":
    try:
        from views.canslim_tab import render_canslim
        render_canslim(ticker)
    except Exception as e:
        logger.error(f"Error rendering CANSLIM: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

elif active_view == "💼 Navellier Grade":
    try:
        from views.navellier_tab import render_navellier
        render_navellier(ticker)
    except Exception as e:
        logger.error(f"Error rendering Navellier Grade: {e}")
        st.error("⚠️ An error occurred. Please try again later.")

