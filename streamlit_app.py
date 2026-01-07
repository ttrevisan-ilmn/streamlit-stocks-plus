import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from asbury_metrics import get_asbury_6_signals, get_asbury_6_historical
from seaf_model import get_seaf_model, get_top_3_sectors
from gamma_profile import get_gamma_profile
from options_flow import get_daily_flow_snapshot, analyze_flow_sentiment
from fundamental_metrics import fetch_fundamental_data, format_large_number
from congress_tracker import fetch_congress_members, fetch_stock_disclosures, get_top_traded_tickers, get_active_traders, check_watchlist_overlap
from macro_analysis import fetch_macro_data, get_yield_curve_data, get_asset_performance, render_yield_curve_chart, render_intermarket_chart
from datetime import datetime

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
        except:
            return 0
    return 0

def save_total_calls(count):
    """Save total API calls to persistent storage."""
    try:
        with open(API_STATS_FILE, 'w') as f:
            json.dump({'total_calls': count}, f)
    except:
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
def calculate_mphinancial_mechanics(df):
    # 1. The EMA Stack (The mphinancial Core)
    for p in [8, 21, 34, 55, 89]:
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    
    # 2. The 200 SMA (The Wind)
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # 3. ADX (Trend Strength) - Manual Calculation for Chromebook Compatibility
    n = 14
    h = df['High'].values.flatten()
    l = df['Low'].values.flatten()
    c = df['Close'].values.flatten()
    
    # Vectorized DM calculation
    plus_dm = np.insert(np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]), np.maximum(h[1:] - h[:-1], 0), 0), 0, 0)
    minus_dm = np.insert(np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]), np.maximum(l[:-1] - l[1:], 0), 0), 0, 0)
    tr = np.insert(np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1]))), 0, 0)
    
    # Ensure Series are 1D to prevent dimension errors
    atr_series = pd.Series(tr, index=df.index).rolling(window=n).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=n).mean() / atr_series)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=n).mean() / atr_series)
    
    # Handle division by zero/NaN for DX calculation
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0)
    
    df['ADX'] = dx.rolling(window=n).mean()
    df['ATR'] = atr_series
    return df

# --- UI SETUP ---
st.set_page_config(page_title="Mphinancial Terminal", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for that "Sober Terminal" Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)


st.sidebar.title("Phinancial Terminal")

# Ticker input
ticker = st.sidebar.text_input("Enter Ticker:", value="SPY", placeholder="e.g. SPY, AAPL").upper()

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

# --- MAIN TABS ---
tab1, tab2, tab5, tab3, tab6, tab4 = st.tabs(["📊 Market Health", "📈 Sector Rotation", "🌐 Intermarket", "📉 Stock Analysis", "🌪️ Options Flow", "🏛️ Congress Trades"])

# --- TAB 1: MARKET HEALTH DASHBOARD ---
with tab1:
    st.title("📊 Market Health Gauge (A6)")
    st.markdown("""
    The Asbury 6 is a quantitative, daily gauge of US equity market internal strength based on six key metrics.
    **Signal:** Buy SPY when 4+ components are green (Positive) • Move to cash when 4+ are red (Negative)
    """)
    
    with st.spinner("⏳ Loading market health data..."):
        track_api_call()  # Track API calls for Asbury 6
        asbury_data = get_asbury_6_signals()
        track_api_call()  # Track historical data
        historical_data = get_asbury_6_historical(days=90)
    
    if 'error' in asbury_data and asbury_data['signal'] == 'ERROR':
        st.error(f"⚠️ Error fetching Asbury 6 data: {asbury_data['error']}")
    else:
        # Overall Signal and Gauge
        signal = asbury_data['signal']
        pos_count = asbury_data['positive_count']
        neg_count = asbury_data['negative_count']
        
        # Create top section with Gauge, Signal, and VIX Term Structure
        top_col1, top_col2 = st.columns([1, 1])
        
        with top_col1:
            # Merged Gauge and Signal
            gauge_col, signal_col = st.columns([1, 1.5])
            
            with gauge_col:
                gauge_value = pos_count  # 0-6 scale
                gauge_color = 'green' if signal == 'BUY' else ('red' if signal == 'CASH' else 'yellow')
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=gauge_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "A6 Score", 'font': {'size': 14, 'color': 'white'}},
                    number={'font': {'size': 30, 'color': 'white'}},
                    gauge={
                        'axis': {'range': [None, 6], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "white",
                        'steps': [
                            {'range': [0, 3], 'color': 'rgba(255,0,0,0.3)'},
                            {'range': [3, 4], 'color': 'rgba(255,255,0,0.3)'},
                            {'range': [4, 6], 'color': 'rgba(0,255,0,0.3)'}
                        ],
                        'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 4}
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                      font={'color': "white"}, height=160, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_gauge, width="stretch")
            
            with signal_col:
                if signal == 'BUY':
                    st.success("### 🟢 BUY SIGNAL")
                    st.markdown(f"**{pos_count}/6 Positive**")
                    st.markdown("Recomm: **Increase Equity**")
                elif signal == 'CASH':
                    st.error("### 🔴 CASH SIGNAL")
                    st.markdown(f"**{neg_count}/6 Negative**")
                    st.markdown("Recomm: **Move to Cash**")
                else:
                    st.warning("### 🟡 NEUTRAL")
                    st.markdown(f"**{pos_count} Pos / {neg_count} Neg**")
                    st.markdown("Recomm: **Hold/Wait**")
                
                st.caption(f"Updated: {asbury_data['timestamp']}")

        with top_col2:
            # Inline VIX Term Structure
            try:
                import yfinance as yf
                vix_spot = yf.download('^VIX', period='1d', progress=False)['Close'].iloc[-1]
                if hasattr(vix_spot, "item"): vix_spot = vix_spot.item()
                
                vix3m = yf.download('^VIX3M', period='1d', progress=False)['Close'].iloc[-1]
                if hasattr(vix3m, "item"): vix3m = vix3m.item()
                
                spread = float(vix3m) - float(vix_spot)
                
                structure_color = "#1a4d2e" if spread > 0 else "#4d1a1a"
                structure_text = "Contango (Bullish)" if spread > 0 else "Backwardation (Bearish)"
                
                st.markdown(f"""
                <div style='background-color: {structure_color}; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #4b5563;'>
                    <h5 style='color: #e5e7eb; margin: 0;'>VIX Term Structure</h5>
                    <div style='display: flex; justify-content: space-around; margin-top: 5px;'>
                        <div><span style='color:#9ca3af; font-size:0.8em'>Spot</span><br><b>{float(vix_spot):.2f}</b></div>
                        <div><span style='color:#9ca3af; font-size:0.8em'>3M Future</span><br><b>{float(vix3m):.2f}</b></div>
                        <div><span style='color:#9ca3af; font-size:0.8em'>Spread</span><br><b>{spread:+.2f}</b></div>
                    </div>
                    <p style='margin: 5px 0 0 0; font-weight: bold; color: white;'>{structure_text}</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.caption("VIX data unavailable")
        
        st.divider()
        
        # Historical Chart with SPX
        if not historical_data.empty:
            st.subheader("📈 A6 Signal History vs SPX Performance")
            
            # Normalize SPY price for comparison (set first value to 100)
            historical_data['SPY_Normalized'] = (historical_data['SPY_Close'] / historical_data['SPY_Close'].iloc[0]) * 100
            
            # Create subplot with two y-axes
            fig_history = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=("SPX Price (Normalized to 100)", "A6 Signal Count"),
                vertical_spacing=0.1
            )
            
            # SPY price line
            fig_history.add_trace(
                go.Scatter(
                    x=historical_data['Date'],
                    y=historical_data['SPY_Normalized'],
                    name='SPX',
                    line=dict(color='cyan', width=2),
                    hovertemplate='Date: %{x}<br>SPX: %{y:.1f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add background shading for BUY/CASH signals
            for i in range(len(historical_data)-1):
                row = historical_data.iloc[i]
                next_row = historical_data.iloc[i+1]
                
                if row['Signal'] == 'BUY':
                    color = 'rgba(0,255,0,0.1)'
                elif row['Signal'] == 'CASH':
                    color = 'rgba(255,0,0,0.1)'
                else:
                    color = 'rgba(255,255,0,0.05)'
                
                fig_history.add_vrect(
                    x0=row['Date'], x1=next_row['Date'],
                    fillcolor=color, layer="below", line_width=0,
                    row=1, col=1
                )
            
            # A6 positive count area chart
            fig_history.add_trace(
                go.Scatter(
                    x=historical_data['Date'],
                    y=historical_data['Positive_Count'],
                    name='Positive Signals',
                    fill='tozeroy',
                    line=dict(color='lime', width=1),
                    hovertemplate='Date: %{x}<br>Positive: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add reference line at 4 (signal threshold)
            fig_history.add_hline(y=4, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)
            
            fig_history.update_xaxes(title_text="Date", row=2, col=1)
            fig_history.update_yaxes(title_text="Normalized Price", row=1, col=1)
            fig_history.update_yaxes(title_text="Count", range=[0, 6], row=2, col=1)
            
            fig_history.update_layout(
                template="plotly_dark",
                height=300,  # Reduced height
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig_history, width="stretch")
        
        st.divider()
        
        # Metric Cards in 3x2 Grid with better styling
        st.subheader("Six Market Health Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        for idx, metric in enumerate(asbury_data['metrics']):
            # Distribute metrics across columns
            if idx % 3 == 0:
                col = col1
            elif idx % 3 == 1:
                col = col2
            else:
                col = col3
            
            with col:
                status_color = "🟢" if metric['status'] == 'Positive' else "🔴"
                status_bg = "#1a4d2e" if metric['status'] == 'Positive' else "#4d1a1a"
                
                # Styled container with better contrast
                # Compact Styled Container
                st.markdown(f"""
                <div style='background-color: {status_bg}; padding: 10px; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid {"#4ade80" if metric["status"] == "Positive" else "#ef4444"}'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h5 style='margin: 0; color: white; font-size: 0.95em;'>{status_color} {metric['name']}</h5>
                        <span style='font-size: 0.8em; color: #e5e7eb; font-weight: bold;'>{metric['status']}</span>
                    </div>
                    <p style='margin: 2px 0; color: #d1d5db; font-size: 0.8em;'>{metric['value']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # --- VIX TERM STRUCTURE REMOVED HERE AS MOVED TO TOP ---
    
    st.markdown("---")



# --- TAB 2: SEAF MODEL (SECTOR ETF ASSET FLOWS) ---
with tab2:
    st.title("📈 SEAF Sector Rotation Model")
    st.markdown("""
    **SEAF (Sector ETF Asset Flows)** - A quantitative sector rotation model that ranks the 11 Select Sector SPDR ETFs 
    by "following the money" across four timeframes. Always fully invested in the top 3 ranked sectors.
    """)
    
    with st.spinner("🔄 Loading sector rankings..."):
        track_api_call()  # Track SEAF API call
        seaf_data = get_seaf_model()
    
    if seaf_data.empty:
        st.error("⚠️ Error calculating SEAF rankings")
    else:
        # Top 3 Allocation Banner
        top_3 = get_top_3_sectors(seaf_data)
        
        st.subheader("🎯 Current Top 3 Allocation")
        
        top_col1, top_col2, top_col3 = st.columns(3)
        
        for idx, (col, (_, row)) in enumerate(zip([top_col1, top_col2, top_col3], top_3.iterrows())):
            with col:
                category_color = "#1a4d2e" if row['Category'] == 'Favored' else ("#4d4d1a" if row['Category'] == 'Neutral' else "#4d1a1a")
                st.markdown(f"""
                <div style='background-color: {category_color}; padding: 20px; border-radius: 10px; text-align: center; border: 3px solid #4ade80;'>
                    <h2 style='margin: 0; color: white;'>#{row['Rank']} {row['Ticker']}</h2>
                    <h4 style='margin: 5px 0; color: #e5e7eb;'>{row['Sector']}</h4>
                    <p style='margin: 5px 0; color: #d1d5db; font-size: 0.9em;'>Score: {row['Total_Score']}</p>
                    <p style='margin: 0; color: #4ade80; font-weight: bold;'>{row['Category']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.caption("Equal allocation across these 3 sectors")
        
        # --- TOP 3 SECTOR CHARTS ---
        st.divider()
        st.subheader("Price Performance - Top 3 Sectors")
        
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        
        # Fetch price data for top 3 sectors
        from seaf_model import fetch_sector_data
        end_str = datetime.now().strftime('%Y-%m-%d')
        start_str = (datetime.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
        
        for col, (_, row) in zip([chart_col1, chart_col2, chart_col3], top_3.iterrows()):
            with col:
                ticker = row['Ticker']
                sector_data = fetch_sector_data(ticker, start_str, end_str)
                
                if not sector_data.empty:
                    # Calculate return
                    first_close = sector_data['Close'].iloc[0]
                    last_close = sector_data['Close'].iloc[-1]
                    pct_return = ((last_close - first_close) / first_close) * 100
                    
                    # Create mini chart
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Scatter(
                        x=sector_data.index,
                        y=sector_data['Close'],
                        mode='lines',
                        fill='tozeroy',
                        fillcolor='rgba(74, 222, 128, 0.2)' if pct_return > 0 else 'rgba(239, 68, 68, 0.2)',
                        line=dict(color='#4ade80' if pct_return > 0 else '#ef4444', width=2),
                        hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
                    ))
                    
                    fig_mini.update_layout(
                        template='plotly_dark',
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0),
                        title=dict(
                            text=f"{ticker}: {pct_return:+.1f}%",
                            font=dict(size=14, color='#4ade80' if pct_return > 0 else '#ef4444')
                        ),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=True, showgrid=False),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_mini, width="stretch")
                else:
                    st.caption(f"{ticker}: No data")
        
        st.caption("90-day price performance")
        st.divider()
        
        # Rankings Table
        st.subheader("📊 Complete Sector Rankings")
        
        # Prepare display dataframe
        display_df = seaf_data[['Rank', 'Ticker', 'Sector', 'Trading', 'Tactical', 
                                 'Strategic', 'Long-term', 'Total_Score', 'Category']].copy()
        
        # Style the dataframe
        def color_category(val):
            if val == 'Favored':
                return 'background-color: #1a4d2e; color: white'
            elif val == 'Neutral':
                return 'background-color: #4d4d1a; color: white'
            else:
                return 'background-color: #4d1a1a; color: white'
        
        styled_df = display_df.style.map(color_category, subset=['Category'])
        
        # Rankings Table (Compacted into Expander)
        with st.expander("📊 View Complete Sector Rankings Table", expanded=False):
            st.dataframe(styled_df, width="stretch", hide_index=True)
            
            st.caption("""
            **How to Read Rankings:**
            - Lower rank numbers (1-3) = strongest asset inflows → Allocate here
            - Each timeframe ranks sectors 1-11 based on asset flows
            - Total Score = sum of all timeframe ranks (lower is better)
            - **Favored** (score ≤20) • **Neutral** (21-32) • **Avoid** (≥33)
            """)
        
        st.divider()
        
        # Scores Visualization
        st.subheader("📈 SEAF Scores Chart")
        
        # Create bar chart
        fig_seaf = go.Figure()
        
        # Color bars by category
        colors = seaf_data['Category'].map({
            'Favored': '#4ade80',
            'Neutral': '#fbbf24',
            'Avoid': '#ef4444'
        })
        
        fig_seaf.add_trace(go.Bar(
            x=seaf_data['Ticker'],
            y=seaf_data['Total_Score'],
            marker_color=colors,
            text=seaf_data['Total_Score'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Score: %{y}<br>%{customdata}<extra></extra>',
            customdata=seaf_data['Sector']
        ))
        
        # Add reference lines
        fig_seaf.add_hline(y=20, line_dash="dash", line_color="green", 
                          annotation_text="Favored Threshold", opacity=0.5)
        fig_seaf.add_hline(y=32, line_dash="dash", line_color="yellow", 
                          annotation_text="Neutral Threshold", opacity=0.5)
        
        fig_seaf.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Sector ETF",
            yaxis_title="Total Score (Lower = Better)",
            yaxis=dict(autorange="reversed"),  # Reverse so lower scores are at top
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        
        st.plotly_chart(fig_seaf, width="stretch")
        
        # Timeframe breakdown with color coding
        with st.expander("📋 View Timeframe Breakdown"):
            st.markdown("**Individual timeframe rankings and asset flow scores:**")
            
            breakdown_df = seaf_data[['Ticker', 'Sector', 
                                     'Trading', 'Trading_Score',
                                     'Tactical', 'Tactical_Score', 
                                     'Strategic', 'Strategic_Score',
                                     'Long-term', 'Long-term_Score']].copy()
            
            # Color code function for rankings (1-3 green, 4-8 yellow, 9-11 red)
            def color_rank(val):
                if isinstance(val, (int, float)):
                    if val <= 3:
                        return 'background-color: #1a4d2e; color: white'
                    elif val <= 8:
                        return 'background-color: #4d4d1a; color: white'
                    else:
                        return 'background-color: #4d1a1a; color: white'
                return ''
            
            # Color code function for scores (positive green, negative red)
            def color_score(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'background-color: #1a4d2e; color: white'
                    else:
                        return 'background-color: #4d1a1a; color: white'
                return ''
            
            # Apply styling
            styled_breakdown = breakdown_df.style.map(
                color_rank, 
                subset=['Trading', 'Tactical', 'Strategic', 'Long-term']
            ).map(
                color_score,
                subset=['Trading_Score', 'Tactical_Score', 'Strategic_Score', 'Long-term_Score']
            )
            
            st.dataframe(styled_breakdown, width="stretch", hide_index=True)
            
            st.caption("""
            **Timeframes:**
            - **Trading**: 20 days (~1 month)
            - **Tactical**: 60 days (~3 months)
            - **Strategic**: 120 days (~6 months)
            - **Long-term**: 252 days (~1 year)
            """)
    
    st.divider()
    
    # --- SECTOR CORRELATION HEATMAP ---
    with st.expander("View Sector Correlation Matrix"):
        st.markdown("**60-day correlation between sector ETFs:**")
        
        try:
            from seaf_model import fetch_sector_data, SECTOR_ETFS
            
            # Fetch 60-day data for all sectors
            end_str = datetime.now().strftime('%Y-%m-%d')
            start_str = (datetime.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
            
            sector_closes = {}
            for ticker in SECTOR_ETFS.keys():
                data = fetch_sector_data(ticker, start_str, end_str)
                if not data.empty:
                    sector_closes[ticker] = data['Close'].iloc[-60:]
            
            if sector_closes:
                close_df = pd.DataFrame(sector_closes)
                corr_matrix = close_df.pct_change().dropna().corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdYlGn',
                    zmin=-1, zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>'
                ))
                
                fig_corr.update_layout(
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                
                st.plotly_chart(fig_corr, width="stretch")
                
                st.caption("""
                **How to interpret:**
                - **Green (close to 1)**: Sectors move together
                - **Red (close to -1)**: Sectors move opposite
                - **Diversification tip**: Pair sectors with lower correlation
                """)
        except Exception as e:
            st.caption(f"Correlation matrix unavailable: {e}")
    
    st.markdown("---")


# --- TAB 5: INTERMARKET ANALYSIS ---
with tab5:
    st.title("🌐 Macro & Intermarket Intelligence")
    st.markdown("""
    **The Big Picture:** Analyzing relationships between bonds, commodities, currencies, and equities.
    """)
    
    with st.spinner("🌍 Loading macro indicators..."):
        track_api_call()
        macro_data = fetch_macro_data()
        
    if not macro_data.empty:
        # --- Top Level Metrics ---
        # Latest values
        try:
            # Safely access data handling both MultiIndex and standard structures
            closes = macro_data['Close'] if 'Close' in macro_data else macro_data
            
            # Helper to get last valid value
            def get_last(ticker):
                if ticker in closes:
                    return closes[ticker].dropna().iloc[-1]
                return 0
            
            us10y = get_last('^TNX')
            oil = get_last('CL=F')
            gold = get_last('GC=F')
            dxy = get_last('DX-Y.NYB')
            btc = get_last('BTC-USD')
            
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            
            m_col1.metric("US 10Y Yield", f"{us10y:.2f}%")
            m_col2.metric("Crude Oil", f"${oil:.2f}")
            m_col3.metric("Gold", f"${gold:,.0f}")
            m_col4.metric("Dollar Index", f"{dxy:.2f}")
            m_col5.metric("Bitcoin", f"${btc:,.0f}")
            
        except Exception as e:
            st.error(f"Error displaying metrics: {e}")
            
        st.divider()
        
        # --- Yield Curve Section ---
        st.subheader("Yield Curve Dynamics")
        yield_data = get_yield_curve_data(macro_data)
        
        if yield_data is not None:
            fig_yield = render_yield_curve_chart(yield_data)
            st.plotly_chart(fig_yield, width="stretch")
            
            last_spread = yield_data['Spread'].dropna().iloc[-1]
            if last_spread < 0:
                st.error(f"🚨 **Yield Curve Inverted ({last_spread:.2f} bps)**: Historical precursor to recession.")
            else:
                st.success(f"✅ **Normal Yield Curve (+{last_spread:.2f} bps)**: Healthy lending environment.")
        
        st.divider()
        
        # --- Intermarket Performance ---
        st.subheader("Asset Class Performance (1 Year)")
        perf_df = get_asset_performance(macro_data)
        
        if perf_df is not None:
            fig_perf = render_intermarket_chart(perf_df)
            st.plotly_chart(fig_perf, width="stretch")
            
            # Correlation Quick Check
            st.info("""
            **Intermarket Insights:**
            - **Strong Dollar** usually pressures Gold & Equities.
            - **Rising Yields** typically hurt Growth Stocks & Gold.
            - **Oil Spikes** can signal inflationary pressure (bad for bonds).
            """)
            
    else:
        st.warning("Unable to load macro data. Please check connection.")
    
    st.markdown("---")


# --- TAB 3: STOCK TICKER ANALYSIS ---
with tab3:
    st.title("📉 Stock Analysis (Mphinancial Engine)")
    
    if ticker:
        # Use 2y to ensure 200 SMA has enough data to stabilize
        track_api_call()  # Track stock data fetch
        data = yf.download(ticker, period="2y", interval="1d")
        
        if not data.empty and len(data) > 200:
            # Flatten MultiIndex columns if present (common with newer yfinance)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = calculate_mphinancial_mechanics(data)
            
            # Pull latest scalars correctly using .item() or float() to avoid Series ambiguity
            last_row = data.iloc[-1]
            price = float(last_row['Close'])
            sma200 = float(last_row['SMA200'])
            adx_val = float(last_row['ADX']) if not pd.isna(last_row['ADX']) else 0.0
            ema21 = float(last_row['EMA21'])
            atr = float(last_row['ATR']) if not pd.isna(last_row['ATR']) else 0.0
            
            # --- HEADER SECTION ---
            st.markdown("### Key Metrics")
            col_m1, col_m2, col_m3 = st.columns(3)
        
            # Use custom HTML for better visibility
            col_m1.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Current Price</p>
                <h2 style='color: white; margin: 5px 0;'>${price:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col_m2.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>ADX Strength</p>
                <h2 style='color: white; margin: 5px 0;'>{adx_val:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col_m3.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>ATR (Volatility)</p>
                <h2 style='color: white; margin: 5px 0;'>${atr:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # --- MAIN CONTENT ---
            chart_col, audit_col = st.columns([2, 1])

            with chart_col:
                st.subheader(f"📊 {ticker} Visual Audit")
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index, open=data['Open'], high=data['High'], 
                    low=data['Low'], close=data['Close'], name="Price")])
                
                # Add the EMA Stack to the chart
                colors = ['#00ffcc', '#00ccff', '#3366ff', '#6633ff', '#ff33cc']
                for i, p in enumerate([8, 21, 34, 55, 89]):
                    fig.add_trace(go.Scatter(x=data.index, y=data[f'EMA{p}'], 
                                             name=f'EMA{p}', line=dict(color=colors[i], width=1.5)))
                
                # Add the Wind (200 SMA)
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], 
                                         name='SMA 200 (The Wind)', line=dict(color='white', width=3, dash='dash')))
                
                fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, width="stretch")

            with audit_col:
                st.subheader("⚙️ Mechanics Check")
                
                # 1. Trend (The Wind)
                if price > sma200:
                    st.success("✅ SAILING WITH THE WIND: Price is above 200 SMA.")
                else:
                    st.error("❌ STAGNANT WATER: Price is below 200 SMA. No Long setup.")

                # 2. The Stack - FIXED: Individual comparisons to avoid "truth value of Series" error
                e8, e21, e34, e55, e89 = (float(last_row['EMA8']), float(last_row['EMA21']), 
                                          float(last_row['EMA34']), float(last_row['EMA55']), float(last_row['EMA89']))
                
                is_stacked = (e8 > e21) and (e21 > e34) and (e34 > e55) and (e55 > e89)
                
                if is_stacked:
                    st.success("✅ BULLISH STACK: EMAs are in perfect alignment.")
                else:
                    st.warning("⚠️ DISORDERED STACK: Trend lacks momentum.")
                
                # Display Raw EMA List
                with st.expander("View Raw EMA Stack Data"):
                    for p in [8, 21, 34, 55, 89]:
                        st.write(f"**EMA {p}:** ${float(last_row[f'EMA{p}']):.2f}")

                # 3. The Buy Zone (ATR logic)
                dist_to_21 = abs(price - ema21)
                in_buy_zone = dist_to_21 <= atr
                
                if in_buy_zone:
                    st.info("🎯 IN THE BUY ZONE: Price is within 1 ATR of the 21 EMA.")
                else:
                    st.warning("⌛ OVEREXTENDED: Price is too far from the mean. Wait for pullback.")

                # --- THE FINAL VERDICT ---
                st.divider()
                # Added ADX check to the final verdict logic
                if price > sma200 and is_stacked and adx_val >= 20 and in_buy_zone:
                    # Balloons removed per user request
                    st.markdown("### 🏆 HIGH QUALITY SETUP")
                    st.write("All Phinancial mechanics are aligned for an entry.")
                else:
                    st.markdown("### 🔍 MONITORING MODE")
                    st.write("Wait for all mechanical criteria to align.")
            
            
            # --- FUNDAMENTAL HEALTH (ROIC.AI STYLE) ---
            st.divider()
            st.subheader(f"📊 Fundamental Health: {ticker}")
            
            track_api_call()  # Track Fundamental Data Fetch
            fund_data = fetch_fundamental_data(ticker)
            
            if fund_data:
                # Row 1: The "Quality" Metrics (ROIC focus)
                f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                
                # Helper for consistent styling
                def metric_box(label, value, subtext="", color="#1e3a5f"):
                    st.markdown(f"""
                    <div style='background-color: {color}; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #374151;'>
                        <p style='color: #94a3b8; margin: 0; font-size: 0.85em; text-transform: uppercase;'>{label}</p>
                        <h3 style='color: white; margin: 4px 0; font-size: 1.4em;'>{value}</h3>
                        <p style='color: #d1d5db; margin: 0; font-size: 0.75em;'>{subtext}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with f_col1:
                    metric_box("ROIC (Est)", f"{fund_data['roic']*100:.1f}%", "Return on Capital", "#1a4d2e" if fund_data['roic'] > 0.15 else "#1e3a5f")
                with f_col2:
                    metric_box("Op Margin", f"{fund_data['operating_margin']*100:.1f}%", "Operational Efficiency")
                with f_col3:
                    metric_box("Free Cash Flow", format_large_number(fund_data['fcf']), "Cash Generation")
                with f_col4:
                    metric_box("ROE", f"{fund_data['roe']*100:.1f}%", "Return on Equity")
                
                # Row 2: Valuation & Growth
                st.markdown("")
                v_col1, v_col2, v_col3, v_col4 = st.columns(4)
                
                with v_col1:
                    metric_box("P/E Ratio", f"{fund_data['pe_ratio']:.1f}", "Trailing 12m")
                with v_col2:
                    metric_box("Rev Growth", f"{fund_data['revenue_growth']*100:.1f}%", "Year over Year", "#1a4d2e" if fund_data['revenue_growth'] > 0.1 else "#1e3a5f")
                with v_col3:
                    metric_box("Total Cash", format_large_number(fund_data['total_cash']), "Balance Sheet")
                with v_col4:
                    metric_box("Total Debt", format_large_number(fund_data['total_debt']), "Balance Sheet")
                    
            else:
                st.warning("Could not fetch fundamental data.")



        elif not data.empty:
            st.warning(f"Insufficient historical data for {ticker} (need at least 200 days).")
            
            # --- OPTIONS INTELLIGENCE TAB MOVED ---
    else:
        st.info("Enter a ticker in the sidebar to begin analysis")


# --- TAB 6: OPTIONS INTELLIGENCE ---
with tab6:
    st.title("🌪️ Options Flow & Gamma Profile")
    
    if ticker:
         # --- ΓAMMA & VOLUME PROFILE ---
        st.subheader(f"📊 Gamma Profile: {ticker}")
        
        with st.spinner(f"Fetching options data for {ticker}..."):
            track_api_call()  # Track options chain fetch
            gamma_data = get_gamma_profile(ticker)
        
        if 'error' in gamma_data:
            st.warning(f"⚠️ {gamma_data['error']}")
            st.caption("Note: Gamma profile requires active options trading on the ticker")
        else:
            # Key metrics
            stats = gamma_data['stats']
            spot = gamma_data['spot']
            
            # Metrics row with better visibility
            st.markdown("### Options Metrics")
            gm1, gm2, gm3, gm4 = st.columns(4)
            
            # Custom styled metrics for better visibility
            net_gex_val = stats['net_gex'] / 1e6
            gex_color_bg = "#1a4d2e" if net_gex_val > 0 else "#4d1a1a"
            gex_emoji = "🟢" if net_gex_val > 0 else "🔴"
            
            gm1.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Spot Price</p>
                <h2 style='color: white; margin: 5px 0;'>${spot:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            gm2.markdown(f"""
            <div style='background-color: {gex_color_bg}; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Net GEX</p>
                <h2 style='color: white; margin: 5px 0;'>{gex_emoji} ${net_gex_val:.1f}M</h2>
            </div>
            """, unsafe_allow_html=True)
            
            
            # Fix max GEX strike display
            max_gex_display = f"${stats['max_gex_strike']:.2f}" if stats['max_gex_strike'] is not None else "N/A"
            
            gm3.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Max GEX Strike</p>
                <h2 style='color: white; margin: 5px 0;'>{max_gex_display}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            put_call_ratio = stats['total_put_volume'] / max(stats['total_call_volume'], 1)
            gm4.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Put/Call Vol</p>
                <h2 style='color: white; margin: 5px 0;'>{put_call_ratio:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Last updated: {gamma_data['timestamp']}")
            
            st.divider()
            
            # Enhanced interpretation with analysis
            if net_gex_val > 0:
                st.info("""📌 **Positive Net GEX**: Dealers are long gamma → **Stabilizing Environment**
                
**What this means:**
- Market makers will buy dips and sell rallies to maintain delta-neutral hedges
- Price tends to revert toward high-GEX strikes (acts as magnet)
- Lower volatility expected as hedging dampens price swings
- Breakouts require stronger momentum to overcome dealer resistance""")
            else:
                st.warning("""📌 **Negative Net GEX**: Dealers are short gamma → **Volatizing Environment**
                
**What this means:**
- Market makers will sell dips and buy rallies (amplifying moves)
- Price can accelerate through strikes with negative GEX
- Higher volatility expected as hedging exacerbates price swings  
- Trends can develop more easily without dealer resistance""")
            
            # Additional analysis
            st.markdown("""**Key Observations:**""")
            analysis_points = []
            
            if stats['max_gex_strike']:
                distance_to_max = ((stats['max_gex_strike'] - spot) / spot) * 100
                analysis_points.append(f"• Max GEX at ${stats['max_gex_strike']:.2f} ({distance_to_max:+.1f}% from spot) - this level acts as a price magnet")
            
            if put_call_ratio > 1.5:
                analysis_points.append(f"• High put/call ratio ({put_call_ratio:.2f}) suggests defensive positioning or bearish sentiment")
            
            for point in analysis_points:
                st.markdown(point)
                
            # Charts
            col_gamma, col_vol = st.columns(2)
            
            with col_gamma:
                st.subheader("Gamma Exposure by Strike")
                gex = gamma_data['gex']
                fig_gamma = go.Figure()
                fig_gamma.add_trace(go.Bar(
                    x=gex.index,
                    y=gex.values / 1e6,
                    name='Total Gamma',
                    marker_color=['#4ade80' if val > 0 else '#ef4444' for val in gex.values]
                ))
                fig_gamma.update_layout(template="plotly_dark", height=400)
                # Update axis titles
                fig_gamma.update_layout(showlegend=False)
                fig_gamma.update_yaxes(title_text="Gamma ($M)")
                fig_gamma.update_xaxes(title_text="Strike Price ($)")
                
                # Add spot line
                fig_gamma.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="Spot")
                
                st.plotly_chart(fig_gamma, width="stretch")
            
            with col_vol:
                st.subheader("Volume Profile")
                vol = gamma_data['volume']
                fig_vol = go.Figure()
                if 'call' in vol.columns:
                    fig_vol.add_trace(go.Bar(
                        x=vol.index,
                        y=vol['call'],
                        name='Call Vol',
                        marker_color='#4ade80'
                    ))
                if 'put' in vol.columns:
                    fig_vol.add_trace(go.Bar(
                        x=vol.index,
                        y=vol['put'],
                        name='Put Vol',
                        marker_color='#ef4444'
                    ))
                fig_vol.update_layout(template="plotly_dark", height=400, barmode='stack')
                fig_vol.update_yaxes(title_text="Volume")
                fig_vol.update_xaxes(title_text="Strike Price ($)")
                fig_vol.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="Spot")
                
                st.plotly_chart(fig_vol, width="stretch")

        
        # --- OPTIONS FLOW ANALYSIS ---
        st.divider()
        st.subheader(f"🌊 Options Flow Analysis: {ticker}")
        
        with st.spinner("Fetching daily flow data..."):
            flow_data = get_daily_flow_snapshot(ticker)
        
        if not flow_data:
            st.warning("No flow data available.")
        else:
            # Metrics
            f1, f2, f3 = st.columns(3)
            
            # Net Premium Color
            net_prem = flow_data['net_premium']
            prem_color = "#1a4d2e" if net_prem > 0 else "#4d1a1a"
            prem_emoji = "🐂" if net_prem > 0 else "🐻"
            
            f1.markdown(f"""
            <div style='background-color: {prem_color}; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Net Premium</p>
                <h2 style='color: white; margin: 5px 0;'>{prem_emoji} ${net_prem:,.0f}</h2>
                <p style='color: #d1d5db; font-size: 0.8em; margin: 0;'>Diff: Call $ - Put $</p>
            </div>
            """, unsafe_allow_html=True)
            
            f2.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>P/C Premium Ratio</p>
                <h2 style='color: white; margin: 5px 0;'>{flow_data['pc_premium_ratio']:.2f}</h2>
                <p style='color: #d1d5db; font-size: 0.8em; margin: 0;'>Bearish > 1.0</p>
            </div>
            """, unsafe_allow_html=True)
            
            f3.markdown(f"""
            <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Unusual Contracts</p>
                <h2 style='color: white; margin: 5px 0;'>{len(flow_data['unusual_calls']) + len(flow_data['unusual_puts'])}</h2>
                <p style='color: #d1d5db; font-size: 0.8em; margin: 0;'>Vol > 1.5x OI</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sentiment Logic
            sentiment = analyze_flow_sentiment(flow_data)
            st.info(f"**Flow Sentiment:** {sentiment}")
            
            st.markdown("---")
            
            # Unusual Activity
            st.markdown("### 🔥 Unusual Activity (Vol > OI)")
            tab_calls, tab_puts = st.tabs(["Unusual Calls", "Unusual Puts"])
            
            with tab_calls:
                if not flow_data['unusual_calls'].empty:
                    unusual_calls_display = flow_data['unusual_calls'][['strike', 'expiration', 'volume', 'openInterest', 'premium']].copy()
                    unusual_calls_display['premium'] = unusual_calls_display['premium'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(unusual_calls_display, width="stretch", hide_index=True)
                else:
                    st.caption("None detected")
            
            with tab_puts:
                if not flow_data['unusual_puts'].empty:
                    unusual_puts_display = flow_data['unusual_puts'][['strike', 'expiration', 'volume', 'openInterest', 'premium']].copy()
                    unusual_puts_display['premium'] = unusual_puts_display['premium'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(unusual_puts_display, width="stretch", hide_index=True)
                else:
                    st.caption("None detected")
            
            # Top Contracts
            with st.expander("View Top Contracts by Premium"):
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Top Calls")
                    if not flow_data['top_calls'].empty:
                        top_calls_display = flow_data['top_calls'].copy()
                        top_calls_display['premium'] = top_calls_display['premium'].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(top_calls_display, width="stretch", hide_index=True)
                    else:
                        st.caption("No data")
                
                with c2:
                    st.subheader("Top Puts")
                    if not flow_data['top_puts'].empty:
                        top_puts_display = flow_data['top_puts'].copy()
                        top_puts_display['premium'] = top_puts_display['premium'].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(top_puts_display, width="stretch", hide_index=True)
                    else:
                        st.caption("No data")
                
            st.caption("""
            **Educational Note:** 
            - **Net Premium**: The total premium spent on Calls minus Puts. Positive = Bullish flow.
            - **Unusual Activity**: Contracts where today's volume is significantly higher than existing Open Interest, suggesting fresh positioning.
            - *Note: This is retail-access data (delayed/aggregated) and not comparable to institutional feeds like Bloomberg.*
            """)
    else:
        st.info("Enter a ticker in the sidebar to view options intelligence.")
with tab4:
    st.title("🏛️ Congressional Trading Tracker")
    
    # API Verification
    api_key = st.session_state.get('user_congress_key') or st.secrets.get("congress_api_key")
    
    if not api_key:
        st.warning("⚠️ **API Config Required:** Please enter your [Congress.gov API Key](https://api.congress.gov/sign-up/) in the sidebar settings ⚙️ to enable this tab.")
        st.stop()
        
    st.markdown("""
    Track stock trades disclosed by members of Congress under the STOCK Act.
    Congress members must disclose trades within 45 days of execution.
    """)
    
    # Verify connectivity (Visual indicator)
    with st.spinner("Connecting to Congress.gov..."):
        members = fetch_congress_members(api_key=api_key)
        if members.empty:
            st.error("❌ Invalid API Key or Connection Failed. Please check your settings.")
        else:
            st.caption(f"✅ Connected: Tracking {len(members)} active members")
    
    with st.spinner("🏛️ Loading Congressional trades..."):
        track_api_call()  # Track Congress API call
        trades_df = fetch_stock_disclosures()
    
    if trades_df.empty:
        st.error("Could not fetch Congressional trading data")
    else:
        # Summary metrics
        st.subheader("Recent Activity Summary")
        
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        
        purchases = len(trades_df[trades_df['transaction'] == 'Purchase'])
        sales = len(trades_df[trades_df['transaction'] == 'Sale'])
        unique_tickers = trades_df['ticker'].nunique()
        unique_traders = trades_df['member'].nunique()
        
        sum_col1.markdown(f"""
        <div style='background-color: #1a4d2e; padding: 15px; border-radius: 8px; text-align: center;'>
            <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Purchases</p>
            <h2 style='color: white; margin: 5px 0;'>{purchases}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        sum_col2.markdown(f"""
        <div style='background-color: #4d1a1a; padding: 15px; border-radius: 8px; text-align: center;'>
            <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Sales</p>
            <h2 style='color: white; margin: 5px 0;'>{sales}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        sum_col3.markdown(f"""
        <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
            <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Unique Tickers</p>
            <h2 style='color: white; margin: 5px 0;'>{unique_tickers}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        sum_col4.markdown(f"""
        <div style='background-color: #1e3a5f; padding: 15px; border-radius: 8px; text-align: center;'>
            <p style='color: #94a3b8; margin: 0; font-size: 0.9em;'>Active Traders</p>
            <h2 style='color: white; margin: 5px 0;'>{unique_traders}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Two columns: Recent Trades and Top Tickers
        trade_col, ticker_col = st.columns([2, 1])
        
        with trade_col:
            st.subheader("Recent Trades")
            
            # Style the trades table
            display_trades = trades_df[['date', 'member', 'party', 'ticker', 'transaction', 'amount']].copy()
            display_trades.columns = ['Date', 'Member', 'Party', 'Ticker', 'Type', 'Amount']
            
            def color_transaction(val):
                if val == 'Purchase':
                    return 'background-color: #1a4d2e; color: white'
                elif val == 'Sale':
                    return 'background-color: #4d1a1a; color: white'
                return ''
            
            def color_party(val):
                if val == 'D':
                    return 'background-color: #1e3a5f; color: white'
                elif val == 'R':
                    return 'background-color: #5f1e1e; color: white'
                return ''
            
            styled_trades = display_trades.style.applymap(
                color_transaction, subset=['Type']
            ).applymap(
                color_party, subset=['Party']
            )
            
            st.dataframe(styled_trades, width="stretch", hide_index=True)
        
        with ticker_col:
            st.subheader("Most Traded Tickers")
            
            top_tickers = get_top_traded_tickers(trades_df, n=5)
            
            if not top_tickers.empty:
                for _, row in top_tickers.iterrows():
                    st.markdown(f"""
                    <div style='background-color: #1e3a5f; padding: 10px; border-radius: 5px; margin-bottom: 8px;'>
                        <h4 style='margin: 0; color: white;'>{row['ticker']}</h4>
                        <p style='margin: 0; color: #94a3b8; font-size: 0.85em;'>{row['trade_count']} trade(s)</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Active Traders
        st.subheader("Most Active Traders")
        
        active_traders = get_active_traders(trades_df, n=5)
        
        if not active_traders.empty:
            trader_cols = st.columns(len(active_traders))
            
            for col, (_, trader) in zip(trader_cols, active_traders.iterrows()):
                party_color = "#1e3a5f" if trader['party'] == 'D' else "#5f1e1e"
                with col:
                    st.markdown(f"""
                    <div style='background-color: {party_color}; padding: 15px; border-radius: 8px; text-align: center;'>
                        <h4 style='margin: 0; color: white; font-size: 0.9em;'>{trader['member']}</h4>
                        <p style='margin: 5px 0; color: #94a3b8;'>{trader['party']} - {trader['state']}</p>
                        <h3 style='margin: 0; color: #4ade80;'>{trader['trade_count']} trades</h3>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Watchlist Overlap
        st.subheader("Check Your Watchlist")
        
        watchlist_input = st.text_input(
            "Enter tickers (comma-separated):",
            placeholder="e.g., NVDA, AAPL, TSLA, META"
        )
        
        if watchlist_input:
            watchlist = [t.strip().upper() for t in watchlist_input.split(',')]
            overlap = check_watchlist_overlap(trades_df, watchlist)
            
            if not overlap.empty:
                st.success(f"Found {len(overlap)} Congressional trade(s) matching your watchlist!")
                st.dataframe(overlap[['date', 'member', 'party', 'ticker', 'transaction', 'amount']], 
                            width="stretch", hide_index=True)
            else:
                st.info("No Congressional trades match your watchlist tickers.")
        
        # Disclaimer
        st.divider()
        st.caption("""
        **Disclaimer:** Congressional trading data is subject to 45-day disclosure delays. 
        This information is for educational purposes only and should not be considered investment advice.
        Past Congressional trades do not guarantee future performance.
        """)
