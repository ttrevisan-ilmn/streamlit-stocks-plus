import streamlit as st
from screener_engine import get_screener_universe, fetch_screener_data, apply_strategy

def render_screener():
    st.header("🔍 Intelligent Stock Screener")
    st.caption("Screening the S&P 500 universe using real-time data.")
    
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    
    with col_s1:
        strategy = st.selectbox("Select Strategy Preset", [
            "Cash Secured Puts (CSP)",
            "Covered Calls (CC)",
            "Short Momentum",
            "Mid Momentum",
            "Safe Long",
            "Ultimate Stacked Bulls",
            "Day Trade Runners",
            "Navellier A-Rated Growth"
        ])
        
        # Add Quick Scan Option
        quick_scan = st.checkbox("⚡ Quick Scan (Top 40 only)", value=True, help="Scan only the first 40 tickers for speed. Uncheck to scan full S&P 500.")
    
    if st.button("Run Screener", key="run_screener"):
        with st.spinner(f"Scanning {'Top 40' if quick_scan else 'S&P 500'} for {strategy} candidates..."):
            # 1. Get Universe
            tickers = get_screener_universe()
            
            # 2. Fetch Data (Cached)
            limit = 40 if quick_scan else None
            data = fetch_screener_data(tickers, limit=limit)
            
            # 3. Apply Logic
            results = apply_strategy(data, strategy)
            
            if not results.empty:
                st.success(f"Found {len(results)} candidates for {strategy}.")
                
                # Clean up raw data before display
                display_cols = ['Price', 'MarketCap', 'Beta', 'PE', 'RSI', 'HV_20',
                                'Change%', 'ADX', 'StochK', 'ROE', 'OpMargin',
                                'RevGrowth', 'EarnGrowth', 'EQGrowth', 'AnalystRec',
                                'DivYield', 'SMA50', 'Float', 'Volume']
                # Only show columns that actually exist in results
                show_cols = [c for c in display_cols if c in results.columns]
                display_df = results[show_cols].copy()
                
                fmt = {
                    'Price':      '${:.2f}',
                    'MarketCap':  '${:,.0f}',
                    'Beta':       '{:.2f}',
                    'PE':         '{:.1f}',
                    'RSI':        '{:.1f}',
                    'SMA50':      '${:.2f}',
                    'HV_20':      '{:.1f}%',
                    'Float':      '{:,.0f}',
                    'Volume':     '{:,.0f}',
                    'Change%':    '{:+.2f}%',
                    'ADX':        '{:.1f}',
                    'StochK':     '{:.1f}',
                    'ROE':        '{:.1%}',
                    'OpMargin':   '{:.1%}',
                    'RevGrowth':  '{:+.1%}',
                    'EarnGrowth': '{:+.1%}',
                    'EQGrowth':   '{:+.1%}',
                    'AnalystRec': '{:.1f}',
                    'DivYield':   '{:.2%}',
                }
                active_fmt = {k: v for k, v in fmt.items() if k in display_df.columns}
                
                st.dataframe(
                    display_df.style.format(active_fmt, na_rep="—"),
                    use_container_width=True,
                    height=500
                )
                
                st.info("💡 **Tip**: Type a ticker from the list into the sidebar search to analyze it further.")
                
            else:
                st.warning("No candidates found matching criteria.")
