import streamlit as st
import pandas as pd
from power_gauge import calculate_power_gauge

def render_power_gauge(ticker):
    st.header(f"⚡ Power Gauge Rating: {ticker}")
    st.caption("A 20-Factor Model analyzing Financials, Earnings, Technicals, and Experts.")
    
    with st.expander("Methodology: Power Gauge Analysis"):
        st.markdown("""
        **20-Factor Weighted Model:**
        - **Financials**: Debt/Equity, ROE, Price/Book, FCF Yield.
        - **Earnings**: 5yr Growth, Estimates, EPS Surprises, Trend.
        - **Technicals**: Relative Strength, Price Trend, Moving Averages.
        - **Experts**: Insider Transactions, Short Interest, Analyst Ratings.
        """)
    
    # Check Session State First
    ad = st.session_state.get('analysis_data', {})
    gauge = None
    
    if ad.get('ticker') == ticker and ad.get('power_gauge'):
        gauge = ad['power_gauge']
    
    # Fallback Button (or "Run Analysis" if not triggered automatically)
    if not gauge:
        if st.button("Generate Power Report", key="run_power_gauge"):
            with st.spinner(f"Analyzing {ticker} across 20 data points..."):
                gauge = calculate_power_gauge(ticker)
    
    if gauge:
        # Top Level Result
        col_g1, col_g2 = st.columns([1, 2])
        
        with col_g1:
            st.metric("Power Rating", gauge['rating'], delta=f"{gauge['score']:.1f}/100")
            
            # Simple Gauge Visual (Progress Bar)
            st.progress(int(gauge['score']))
            if gauge['rating'] == "BULLISH":
                st.success("Strong Buy Signal")
            elif gauge['rating'] == "BEARISH":
                st.error("Avoid / Sell Signal")
            else:
                st.warning("Neutral / Hold")
                
        with col_g2:
            # Radar Chart or Bar Chart of Categories
            cat_df = pd.DataFrame.from_dict(gauge['categories'], orient='index', columns=['Score'])
            st.bar_chart(cat_df)
        
        st.divider()
        
        # Detailed Breakdown (4 Quadrants)
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        
        # Custom CSS for progress bars and layout
        st.markdown("""
        <style>
        .pg-card {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #334155;
        }
        .pg-title {
            color: #e2e8f0;
            font-size: 1.2rem;
            margin-bottom: 15px;
            font-weight: 600;
        }
        .pg-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .pg-label {
            color: #94a3b8;
            font-size: 0.9rem;
            width: 40%;
        }
        .pg-bar-container {
            width: 55%;
            background-color: #334155;
            height: 10px;
            border-radius: 5px;
            position: relative;
            overflow: hidden;
        }
        .pg-bar-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease-in-out;
        }
        
        /* Gradients based on score */
        .color-green { background: linear-gradient(90deg, #ef4444 0%, #eab308 50%, #22c55e 100%); }
        .color-yellow { background: linear-gradient(90deg, #ef4444 0%, #eab308 100%); }
        .color-red { background: linear-gradient(90deg, #ef4444 100%, #ef4444 100%); }
        
        .expert-highlight .pg-title { color: #38bdf8; }
        .expert-highlight { border: 1px solid #0ea5e9; box-shadow: 0 0 10px rgba(14, 165, 233, 0.2); }
        </style>
        """, unsafe_allow_html=True)
        
        def format_currency(val):
            if val is None or pd.isna(val): return "N/A"
            if val >= 1e9: return f"${val/1e9:.2f}b"
            if val >= 1e6: return f"${val/1e6:.2f}m"
            return f"${val:,.2f}"

        def format_num(val, decimals=2):
            if val is None or pd.isna(val): return "N/A"
            return f"{val:.{decimals}f}"

        def render_category(col, title, data, cat_name, md=None, highlight=False):
            highlight_class = "expert-highlight" if highlight else ""
            
            html = f"""
            <div class="pg-card {highlight_class}">
                <div class="pg-title">{title}</div>
            """
            
            for factor, score in data.items():
                if score >= 65: color_class = "color-green"
                elif score >= 35: color_class = "color-yellow"
                else: color_class = "color-red"
                
                label_display = factor
                if factor == "Chaikin Money Flow":
                    label_display = f'<span style="color: #22c55e; font-weight: bold;">{factor} (CMF)</span>'
                
                html += f"""
                <div class="pg-row">
                    <div class="pg-label">{label_display}</div>
                    <div class="pg-bar-container">
                        <div class="pg-bar-fill {color_class}" style="width: {score}%;"></div>
                    </div>
                </div>
                """
            
            # Additional tables for specific categories matching Chaikin style
            if cat_name == "Financials" and md:
                html += f"""
                <div style="margin-top:20px; font-size:0.85rem; color:#94a3b8; display:flex; justify-content:space-between; text-align:left;">
                    <div>
                        <div style="color:#60a5fa; margin-bottom:5px;">Assets & Liab</div>
                        <div>Curr Ratio: <span style="color:#fff">{format_num(md.get('currentRatio'))}</span></div>
                        <div>LT Debt/Eq: <span style="color:#fff">{format_num(md.get('debtToEquity'))}</span></div>
                        <div>Mkt Cap: <span style="color:#fff">{format_currency(md.get('marketCap'))}</span></div>
                        <div>Revenue: <span style="color:#fff">{format_currency(md.get('totalRevenue'))}</span></div>
                    </div>
                    <div>
                        <div style="color:#60a5fa; margin-bottom:5px;">Valuation</div>
                        <div>P/E: <span style="color:#fff">{format_num(md.get('trailingPE'))}</span></div>
                        <div>PEG: <span style="color:#fff">{format_num(md.get('pegRatio'))}</span></div>
                        <div>P/B: <span style="color:#fff">{format_num(md.get('priceToBook'))}</span></div>
                        <div>P/S: <span style="color:#fff">{format_num(md.get('priceToSales'))}</span></div>
                    </div>
                    <div>
                        <div style="color:#60a5fa; margin-bottom:5px;">Dividends</div>
                        <div>Yield: <span style="color:#fff">{format_num(md.get('dividendYield'))}%</span></div>
                    </div>
                </div>
                """
            elif cat_name == "Technicals" and md:
                vol_str = "More Volatile" if (md.get('beta') or 1) > 1 else "Less Volatile"
                html += f"""
                <div style="margin-top:20px; font-size:0.85rem; color:#94a3b8; display:flex; justify-content:space-between; text-align:left;">
                    <div>
                        <div style="color:#60a5fa; margin-bottom:5px;">Price Activity</div>
                        <div>52w High: <span style="color:#fff">{format_num(md.get('fiftyTwoWeekHigh'))}</span></div>
                        <div>52w Low: <span style="color:#fff">{format_num(md.get('fiftyTwoWeekLow'))}</span></div>
                    </div>
                    <div>
                        <div style="color:#60a5fa; margin-bottom:5px;">Price % Chg</div>
                        <div>4wk chg: <span style="color:#fff">{format_num(md.get('chg4wk'))}%</span></div>
                        <div>24wk chg: <span style="color:#fff">{format_num(md.get('chg24wk'))}%</span></div>
                    </div>
                    <div>
                        <div style="color:#60a5fa; margin-bottom:5px;">Volume</div>
                        <div>20d Avg: <span style="color:#fff">{format_num(md.get('avgVol20Day'), 0)}</span></div>
                        <div>90d Avg: <span style="color:#fff">{format_num(md.get('avgVol90Day'), 0)}</span></div>
                    </div>
                    <div>
                        <div style="color:#60a5fa; margin-bottom:5px;">Volatility</div>
                        <div>Beta: <span style="color:#fff">{format_num(md.get('beta'))}</span></div>
                        <div><span style="color:#fff">{vol_str}</span></div>
                    </div>
                </div>
                """

            html += "</div>"
            with col:
                st.markdown(html, unsafe_allow_html=True)
        
        md = gauge.get('metadata', {})
        render_category(c1, "💰 Financials", gauge['details']['Financials'], "Financials", md)
        render_category(c2, "📈 Earnings", gauge['details']['Earnings'], "Earnings", md)
        render_category(c3, "🛠️ Technicals", gauge['details']['Technicals'], "Technicals", md)
        render_category(c4, "🧠 Experts (The Secret Sauce)", gauge['details']['Experts'], "Experts", md, highlight=True)
    elif not gauge and ticker:
        st.warning(f"⚠️ Power Gauge data unavailable for {ticker}. Check connection or API limits.")
        if st.button("Retry Power Gauge", key="retry_power_gauge"):
             st.session_state.analysis_data = {'ticker': None} # Force reset
             st.rerun()
