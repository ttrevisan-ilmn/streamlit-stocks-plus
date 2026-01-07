# Mphinancial Stock Analysis Terminal

A comprehensive quantitative research platform built with Streamlit, integrating technical analysis, sector rotation models, options flow intelligence, and alternative data.

## 🚀 Key Features

The terminal is organized into five specialized modules:

### 1. 📊 Market Health (Asbury 6)
A quantitative daily gauge of US equity market internal strength.
- **Six Metrics**: Breadth, Volume, Relative Performance, Asset Flows, VIX, Price Momentum.
- **Signals**: BUY (Green), NEUTRAL (Yellow), CASH (Red).
- **History**: Interactive charts tracking signal evolution against SPX performance.

### 2. 📈 Sector Rotation (SEAF Model)
The **Sector ETF Asset Flows (SEAF)** model ranks all 11 Select Sector SPDR ETFs.
- **Methodology**: Ranks sectors by "Money Flow" (Volume-Weighted Momentum + Relative Strength) across 4 timeframes (20d, 60d, 120d, 252d).
- **Actionable**: Highlights the **Top 3 Sectors** for immediate allocation.
- **Visualization**: Correlation heatmaps and score rankings.

### 3. 🌐 Intermarket Analysis
Macro-level context for equity decisions.
- **Yield Curve**: Visualizes 10Y-3M spread to detect recession warnings (Contango vs Backwardation).
- **Commodities & Forex**: Dashboard for Oil, Gold, Bitcoin, and DXY.
- **Asset Performance**: Normalized comparison of major asset classes over the last year.

### 4. 📉 Stock Analysis (The Core IO)
A deep-dive scanner for individual tickers.
- **Mphinancial Verdict**: Technical audit (EMAs, SMA200, ADX, ATR Key Levels) to identify "High Quality Setups".
- **Fundamental Health**: ROIC-style dashboard (ROIC, FCF, Margins, Valuation) powered by yfinance.
- **Gamma & Volume Profile**: Analysis of options dealer positioning (GEX) and support/resistance levels.
- **Options Flow**: Net premium tracking, unusual activity scanner, and Put/Call ratios.

### 5. 🏛️ Congressional Trading
Track the "Smart Money" in Washington using the STOCK Act disclosures.
- **Recent Trades**: Sortable table of latest filings by House/Senate members.
- **Leaderboard**: Top traded tickers and most active politicians.
- **Watchlist Check**: Automatically flags if your watchlist stocks are being bought/sold by Congress.

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ttrevisan-ilmn/streamlit-stocks-plus.git
   cd streamlit-stocks-plus
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets** (Two Options)

   **Option A: Sidebar Input (Easiest)**
   Simply run the app. If no secrets are found, a "⚙️ Settings" menu will appear in the sidebar where you can paste your API key.

   **Option B: secrets.toml (Persistent)**
   To avoid entering the key every time, create a file at `.streamlit/secrets.toml`:
   ```toml
   # .streamlit/secrets.toml
   congress_api_key = "YOUR_CONGRESS_GOV_API_KEY"
   ```
   *(Note: You can get a free key at [api.congress.gov](https://api.congress.gov))*

4. **Run the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🔒 Privacy & Data
- **Local Persistence**: User watchlists and total API usage stats are stored locally in `api_stats.json`.
- **No External Tracking**: All data fetching happens directly from your machine to the data providers (Yahoo Finance, Congress.gov).

## ⚠️ Disclaimer
This tool is for educational and research purposes only. It is not financial advice. 
