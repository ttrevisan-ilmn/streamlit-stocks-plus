The Mphinancial Terminal is a specialized trading tool built with Streamlit to perform a "Tao Audit" on stock tickers. It uses a specific set of mechanical criteria to identify high quality trend-following setups.

![Mphinancial Terminal Interface](Screenshot%202026-01-02%208.30.26%20PM.png)
The visual interface showing a High Quality Setup audit for GOOGL.

## ⚙️ The Mphinancial Engine
The application performs several quantitative calculations to evaluate the current state of a stock:

- The EMA Stack: Calculates five Exponential Moving Averages (8, 21, 34, 55, and 89) to determine trend alignment.
- The Wind (200 SMA): Uses the 200-period Simple Moving Average as the primary trend filter.
- ADX (Trend Strength): A custom manual calculation of the Average Directional Index to ensure compatibility across various systems, including Chromebooks.
- ATR (Volatility): Measures the Average True Range to define the current "Buy Zone" relative to price action.

##  🔍 Mechanical Audit Criteria
The terminal evaluates every ticker against four primary mechanical checks to provide a final verdict:

- Trend Filter: The current price must be above the 200 SMA (Sailing with the Wind).
- Bullish Stack: All five EMAs must be in perfect numerical alignment (8 > 21 > 34 > 55 > 89) to confirm momentum.
- The Buy Zone: The current price must be within 1 ATR of the 21 EMA to avoid overextension.
- ADX Strength: The ADX value must be at or above 20 to signify a trending market rather than a ranging one.

When all these criteria align, the terminal signals a High Quality Setup.

## 🛠️ Installation and Setup

**Dependencies**

The project relies on the following Python libraries:
- Streamlit: For the web interface and dashboard.
- Yfinance: To fetch historical market data.
- Pandas and Numpy: For data manipulation and mathematical calculations.
 -Plotly: For interactive financial charting and candlestick visualization.

**Running the App**

1. Ensure you have Python installed on your system.
2. Install the required packages: pip install -r requirements.txt.
3. Launch the terminal: ```streamlit run streamlit_app.py```
4. Enter Ticker

Enter a ticker symbol in the sidebar to begin an audit.
