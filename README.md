# Live-Stock-Data-Strategy-Backtesting

Backtesting is the process of simulating the performance of a trading strategy using historical data. It allows traders and investors to evaluate the potential performance of a strategy before actually implementing it in the real market. By backtesting a strategy, traders can identify any potential weaknesses or flaws, and make adjustments to improve its performance.

Backtesting is an important tool for traders because it can help them to make more informed decisions about which strategies to use and how to manage risk. It allows traders to assess the potential profitability of a strategy, as well as how it would have performed under different market conditions. Additionally, backtesting can also be used to optimize the parameters of a strategy, such as when to enter or exit a trade.

Moreover, backtesting allows traders to check the robustness of the strategy by looking at different market conditions and historical periods, thus making it possible to evaluate its ability to perform in different scenarios.

In short, backtesting is an essential tool for traders to evaluate and improve their strategies, and make more informed decisions about when to enter and exit trades.

## Installation

### Prerequisites
- Python 3.8+ 
- pip (Python package manager)

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd Backtesting-Engine
```

2. Create and activate a virtual environment:
```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Development (Interactive Mode)
```bash
# Linux/Mac with protobuf workaround
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run final.py

# Or simply (if protobuf is properly configured)
streamlit run final.py
```

The app will open at `http://localhost:8501`

### Usage
1. Select country (US or Indian stocks)
2. Choose a stock ticker
3. Select timeframe and period for historical data
4. Pick a trading strategy from the sidebar
5. Adjust strategy parameters using sliders
6. View backtest results, charts, and performance metrics

## Available Strategies

- **Moving Average Crossover**: Trend-following strategy using fast and slow moving averages
- **Bollinger Bands**: Mean-reversion strategy based on price bands and RSI
- **Bollinger Bands with Fibonacci**: Combines Bollinger Bands with Fibonacci retracement levels
- **4 EMA Crossover**: Multi-EMA momentum strategy with stop loss
- **MACD**: Trend identification using MACD crossovers

## Troubleshooting

### Protobuf Error
If you encounter `TypeError: Descriptors cannot be created directly`, use the protobuf workaround:
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run final.py
```

Or downgrade protobuf:
```bash
pip install 'protobuf==3.20.0'
```

## Project Structure
- `final.py`: Main Streamlit application with all strategies and indicators
- `requirements.txt`: Python package dependencies
- `.github/copilot-instructions.md`: AI coding guidelines for this project
- `.gitignore`: Git ignore rules

## Notes
- Backtesting results are not indicative of future performance
- Real trading includes slippage, commissions, and liquidity constraints
- Always validate strategies with paper trading before live deployment
