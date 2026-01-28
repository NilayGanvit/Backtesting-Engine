# Backtesting Engine AI Instructions

## Project Overview
Live-Stock-Data-Strategy-Backtesting is a Streamlit-based web application for simulating trading strategies using historical stock data. It evaluates strategy performance without real market risk by replaying historical price movements.

## Architecture & Data Flow

### Core Components
- **Data Acquisition** ([final.py#L1-L40](final.py#L1-L40)): Uses `yfinance` to fetch OHLC data for US/Indian stocks across multiple timeframes (1m to 10000d periods)
- **Technical Indicators** ([final.py#L111-L135](final.py#L111-L135)): Helper functions compute MA, EMA, RSI, Stochastic, Momentum, ROC
- **Strategy Implementations** ([final.py#L360-L800](final.py#L360-L800)): Four trading strategies—each generates entry/exit signals and calculates performance metrics
- **Performance Metrics**: Standardized calculations (Sharpe ratio, max drawdown, profit factor, etc.) used across all strategies
- **UI/Visualization** ([final.py#L1000-1090](final.py#L1000-1090)): Streamlit sidebar controls parameters; matplotlib displays price charts and signal overlays

### Strategy Pattern
All strategies follow this pattern:
1. Calculate technical indicators from price data
2. Iterate through historical data generating buy/exit signals via flag logic (`flag=0` = no position, `flag=1` = holding)
3. Collect entry prices (`bl`), exit prices (`sl`), quantities (`q`)
4. Calculate P&L for each closed trade
5. Compute metrics and render charts

## Key Patterns & Conventions

### Trade Data Structure
- `bl`: Buy/entry prices list
- `sl`: Sell/exit prices list  
- `q`: Share quantities purchased (`q.append(init_cash/buy)` = position size based on capital)
- P&L calculated as `(sl[i]-bl[i])*q[i]` per trade

### Metrics Functions
- Located at [final.py#L135-L230](final.py#L135-L230)
- **Input**: Lists of entry prices, exit prices, quantities
- **No side-effects**: Pure calculations; don't modify dataframes
- Example: `maxprofit(sl, bl, q)` iterates pairs, returns max single-trade profit

### Visualizations
- **3-subplot strategy plots**: Price + indicators (top), buy/sell markers, signal overlay
- Color conventions: green=buy signals (^), red=sell signals (v)
- Plotting libraries used: matplotlib (strategy), st.pyplot() (display)

## Developer Workflows

### Running the App
```bash
streamlit run final.py
```
Select country → stock → timeframe/period → choose strategy from sidebar → adjust parameters

### Adding a New Strategy
1. Create function `def new_strategy(data, param1, param2, init_cash):`
2. Calculate indicators: `data['indicator'] = ...`
3. Implement entry/exit logic with `flag` pattern, populate `bl`, `sl`, `q` lists
4. Call metrics functions (reuse `maxprofit`, `sharpe_ratio`, etc.)
5. Create matplotlib figure, call `st.pyplot(fig=fig, clear_figure=None)`
6. Add to `activities` list and `if choice==` block in `main()` ([final.py#L1000](final.py#L1000))
7. Add `st.sidebar.slider()` widgets for parameters

### Common Issues & Solutions
- **MultiIndex columns after download**: Use `data = data.reset_index(drop=False)` and flatten
- **NaN values in indicators**: Functions handle via `min_periods` parameter (e.g., `rolling(n, min_periods=n).mean()`)
- **Quantities mismatch**: `df = df[:len(sl)]` truncates to shortest list (handles edge cases)
- **Negative risk-free rates**: `sharpe_ratio()` default is 0.0; adjust if needed

## Critical Integration Points

### External Dependencies
- `yfinance`: Downloads OHLC data; handles multiple markets (tickers like "AAPL" vs "TCS.NS")
- `pandas`: Core data structure (rolling windows, ewm calculations)
- `streamlit`: UI framework; note state reset on each interaction
- `matplotlib`: Charting; Streamlit requires `clear_figure=None` to prevent duplicate renders

### Data Assumptions
- Downloaded data has columns: `Open`, `High`, `Low`, `Close`, `Volume` (after `auto_adjust=True`)
- Index is datetime; used for chart labels and signal timing
- Gaps in data are acceptable; rolling windows skip NaN values

## Performance Optimization Tips
- RSI calculation ([final.py#L96-L102](final.py#L96-L102)) is slow for large datasets; uses loop-based exponential smoothing
- For large backtests (10000+ days), consider downsampling or caching yfinance downloads
- Streamlit reruns entire script on interaction; strategy computation happens client-side

## Files to Edit When Extending
- `final.py`: Add strategy function, metrics function, or UI control
- `requirements.txt`: Add dependencies if using new libraries
- No configuration files or environment variables used

## Testing Strategies
- Verify buy/sell counts match: `len(bl) ≈ len(sl)` (last trade may not close)
- Check P&L logic: `total = sum([(sl[i]-bl[i])*q[i] for i in range(len(bl))])`
- Compare against manual calculation: Select simple data subset, trace flag transitions
- Validate metrics: Sharpe ratio should reflect volatility; max drawdown should show trough-to-peak dips

## Parameter Tuning Best Practices

### Moving Average Crossover
- `fast_period`: 20-50 (default 50) - responds quickly to price changes
- `slow_period`: 100-250 (default 200) - captures longer-term trends
- **Rule**: `slow > fast * 2` typically improves signal quality
- Faster MA pairs = more trades, higher drawdown; slower = fewer trades, lag risk

### Bollinger Bands
- `period`: 15-30 (default 20) - lookback window
- `stdDev`: 1.5-3.0 (default 2) - bands distance; higher = fewer signals
- `RSI_oversold`: 20-40 (default 40) - entry threshold
- `RSI_overbought`: 60-80 (default 60) - exit threshold

### EMA Crossover (4-series)
- `fast1 < fast2 < fast3 < slow` strictly enforced by sliders
- Typical: 8 / 13 / 21 / 55 (Fibonacci-inspired)
- Alternative aggressive: 5 / 10 / 20 / 50
- Stop loss included: exit if price < 0.85 × entry (15% loss)

### Bollinger Bands + Fibonacci
- `BB_length`: 15-25 (default 20)
- `stdDev`: 1.5-3.0 (default 2)
- Fibonacci levels auto-calculated from min/max over full dataset
- Entry: Close < lower BB AND Close < 23.6% retracement
- Exit: Close > upper BB OR Close > 61.8% retracement

### MACD
- `fast_period`: 10-14 (default 12)
- `slow_period`: 24-30 (default 26)
- `smooth`: 7-11 (default 9)
- Signal generated on MACD-signal line crossover

### Initial Cash
- Range: 1,000 to 10,000,000
- Affects position sizing: `qty = init_cash / entry_price`
- Larger capital = smaller position sizes, lower volatility per trade

## Strategy Comparison & Selection

### Backtesting Metrics Hierarchy
1. **Primary**: Total Profit % (must be positive)
2. **Risk**: Max Drawdown (< 20% is good, < 10% is excellent)
3. **Consistency**: Sharpe Ratio (> 1.0 is acceptable, > 2.0 is strong)
4. **Win Rate**: `(max_consecutive_profit / total_trades) * 100` (> 50% is profitable)
5. **Risk/Reward**: Profit Factor = Total Profit / Total Loss (> 1.5 is strong, > 2.0 is excellent)

### Market Condition Selection
- **Trending Markets**: Favor MA Crossover, EMA 4-series (momentum-based)
- **Mean-Reverting Markets**: Favor Bollinger Bands, BB+Fibonacci (range-bound)
- **Volatile Markets**: MACD (smoother signals, fewer false breakouts)
- **Uncertain**: Start with 1h timeframe, 60-90 days data to reduce noise

### Timeframe Implications
- **1m to 5m**: High-frequency, many trades, high slippage risk
- **15m to 1h**: Moderate frequency, medium noise, best for parameter testing
- **1d to 1wk**: Swingtrading, fewer signals, calendar-based events matter
- **Rule**: Test parameters on 1h first, validate on 1d data

## Deployment & Production Considerations

### Prerequisites
- Python 3.8+ required
- Install all dependencies: `pip install -r requirements.txt`
- Note: `autocalc` and `ipywidgets` imports unused (legacy Jupyter code); can be removed

### Running in Production
```bash
# Development (interactive)
streamlit run final.py

# Production (non-interactive, headless not supported by Streamlit)
# Consider running in Docker or systemd service with persistent Streamlit cache
streamlit run final.py --logger.level=error --client.showErrorDetails=false
```

### Data Caching Optimization
- yfinance downloads happen on each stock/interval change (can be slow)
- **Solution**: Add `@st.cache_data` decorator to `yf.download()` calls with 24h TTL
- Example: `@st.cache_data(ttl=86400) def fetch_data(ticker, period, interval):`
- Reduces redundant API calls by ~70% during development/testing

### Scalability Limitations
- Streamlit reruns entire script on UI interaction (inefficient for long backtests)
- Large datasets (10000+ days): Consider chunking or pre-computation
- Strategy computation is single-threaded; parallel testing requires separate architecture

### Risk Warnings for Users
- Backtesting is **not indicative of future performance** (overfitting risk)
- Real trading includes: slippage, commissions, gaps, liquidity constraints
- Paper trading validation recommended before live deployment
- No risk management: strategies can lose entire capital

## Code Quality & Technical Debt

### Known Issues
1. **Metric Calculation Duplication**: Each strategy function repeats 15+ metrics calls
   - **Refactor opportunity**: Extract into `calculate_all_metrics(bl, sl, q, data)` function
   
2. **State Management**: Streamlit reruns invalidate `flag` logic across interactions
   - **Risk**: Multi-session backtests may produce inconsistent results
   - **Mitigation**: Store trade history in session state if adding real-time features

3. **Error Handling**: No validation for:
   - Empty trade lists (`bl`, `sl`) → causes division by zero in Sharpe ratio
   - Invalid ticker symbols → crashes without user feedback
   - **Fix**: Add try-catch around yfinance downloads, check `len(bl) > 0` before metrics

4. **Unused Imports**: `autocalc`, `ipywidgets`, `pyopencl` (line ~950) not used
   - **Action**: Remove or document if planning future features

5. **DataFrame Edge Cases**:
   - Last trade may not close (`len(bl) > len(sl)`)
   - Current code truncates: `df = df[:len(sl)]` (works but silent data loss)
   - **Better**: Log warning if > 1 unclosed trade

### Performance Bottlenecks
- **RSI calculation** ([final.py#L96-L102](final.py#L96-L102)): Loop-based exponential smoothing, O(n²) worst case
  - For 10000+ candles: ~2-3 seconds
  - **Optimization**: Use pandas `.ewm()` method (10x faster)
  
- **DataFrame operations**: Multiple passes over data (indicators, signals, metrics)
  - **Optimization**: Consolidate into single loop for large datasets

### Refactoring Roadmap
- [ ] Extract metrics into reusable module
- [ ] Add comprehensive error handling and logging
- [ ] Implement caching for yfinance data
- [ ] Add session state persistence for reproducibility
- [ ] Create unit tests for indicator functions
- [ ] Document strategy logic in docstrings (currently missing)
