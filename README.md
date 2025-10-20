# Trading Bot Project

This project implements a trading bot for Bitcoin using historical trading data. The bot utilizes machine learning models and scalp trading strategies to make informed trading decisions. The project includes various components for data loading, feature engineering, strategy implementation, and performance evaluation.

## Project Structure

```
trading-bot-project
├── data
│   └── raw
│       └── bitcoin.csv          # Raw Bitcoin trading data
├── notebooks
│   └── exploration.ipynb        # Jupyter notebook for exploratory data analysis
├── src
│   ├── __init__.py              # Marks the src directory as a Python package
│   ├── config.py                # Configuration settings for the project
│   ├── data_loader.py           # Functions to load and preprocess data
│   ├── features
│   │   ├── __init__.py          # Marks the features directory as a Python package
│   │   └── feature_engineering.py # Functions for feature engineering
│   ├── models
│   │   ├── __init__.py          # Marks the models directory as a Python package
│   │   └── ml_model.py          # Machine learning model for trading decisions
│   ├── strategies
│   │   ├── __init__.py          # Marks the strategies directory as a Python package
│   │   └── scalp_strategy.py     # Implementation of the scalp trading strategy
│   ├── trading
│   │   ├── __init__.py          # Marks the trading directory as a Python package
│   │   └── engine.py            # Main trading engine for executing trades
│   └── evaluation
│       ├── __init__.py          # Marks the evaluation directory as a Python package
│       └── performance_metrics.py # Functions to calculate performance metrics
├── tests
│   ├── __init__.py              # Marks the tests directory as a Python package
│   └── test_performance.py       # Unit tests for performance metrics functions
├── requirements.txt              # Lists project dependencies
└── README.md                     # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd trading-bot-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the data:
   - Place your raw Bitcoin trading data in the `data/raw/` directory as `bitcoin.csv`.

4. Run the Jupyter notebook for exploratory data analysis:
   ```
   jupyter notebook notebooks/exploration.ipynb
   ```

## Usage

- The bot uses historical Bitcoin data to calculate various trading indicators and metrics.
- The machine learning model is trained on features derived from the data to make trading decisions.
- The scalp trading strategy is implemented to execute trades based on short-term price movements.
- Performance metrics such as cumulative return, Sharpe ratio, drawdowns, and Sortino ratio are calculated to evaluate the bot's performance.

## Recommendations

- Consider adjusting the moving average durations for `ma_long` and `ma_short` based on backtesting results to optimize trading performance.
- Continuously monitor and refine the machine learning model and trading strategy based on market conditions and performance metrics.