# S&P 500 Next-Day Direction Forecaster

Quick project where I tried to guess if SPY closes higher tomorrow using a small logistic regression model. Thought it would be fun to see if simple price features can beat a plain buy-and-hold, then wrapped it in a FastAPI endpoint + GitHub Pages front-end.

## What it does (these bullets are AI generated)
- pulls daily SPY data from Yahoo Finance starting 2005
- builds a few lagged features (returns, RSI, overnight gap)
- trains logistic regression on pre-2019 data, keeps 2019–2021 for validation, 2022+ as holdout
- backtests a long-or-cash strategy with 2 bps trading costs and compares to buy-and-hold
- clamps user-selected start/end dates to the cached holdout window (2022-01-01 onward) so the equity curve always reflects out-of-sample data
- optional plot of equity curve + drawdown



## Requirements
Tested with Python 3.10. Packages I used:
```
pandas
numpy
scikit-learn
yfinance
matplotlib
```

## How to run
```
python spy_nextday.py --threshold 0.55 --plot
```
- `--threshold` sets the probability cutoff for when the model goes long (default 0.55)
- `--plot` shows the equity curves and drawdown; leave it out if you just want text output
- `--start` / `--end` let you change the date window if you’re curious; anything before 2022 just increases the training set, because the model only reports out-of-sample performance from the holdout period onward

## What you get
- counts for train/validation/test windows
- test accuracy, precision, recall, ROC-AUC, and class balance
- strategy vs. buy-and-hold stats: final multiple, Sharpe, CAGR, max drawdown, turnover
- plot (if you passed `--plot`)

Mostly a learning exercise, but it’s neat to see how just a few features can already tell a story about market regime :D

## Backend / API notes
- The FastAPI app in `app.py` serves `/api/spy-backtest` and `/api/spy-window` for the GitHub Pages front-end.
- Runtime calls never hit `yfinance`; instead they read from the bundled SQLite cache at `data/prices.db`, which now gets loaded once and reused in memory for faster responses.
- Keep the cache fresh by running `python seed_prices.py --ticker SPY --start 2000-01-01 --end <YYYY-MM-DD>` locally before deploying. This script can still use `yfinance`.
- If you need the API to refill data remotely (e.g., during development), call `backtest.run_strategy(..., allow_remote_download=True)`; restart the service afterward so the cached model picks up the new data.
