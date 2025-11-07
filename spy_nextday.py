#!/usr/bin/env python3
import argparse
import sys
import warnings
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings(
    "ignore",
    message="Pandas requires version '1.3.6' or newer of 'bottleneck'",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from price_store import get_available_span, load_prices, upsert_prices

pd.set_option("display.float_format", "{:.6f}".format)

DEFAULT_HISTORY_START = pd.Timestamp("2000-01-01")
_NO_DATA_HINTS = ("no price data", "possibly delisted", "yfpricesmissingerror", "empty frame")


def _no_data_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _NO_DATA_HINTS)


def _normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(-1)
    rename_map = {
        "Adj Close": "Adj Close",
        "Close": "Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    return df


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _fetch_from_yahoo(ticker: str, start: str, end: str, retries: int = 2) -> pd.DataFrame:
    errors = []
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return _normalize_price_columns(df).sort_index()
            errors.append(f"download attempt {attempt + 1}: empty frame")
        except Exception as exc:
            errors.append(f"download attempt {attempt + 1}: {exc}")
    try:
        ticker_obj = yf.Ticker(ticker)
        df_alt = ticker_obj.history(start=start, end=end, auto_adjust=True, interval="1d")
        if not df_alt.empty:
            df_alt.index = pd.to_datetime(df_alt.index).tz_localize(None)
            return _normalize_price_columns(df_alt).sort_index()
        errors.append("ticker.history returned empty frame")
    except Exception as exc:
        errors.append(f"ticker.history error: {exc}")
    detail = "; ".join(errors) if errors else "unknown error"
    raise RuntimeError(f"Failed to download data for {ticker}. Details: {detail}")


def download_price_data(ticker, start, end, retries=2, allow_remote=True):
    start_ts = pd.Timestamp(start or DEFAULT_HISTORY_START)
    end_ts = pd.Timestamp(end or datetime.today().strftime("%Y-%m-%d"))
    if start_ts > end_ts:
        raise ValueError(f"Start date {start_ts.date()} is after end date {end_ts.date()}.")
    span = get_available_span(ticker)
    missing_segments = []

    if span is None:
        cache_start = cache_end = None
        segment_start = min(start_ts, DEFAULT_HISTORY_START)
        missing_segments.append((segment_start, end_ts))
    else:
        cache_start, cache_end = span
        if start_ts < cache_start:
            seg_start = min(start_ts, DEFAULT_HISTORY_START)
            seg_end = cache_start - timedelta(days=1)
            if seg_start <= seg_end:
                missing_segments.append((seg_start, seg_end))
        if end_ts > cache_end:
            seg_start = cache_end + timedelta(days=1)
            seg_end = end_ts
            if seg_start <= seg_end:
                missing_segments.append((seg_start, seg_end))

    if missing_segments and not allow_remote:
        if span is None:
            raise RuntimeError(
                "Local price cache is empty. Run `python seed_prices.py --ticker SPY` "
                "to populate the SQLite store before using cache-only mode."
            )
        missing_desc = ", ".join(f"{s.date()}-{e.date()}" for s, e in missing_segments)
        raise ValueError(
            f"Requested window {start_ts.date()} - {end_ts.date()} falls outside the cached span "
            f"{cache_start.date()} - {cache_end.date()} (missing {missing_desc}). "
            "Adjust the dates or refresh the cache via `python seed_prices.py`."
        )

    if allow_remote:
        for seg_start, seg_end in missing_segments:
            cursor = seg_start
            filled = False
            while cursor <= seg_end:
                try:
                    fresh = _fetch_from_yahoo(
                        ticker,
                        start=cursor.strftime("%Y-%m-%d"),
                        end=(seg_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                        retries=retries,
                    )
                except RuntimeError as exc:
                    if _no_data_error(exc):
                        cursor += timedelta(days=1)
                        continue
                    raise
                if fresh.empty:
                    cursor += timedelta(days=1)
                    continue
                upsert_prices(ticker, fresh)
                filled = True
                break
            if not filled:
                raise RuntimeError(
                    f"Failed to backfill prices for {ticker} between "
                    f"{seg_start.date()} and {seg_end.date()}."
                )

    frame = load_prices(ticker, start_ts, end_ts)
    if frame.empty:
        raise RuntimeError(
            f"No stored data available for {ticker} between {start_ts.date()} and {end_ts.date()}. "
            "Refresh the cache with `python seed_prices.py`."
        )
    frame = frame.sort_index()
    frame_start, frame_end = frame.index.min(), frame.index.max()
    if frame_start > start_ts:
        raise RuntimeError(
            f"Stored data for {ticker} starts at {frame_start.date()}, "
            f"which is after requested start {start_ts.date()}. "
            "Refresh the cache or request a later start date."
        )
    effective_end = min(end_ts, frame_end)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= effective_end)].copy()


def make_features(df):
    close = df["Close"]
    open_ = df["Open"]
    ret1 = close.pct_change()

    features = pd.DataFrame(index=df.index)
    features["r5"] = close.pct_change(5).shift(1)
    features["r20"] = close.pct_change(20).shift(1)
    features["vol20"] = ret1.rolling(20).std().shift(1)
    features["gap"] = (open_ / close.shift(1) - 1).shift(1)
    features["rsi14"] = compute_rsi(close).shift(1)

    target = (ret1.shift(-1) > 0).astype(int)
    target.name = "target"
    ret1.name = "ret1"

    data = features.copy()
    data["target"] = target
    data["ret1"] = ret1
    data = data.dropna()

    X = data.loc[:, features.columns]
    y = data.loc[:, "target"]
    returns = data.loc[:, "ret1"]
    assert (X.index == y.index).all(), "Feature and target indices misaligned."
    assert (X.index == returns.index).all(), "Returns index misaligned."
    return X, y, returns


def train_logit(X_train, y_train):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification(y_true, proba, threshold=0.5):
    if len(y_true) == 0:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "roc_auc": np.nan}
    preds = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, proba)
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics


def risk_metrics(returns, freq=252):
    if returns.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    equity = (1 + returns).cumprod()
    n_days = len(returns)
    total_return = equity.iloc[-1]
    years = n_days / freq if n_days else np.nan
    cagr = total_return ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = returns.std(ddof=0)
    sharpe = np.sqrt(freq) * returns.mean() / (vol + 1e-12)
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()
    return {"cagr": cagr, "sharpe": sharpe, "max_drawdown": max_drawdown}


def backtest_long_flat(
    returns,
    proba,
    threshold=0.55,
    fee_bps=2.0,
    freq=252,
):
    if returns.empty or proba.empty:
        return {
            "positions": pd.Series(dtype=float),
            "net_returns": pd.Series(dtype=float),
            "equity": pd.Series(dtype=float),
            "metrics": {"cagr": np.nan, "sharpe": np.nan, "max_drawdown": np.nan},
            "final_multiple": np.nan,
            "turnover": np.nan,
        }
    proba = proba.reindex(returns.index).fillna(method="ffill").fillna(0.5)
    positions = (proba > threshold).astype(int)
    fee_rate = fee_bps / 10000.0
    position_change = positions.diff().fillna(positions)
    fees = position_change.abs() * fee_rate
    pnl = positions * returns
    net_returns = pnl - fees
    equity = (1 + net_returns).cumprod()
    metrics = risk_metrics(net_returns, freq=freq)
    final_multiple = equity.iloc[-1] if not equity.empty else np.nan
    turnover = (position_change.abs() > 0).sum() / len(positions)
    return {
        "positions": positions,
        "net_returns": net_returns,
        "equity": equity,
        "metrics": metrics,
        "final_multiple": final_multiple,
        "turnover": turnover,
    }


def format_percentage(value):
    if pd.isna(value):
        return "nan"
    return f"{value * 100:.2f}%"


def format_ratio(value):
    if pd.isna(value):
        return "nan"
    return f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Predict next-day SPY direction with logistic regression.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Probability threshold for going long (default: 0.55).")
    parser.add_argument("--plot", action="store_true", help="Show equity curve and drawdown plots.")
    parser.add_argument("--start", type=str, default="2005-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD).")
    args = parser.parse_args()

    print("Example: python spy_nextday.py --threshold 0.55 --plot")

    try:
        price_data = download_price_data("SPY", args.start, args.end)
    except Exception as exc:
        print(f"Error downloading data: {exc}")
        sys.exit(1)

    X, y, returns = make_features(price_data)

    split_train_end = pd.Timestamp("2019-01-01")
    split_valid_end = pd.Timestamp("2022-01-01")

    train_mask = X.index < split_train_end
    valid_mask = (X.index >= split_train_end) & (X.index < split_valid_end)
    test_mask = X.index >= split_valid_end

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    returns_test = returns.loc[X_test.index]

    print("Samples & date ranges:")
    if not X_train.empty:
        print(f"  Train: {len(X_train)} [{X_train.index.min().date()} - {X_train.index.max().date()}]")
    else:
        print("  Train: 0")
    if not X_valid.empty:
        print(f"  Valid: {len(X_valid)} [{X_valid.index.min().date()} - {X_valid.index.max().date()}]")
    else:
        print("  Valid: 0")
    if not X_test.empty:
        print(f"  Test:  {len(X_test)} [{X_test.index.min().date()} - {X_test.index.max().date()}]")
    else:
        print("  Test: 0")

    if X_train.empty or X_test.empty:
        print("Insufficient data after feature engineering for training/testing.")
        sys.exit(1)

    model = train_logit(X_train, y_train)
    if not X_valid.empty:
        valid_proba = model.predict_proba(X_valid)[:, 1]
        valid_metrics = evaluate_classification(y_valid, valid_proba)
        print(
            "Validation Accuracy: "
            f"{valid_metrics['accuracy']:.4f} (Precision {valid_metrics['precision']:.4f}, "
            f"Recall {valid_metrics['recall']:.4f})"
        )

    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_classification(y_test, test_proba)

    print("Classification (Test):")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print(f"  ROC-AUC  : {test_metrics['roc_auc']:.4f}")

    total_test = len(y_test)
    print("  Class balance:")
    for cls in [0, 1]:
        count = int((y_test == cls).sum())
        frac = count / total_test if total_test else float("nan")
        print(f"    {cls}: {count} ({frac:.2%})" if not np.isnan(frac) else f"    {cls}: {count} (nan)")

    test_proba_series = pd.Series(test_proba, index=X_test.index, name="p_up")
    backtest_results = backtest_long_flat(returns_test, test_proba_series, threshold=args.threshold, fee_bps=2.0)
    baseline_equity = (1 + returns_test).cumprod()
    baseline_metrics = risk_metrics(returns_test)
    baseline_final = baseline_equity.iloc[-1] if not baseline_equity.empty else np.nan
    strat_metrics = backtest_results["metrics"]
    print("Backtest (Strategy):")
    print(f"  Threshold      : {args.threshold:.2f}")
    print("  Fees (per trade): 0.0002 (2 bps)")
    print(f"  Final multiple : {backtest_results['final_multiple']:.4f}")
    print(f"  CAGR           : {format_percentage(strat_metrics['cagr'])}")
    print(f"  Sharpe (ann.)  : {format_ratio(strat_metrics['sharpe'])}")
    print(f"  Max Drawdown   : {format_percentage(strat_metrics['max_drawdown'])}")
    print(f"  Turnover       : {format_percentage(backtest_results['turnover'])}")

    print("Baseline (Buy & Hold):")
    print(f"  Final multiple : {baseline_final:.4f}" if not pd.isna(baseline_final) else "  Final multiple : nan")
    print(f"  CAGR           : {format_percentage(baseline_metrics['cagr'])}")
    print(f"  Sharpe (ann.)  : {format_ratio(baseline_metrics['sharpe'])}")
    print(f"  Max Drawdown   : {format_percentage(baseline_metrics['max_drawdown'])}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        equity = backtest_results["equity"]
        ax_eq.plot(equity.index, equity.values, label="Strategy")
        ax_eq.plot(baseline_equity.index, baseline_equity.values, label="Buy & Hold")
        ax_eq.set_yscale("log")
        ax_eq.set_ylabel("Equity (log scale)")
        ax_eq.set_title("Equity Curve")
        ax_eq.legend()

        strategy_drawdown = equity / equity.cummax() - 1 if not equity.empty else pd.Series(dtype=float)
        ax_dd.fill_between(strategy_drawdown.index, strategy_drawdown.values * 100, label="Drawdown", alpha=0.3)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_xlabel("Date")
        ax_dd.set_title("Strategy Drawdown")
        ax_dd.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
