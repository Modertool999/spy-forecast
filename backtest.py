from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from price_store import get_available_span
from spy_nextday import (
    DEFAULT_HISTORY_START,
    backtest_long_flat,
    download_price_data,
    make_features,
    risk_metrics,
    train_logit,
)

DEFAULT_TICKER = "SPY"
DEFAULT_HISTORY_START_STR = DEFAULT_HISTORY_START.strftime("%Y-%m-%d")


def _resolve_split_dates(index: pd.Index) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Attempt to reuse the original train/validation/test split. If the caller
    requests a shorter history, fall back to a proportional 60/20/20 split.
    """
    default_train_end = pd.Timestamp("2019-01-01")
    default_valid_end = pd.Timestamp("2022-01-01")

    if len(index) == 0:
        raise ValueError("No samples available after feature engineering.")

    if index.min() <= default_train_end and index.max() >= default_valid_end:
        return default_train_end, default_valid_end

    ordered = index.sort_values()
    n = len(ordered)
    train_cut = ordered[int(n * 0.6)]
    valid_cut = ordered[int(n * 0.8)]
    return train_cut, valid_cut


def _safe_number(value: float | int | np.floating | None) -> Optional[float]:
    if value is None:
        return None
    val = float(value)
    if np.isnan(val):
        return None
    return val


def _pct(value: float | int | np.floating | None) -> Optional[float]:
    val = _safe_number(value)
    if val is None:
        return None
    return val * 100.0


def _resolve_requested_window(
    start: Optional[str],
    end: Optional[str],
) -> tuple[str, str]:
    """
    Favors the cached SPY span so the API immediately reflects real bounds.
    Falls back to the historical default and today's date when the cache is empty.
    """
    span = get_available_span(DEFAULT_TICKER)
    fallback_start = DEFAULT_HISTORY_START_STR
    fallback_end = datetime.today().strftime("%Y-%m-%d")
    if span:
        fallback_start = span[0].strftime("%Y-%m-%d")
        fallback_end = span[1].strftime("%Y-%m-%d")
    return start or fallback_start, end or fallback_end


def run_strategy(
    threshold: float = 0.55,
    start: Optional[str] = None,
    end: Optional[str] = None,
    allow_remote_download: bool = False,
) -> Dict[str, object]:
    """
    Train the logistic regression model on SPY history and run the
    out-of-sample backtest. Returns compact JSON-friendly stats.
    Set `allow_remote_download=True` to let the helper refresh prices
    from yfinance; the API keeps it False to stay cache-only.
    """
    requested_start, requested_end = _resolve_requested_window(start, end)

    price_data = download_price_data(
        DEFAULT_TICKER,
        requested_start,
        requested_end,
        allow_remote=allow_remote_download,
    )
    X, y, returns = make_features(price_data)

    if X.empty:
        raise ValueError("Not enough price history to compute features.")

    actual_start = price_data.index.min().strftime("%Y-%m-%d")
    actual_end = price_data.index.max().strftime("%Y-%m-%d")

    split_train_end, split_valid_end = _resolve_split_dates(X.index)

    train_mask = X.index < split_train_end
    valid_mask = (X.index >= split_train_end) & (X.index < split_valid_end)
    test_mask = X.index >= split_valid_end

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    returns_test = returns.loc[X_test.index]

    if X_train.empty or X_test.empty:
        raise ValueError("Need non-empty train and test samples for backtest.")

    model = train_logit(X_train, y_train)

    if not X_valid.empty:
        _ = model.predict_proba(X_valid)[:, 1]

    test_proba = model.predict_proba(X_test)[:, 1]
    proba_series = pd.Series(test_proba, index=X_test.index, name="p_up")

    backtest_results = backtest_long_flat(
        returns_test,
        proba_series,
        threshold=threshold,
        fee_bps=2.0,
    )
    baseline_equity = (1 + returns_test).cumprod()
    baseline_metrics = risk_metrics(returns_test)

    equity = backtest_results["equity"]
    metrics = backtest_results["metrics"]
    equity_list: List[float] = []
    if not equity.empty:
        equity_list = [float(val) for val in equity.round(6).values]
        total_return = _safe_number(equity.iloc[-1])
        total_return = total_return - 1.0 if total_return is not None else None
    else:
        total_return = None

    baseline_equity_list: List[float] = []
    if not baseline_equity.empty:
        baseline_equity_list = [float(val) for val in baseline_equity.round(6).values]
        baseline_total_return = _safe_number(baseline_equity.iloc[-1])
        if baseline_total_return is not None:
            baseline_total_return -= 1.0
    else:
        baseline_total_return = None

    return {
        "equity": equity_list,
        "total_ret": _pct(total_return),
        "cagr": _pct(metrics.get("cagr")),
        "sharpe": _safe_number(metrics.get("sharpe")),
        "max_drawdown": _pct(metrics.get("max_drawdown")),
        "baseline_equity": baseline_equity_list,
        "baseline_total_ret": _pct(baseline_total_return),
        "baseline_cagr": _pct(baseline_metrics.get("cagr")),
        "baseline_sharpe": _safe_number(baseline_metrics.get("sharpe")),
        "baseline_max_drawdown": _pct(baseline_metrics.get("max_drawdown")),
        "threshold": float(threshold),
        "start": actual_start,
        "end": actual_end,
    }
