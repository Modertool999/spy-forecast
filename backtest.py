from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

StrategyCache = Dict[str, object]
_STRATEGY_CACHE: Optional[StrategyCache] = None
_CACHE_KEY: Optional[Tuple[str, str]] = None


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


def _ensure_strategy_cache(
    allow_remote_download: bool,
    requested_end: pd.Timestamp,
) -> StrategyCache:
    """
    Build (or reuse) a cache of model outputs backed by the full local history.
    The cache is invalidated whenever the stored DB span changes.
    """
    global _STRATEGY_CACHE, _CACHE_KEY

    span = get_available_span(DEFAULT_TICKER)
    if span is None:
        span_start = DEFAULT_HISTORY_START
        span_end = pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))
    else:
        span_start, span_end = span

    if allow_remote_download and requested_end > span_end:
        span_end = requested_end

    span_key = (span_start.strftime("%Y-%m-%d"), span_end.strftime("%Y-%m-%d"))
    if _STRATEGY_CACHE is not None and _CACHE_KEY == span_key:
        return _STRATEGY_CACHE

    price_data = download_price_data(
        DEFAULT_TICKER,
        span_key[0],
        span_key[1],
        allow_remote=allow_remote_download,
    )
    X, y, returns = make_features(price_data)
    if X.empty:
        raise ValueError("Not enough price history to compute features.")

    split_train_end, split_valid_end = _resolve_split_dates(X.index)

    train_mask = X.index < split_train_end
    valid_mask = (X.index >= split_train_end) & (X.index < split_valid_end)
    test_mask = X.index >= split_valid_end

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_valid = X.loc[valid_mask]
    X_test, returns_test = X.loc[test_mask], returns.loc[test_mask]

    if X_train.empty or X_test.empty:
        raise ValueError("Need non-empty train and test samples for backtest.")

    model = train_logit(X_train, y_train)
    if not X_valid.empty:
        _ = model.predict_proba(X_valid)[:, 1]

    test_proba = model.predict_proba(X_test)[:, 1]
    proba_series = pd.Series(test_proba, index=X_test.index, name="p_up")
    _STRATEGY_CACHE = {
        "returns_test": returns_test,
        "proba_series": proba_series,
    }
    _CACHE_KEY = span_key
    return _STRATEGY_CACHE


def get_holdout_window(allow_remote_download: bool = False) -> tuple[str, str]:
    """
    Return the actual out-of-sample window (in YYYY-MM-DD) backed by cached data.
    Forces the strategy cache to be hydrated so callers see the same bounds that
    the backtest uses for every request.
    """
    today = pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))
    cache = _ensure_strategy_cache(allow_remote_download, today)
    returns_test: pd.Series = cache["returns_test"]
    if returns_test.empty:
        raise ValueError("Holdout window unavailable; seed the price cache first.")
    start = returns_test.index.min().strftime("%Y-%m-%d")
    end = returns_test.index.max().strftime("%Y-%m-%d")
    return start, end


def run_strategy(
    threshold: float = 0.55,
    start: Optional[str] = None,
    end: Optional[str] = None,
    allow_remote_download: bool = False,
) -> Dict[str, object]:
    """
    Train (or reuse) the logistic regression model and return JSON-friendly stats.
    Set `allow_remote_download=True` to let the helper refresh prices
    from yfinance; the API keeps it False to stay cache-only.
    """
    requested_start, requested_end = _resolve_requested_window(start, end)
    requested_start_ts = pd.Timestamp(requested_start)
    requested_end_ts = pd.Timestamp(requested_end)
    if requested_start_ts > requested_end_ts:
        raise ValueError(
            f"Start date {requested_start_ts.date()} is after end date {requested_end_ts.date()}."
        )

    cache = _ensure_strategy_cache(allow_remote_download, requested_end_ts)
    returns_test: pd.Series = cache["returns_test"]
    proba_series: pd.Series = cache["proba_series"]

    available_start = returns_test.index.min()
    available_end = returns_test.index.max()
    overlap_start = max(requested_start_ts, available_start)
    overlap_end = min(requested_end_ts, available_end)

    mask = (returns_test.index >= overlap_start) & (returns_test.index <= overlap_end)
    if not mask.any():
        raise ValueError(
            "Requested window "
            f"{requested_start_ts.date()} - {requested_end_ts.date()} "
            "does not overlap the available backtest window "
            f"{available_start.date()} - {available_end.date()}."
        )

    returns_subset = returns_test.loc[mask]
    proba_subset = proba_series.loc[returns_subset.index]

    backtest_results = backtest_long_flat(
        returns_subset,
        proba_subset,
        threshold=threshold,
        fee_bps=2.0,
    )
    baseline_equity = (1 + returns_subset).cumprod()
    baseline_metrics = risk_metrics(returns_subset)

    equity = backtest_results["equity"]
    metrics = backtest_results["metrics"]
    equity_list: List[float] = []
    equity_dates: List[str] = []
    if not equity.empty:
        equity_list = [float(val) for val in equity.round(6).values]
        equity_dates = [idx.strftime("%Y-%m-%d") for idx in equity.index]
        total_return = _safe_number(equity.iloc[-1])
        total_return = total_return - 1.0 if total_return is not None else None
    else:
        total_return = None

    baseline_equity_list: List[float] = []
    baseline_dates: List[str] = []
    if not baseline_equity.empty:
        baseline_equity_list = [float(val) for val in baseline_equity.round(6).values]
        baseline_dates = [idx.strftime("%Y-%m-%d") for idx in baseline_equity.index]
        baseline_total_return = _safe_number(baseline_equity.iloc[-1])
        if baseline_total_return is not None:
            baseline_total_return -= 1.0
    else:
        baseline_total_return = None

    actual_start = equity.index.min().strftime("%Y-%m-%d") if not equity.empty else overlap_start.strftime("%Y-%m-%d")
    actual_end = equity.index.max().strftime("%Y-%m-%d") if not equity.empty else overlap_end.strftime("%Y-%m-%d")

    return {
        "equity": equity_list,
        "equity_dates": equity_dates,
        "total_ret": _pct(total_return),
        "cagr": _pct(metrics.get("cagr")),
        "sharpe": _safe_number(metrics.get("sharpe")),
        "max_drawdown": _pct(metrics.get("max_drawdown")),
        "baseline_equity": baseline_equity_list,
        "baseline_dates": baseline_dates,
        "baseline_total_ret": _pct(baseline_total_return),
        "baseline_cagr": _pct(baseline_metrics.get("cagr")),
        "baseline_sharpe": _safe_number(baseline_metrics.get("sharpe")),
        "baseline_max_drawdown": _pct(baseline_metrics.get("max_drawdown")),
        "threshold": float(threshold),
        "start": actual_start,
        "end": actual_end,
    }
