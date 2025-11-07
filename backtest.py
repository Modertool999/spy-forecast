from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from spy_nextday import (
    backtest_long_flat,
    download_price_data,
    make_features,
    train_logit,
)


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
    start_date = start or "2005-01-01"
    end_date = end or datetime.today().strftime("%Y-%m-%d")

    price_data = download_price_data(
        "SPY",
        start_date,
        end_date,
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

    equity = backtest_results["equity"]
    metrics = backtest_results["metrics"]
    equity_list: List[float] = []
    if not equity.empty:
        equity_list = [float(val) for val in equity.round(6).values]
        total_return = _safe_number(equity.iloc[-1])
        total_return = total_return - 1.0 if total_return is not None else None
    else:
        total_return = None

    return {
        "equity": equity_list,
        "total_ret": _pct(total_return),
        "cagr": _pct(metrics.get("cagr")),
        "sharpe": _safe_number(metrics.get("sharpe")),
        "max_drawdown": _pct(metrics.get("max_drawdown")),
        "threshold": float(threshold),
        "start": start_date,
        "end": end_date,
    }
