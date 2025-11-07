from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

_DB_PATH = Path(__file__).resolve().parent / "data" / "prices.db"


def _ensure_db_dir() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_db_dir()
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_history (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_price_history_ticker_date
            ON price_history (ticker, date)
            """
        )


def _ts_to_str(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def upsert_prices(ticker: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    init_db()
    frame = frame.copy()
    if "Adj Close" not in frame.columns and "Close" in frame.columns:
        frame["Adj Close"] = frame["Close"]
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in expected_cols:
        if col not in frame.columns:
            frame[col] = float("nan")
    records = []
    for idx, row in frame.sort_index().iterrows():
        idx_ts = pd.Timestamp(idx).tz_localize(None)
        records.append(
            (
                ticker,
                _ts_to_str(idx_ts),
                None if pd.isna(row["Open"]) else float(row["Open"]),
                None if pd.isna(row["High"]) else float(row["High"]),
                None if pd.isna(row["Low"]) else float(row["Low"]),
                None if pd.isna(row["Close"]) else float(row["Close"]),
                None if pd.isna(row["Adj Close"]) else float(row["Adj Close"]),
                None if pd.isna(row["Volume"]) else float(row["Volume"]),
            )
        )
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO price_history (
                ticker, date, open, high, low, close, adj_close, volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
    return len(records)


def load_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    init_db()
    query = """
        SELECT date, open, high, low, close, adj_close, volume
        FROM price_history
        WHERE ticker = ?
          AND date BETWEEN ? AND ?
        ORDER BY date
    """
    with _connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(ticker, _ts_to_str(start), _ts_to_str(end)),
            parse_dates=["date"],
        )
    if df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    ).set_index("Date")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_available_span(ticker: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    init_db()
    query = "SELECT MIN(date) AS start, MAX(date) AS end FROM price_history WHERE ticker = ?"
    with _connect() as conn:
        row = conn.execute(query, (ticker,)).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    return pd.Timestamp(row[0]), pd.Timestamp(row[1])
