#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime

from price_store import get_available_span
from spy_nextday import download_price_data


def _default_end() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed or refresh the local price database.")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker to download (default: SPY)")
    parser.add_argument(
        "--start",
        type=str,
        default="2000-01-01",
        help="Historical start date (YYYY-MM-DD, default: 2000-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=_default_end(),
        help="Historical end date (YYYY-MM-DD, default: today)",
    )
    args = parser.parse_args()

    before_span = get_available_span(args.ticker)
    frame = download_price_data(args.ticker, args.start, args.end)
    after_span = get_available_span(args.ticker)

    if frame.empty:
        print("No data downloaded. Check the provided dates.")
        return

    print(
        f"Saved {len(frame)} rows for {args.ticker} covering "
        f"{frame.index.min().date()} - {frame.index.max().date()}."
    )
    if before_span == after_span:
        print("Database span unchanged.")
    else:
        if before_span is None:
            print(
                "Database initialized with "
                f"{after_span[0].date()} - {after_span[1].date()} coverage."
            )
        else:
            print(
                "Database span updated: "
                f"{before_span[0].date()} - {before_span[1].date()} "
                f"-> {after_span[0].date()} - {after_span[1].date()}."
            )


if __name__ == "__main__":
    main()
