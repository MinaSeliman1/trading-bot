#!/usr/bin/env python3
"""Exploratory only: append recent 1-minute bars from Yahoo Finance to a growing Parquet file.   [NOT EXECUTED]

  python fetch/yfinance_fetch.py SPY data/SPY_1m_yf.parquet      # run it every few days to accumulate history

Yahoo serves 1-minute bars for roughly the last 30 days only (7 days per request); that is NOT enough
for a statistically meaningful study. auto_adjust=False keeps raw prices; prepost=False keeps RTH only.
Yahoo bars are stamped with their OPEN time. Tickers: SPY, QQQ, ES=F, NQ=F (continuous front month,
no contract id -> roll integrity cannot be verified; the QC report says so).
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


def main(ticker, out):
    new = yf.download(ticker, period="7d", interval="1m", auto_adjust=False, prepost=False, progress=False)
    if isinstance(new.columns, pd.MultiIndex):
        new.columns = new.columns.get_level_values(0)
    new = new.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    new.index = pd.DatetimeIndex(new.index).tz_convert("UTC")
    new.index.name = "timestamp"
    p = Path(out)
    if p.exists():
        new = pd.concat([pd.read_parquet(p), new])
        new = new[~new.index.duplicated(keep="last")].sort_index()
    new.to_parquet(p)
    print("rows:", len(new), new.index.min(), "->", new.index.max())


if __name__ == "__main__":
    main(*sys.argv[1:3])
