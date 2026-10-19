#!/usr/bin/env python3
"""Download 1-minute SIP bars (raw, unadjusted) for SPY/QQQ from Alpaca into Parquet.   [NOT EXECUTED]

  export ALPACA_API_KEY=... ALPACA_API_SECRET=...
  python fetch/alpaca_fetch.py SPY 2021-01-01 2025-12-31 data/SPY_1m.parquet

Notes
* feed=SIP and adjustment=RAW are explicit. The IEX feed only contains IEX-venue trades
  (many missing minutes, different OHLC) and must not be used for this study.
* The free/paper SIP feed refuses very recent data ("subscription does not permit querying
  recent SIP data"); the end date is capped at now-24h like the rest of this repo.
* Bars are stamped with their OPEN time in UTC. Load with the default --ts-convention start.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def main(symbol, start, end, out):
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
    start = pd.Timestamp(start, tz="UTC")
    end = min(pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1), pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=24)))
    frames = []
    for lo in pd.date_range(start, end, freq="MS", tz="UTC").union([start]):
        hi = min(lo + pd.offsets.MonthBegin(1), end)
        if lo >= hi:
            continue
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=lo.to_pydatetime(),
                               end=hi.to_pydatetime(), feed=DataFeed.SIP, adjustment=Adjustment.RAW)
        df = client.get_stock_bars(req).df
        if len(df):
            frames.append(df.reset_index(level=0, drop=True))
        print(symbol, lo.date(), "->", hi.date(), len(df), "rows", flush=True)
    all_ = pd.concat(frames).sort_index()
    all_ = all_[~all_.index.duplicated(keep="first")]
    all_.index.name = "timestamp"
    all_[["open", "high", "low", "close", "volume"]].to_parquet(out)
    print("wrote", out, len(all_))


if __name__ == "__main__":
    main(*sys.argv[1:5])
