#!/usr/bin/env python3
"""Download 1-minute OHLCV for ES / NQ outright contracts from Databento (CME Globex) to Parquet.   [NOT EXECUTED]

  export DATABENTO_API_KEY=...
  python fetch/databento_fetch.py ES 2022-01-03 2025-12-31 data/ES_1m.parquet --cost-only
  python fetch/databento_fetch.py ES 2022-01-03 2025-12-31 data/ES_1m.parquet

Why per-contract data: the backtest keeps ONE contract per session (the most traded one that day),
so a roll can never create a fake price jump inside a session. The 'contract' column carries the
raw symbol (e.g. ESH5). Calendar spreads (symbols containing '-') are removed.
Timestamps (ts_event) are bar OPEN times in UTC. Prices are raw, unadjusted.
"""
import os
import re
import sys

import databento as db

PARENT = {"ES": "ES.FUT", "NQ": "NQ.FUT"}


def main(root, start, end, out, cost_only=False):
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    kw = dict(dataset="GLBX.MDP3", schema="ohlcv-1m", symbols=[PARENT[root]], stype_in="parent", start=start, end=end)
    cost = client.metadata.get_cost(**kw)
    print(f"estimated cost for {root} {start}..{end}: {cost}")
    if cost_only:
        return
    df = client.timeseries.get_range(**kw).to_df()          # index: ts_event (UTC)
    df = df[df["symbol"].str.match(rf"^{root}[FGHJKMNQUVXZ]\d{{1,2}}$")]       # outright contracts only
    df = df.rename(columns={"symbol": "contract"})
    df.index.name = "timestamp"
    df[["open", "high", "low", "close", "volume", "contract"]].to_parquet(out)
    print("wrote", out, len(df))


if __name__ == "__main__":
    main(*sys.argv[1:5], cost_only="--cost-only" in sys.argv)
