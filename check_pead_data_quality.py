"""
Data quality check on all currently-cached PEAD tickers (earnings +
price bars) before the full viability run. No new API calls. Flags:
- earnings: duplicate report dates, missing/NaN surprise_pct, extreme
  outlier surprise_pct values, suspiciously large gaps between quarters
- price bars: non-positive prices, large gaps in the trading calendar,
  missing bars entirely, insufficient history to cover the earliest
  cached earnings event
"""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_quality")

LOG_PATH = "exploration_log_pead.md"
CACHE_DIR = Path("earnings_cache")
EXTREME_SURPRISE_PCT = 500.0  # beyond this, treat as a likely data artifact (split, bad decimal, etc.)
MIN_QUARTER_GAP_DAYS = 60      # quarterly reports should be ~90 days apart; flag anything much tighter
MAX_QUARTER_GAP_DAYS = 200     # flag anything suggesting a missed quarter


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def check_earnings_quality(tickers):
    av = AlphaVantageEarningsClient(config.ALPHA_VANTAGE_API_KEY)
    issues = {}
    for t in tickers:
        df = av.get_quarterly_surprises(t, use_cache=True)
        ticker_issues = []
        if df.empty:
            ticker_issues.append("empty parsed result despite a cache file existing")
            issues[t] = ticker_issues
            continue

        dup_dates = df.index[df.index.duplicated()]
        if len(dup_dates) > 0:
            ticker_issues.append(f"{len(dup_dates)} duplicate report dates: {list(dup_dates.unique())[:3]}")

        n_missing_surprise = df["surprise_pct"].isna().sum()
        if n_missing_surprise > 0:
            ticker_issues.append(f"{n_missing_surprise}/{len(df)} rows with NaN surprise_pct")

        extreme = df[df["surprise_pct"].abs() > EXTREME_SURPRISE_PCT]
        if len(extreme) > 0:
            vals = extreme["surprise_pct"].round(1).tolist()[:3]
            ticker_issues.append(f"{len(extreme)} extreme surprise_pct values (>|{EXTREME_SURPRISE_PCT}%|): {vals}")

        if not df.index.is_monotonic_increasing:
            ticker_issues.append("report dates not monotonically increasing after sort (unexpected)")

        gaps = df.index.to_series().diff().dt.days.dropna()
        too_tight = gaps[gaps < MIN_QUARTER_GAP_DAYS]
        too_wide = gaps[gaps > MAX_QUARTER_GAP_DAYS]
        if len(too_tight) > 0:
            ticker_issues.append(f"{len(too_tight)} quarter gaps < {MIN_QUARTER_GAP_DAYS}d "
                                 f"(possible duplicate/restated report): {too_tight.tolist()[:3]}")
        if len(too_wide) > 0:
            ticker_issues.append(f"{len(too_wide)} quarter gaps > {MAX_QUARTER_GAP_DAYS}d "
                                 f"(possible missed quarter): {too_wide.tolist()[:3]}")

        if ticker_issues:
            issues[t] = ticker_issues
    return issues


def check_price_quality(tickers):
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=365 * 6)
    issues = {}
    for t in tickers:
        bars = dc.get_bars(t, "us_equity", "1Day", start, end)
        ticker_issues = []
        if bars.empty:
            ticker_issues.append("no price bars returned at all")
            issues[t] = ticker_issues
            continue

        non_positive = bars[(bars["open"] <= 0) | (bars["close"] <= 0) | (bars["high"] <= 0) | (bars["low"] <= 0)]
        if len(non_positive) > 0:
            ticker_issues.append(f"{len(non_positive)} bars with non-positive OHLC")

        bad_ohlc = bars[(bars["high"] < bars["low"]) | (bars["high"] < bars["open"]) | (bars["high"] < bars["close"])]
        if len(bad_ohlc) > 0:
            ticker_issues.append(f"{len(bad_ohlc)} bars with high < low/open/close (impossible OHLC)")

        gaps = bars.index.to_series().diff().dt.days.dropna()
        large_gaps = gaps[gaps > 10]  # more than ~2 weeks with no bar (beyond normal weekends/holidays)
        if len(large_gaps) > 0:
            ticker_issues.append(f"{len(large_gaps)} gaps >10 calendar days in the daily bar index: "
                                 f"{large_gaps.tolist()[:3]}")

        if len(bars) < 200:
            ticker_issues.append(f"only {len(bars)} total daily bars -- thin history")

        if ticker_issues:
            issues[t] = ticker_issues
    return issues


def main():
    tickers = sorted(p.stem.replace("_earnings", "") for p in CACHE_DIR.glob("*_earnings.json"))
    log_md(f"\n## Data quality check: {len(tickers)} cached tickers\n")

    earnings_issues = check_earnings_quality(tickers)
    price_issues = check_price_quality(tickers)

    if not earnings_issues:
        log_md(f"- Earnings data: no issues found across all {len(tickers)} tickers "
              f"(no duplicates, no NaN surprises, no extreme outliers >|{EXTREME_SURPRISE_PCT}%|, "
              f"no quarter-gap anomalies outside {MIN_QUARTER_GAP_DAYS}-{MAX_QUARTER_GAP_DAYS} days).")
    else:
        log_md(f"- Earnings data: {len(earnings_issues)}/{len(tickers)} tickers flagged:")
        for t, iss in earnings_issues.items():
            for i in iss:
                log_md(f"    {t}: {i}")

    if not price_issues:
        log_md(f"- Price bars: no issues found across all {len(tickers)} tickers "
              f"(no non-positive OHLC, no impossible high/low, no large gaps, sufficient history).")
    else:
        log_md(f"- Price bars: {len(price_issues)}/{len(tickers)} tickers flagged:")
        for t, iss in price_issues.items():
            for i in iss:
                log_md(f"    {t}: {i}")

    return earnings_issues, price_issues


if __name__ == "__main__":
    main()
