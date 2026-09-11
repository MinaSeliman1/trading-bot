import pandas as pd

from bot.data_client import DataClient


def test_closed_candle_enforcement_excludes_in_progress_bar():
    idx = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx)
    now = idx[-1] + pd.Timedelta(minutes=5)  # last bar's 15-min interval hasn't elapsed yet

    filtered = DataClient.drop_unclosed_candle(bars, "15Min", now=now)

    assert idx[-1] not in filtered.index
    assert len(filtered) == 4


def test_closed_candle_enforcement_keeps_fully_elapsed_bars():
    idx = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx)
    now = idx[-1] + pd.Timedelta(minutes=16)  # interval has fully elapsed

    filtered = DataClient.drop_unclosed_candle(bars, "15Min", now=now)

    assert idx[-1] in filtered.index
    assert len(filtered) == 5


def test_liquidity_filter_blocks_low_volume_bar():
    idx = pd.date_range("2026-01-01", periods=25, freq="h", tz="UTC")
    volumes = [1000] * 24 + [200]
    bars = pd.DataFrame({"close": range(25), "volume": volumes}, index=idx)
    assert DataClient.liquidity_ok(bars, lookback=20, min_volume_ratio=0.5) is False


def test_liquidity_filter_allows_normal_volume_bar():
    idx = pd.date_range("2026-01-01", periods=25, freq="h", tz="UTC")
    volumes = [1000] * 24 + [900]
    bars = pd.DataFrame({"close": range(25), "volume": volumes}, index=idx)
    assert DataClient.liquidity_ok(bars, lookback=20, min_volume_ratio=0.5) is True


def test_liquidity_filter_blocks_wide_spread():
    idx = pd.date_range("2026-01-01", periods=25, freq="h", tz="UTC")
    volumes = [1000] * 25
    bars = pd.DataFrame({"close": range(25), "volume": volumes}, index=idx)
    assert DataClient.liquidity_ok(bars, lookback=20, min_volume_ratio=0.5, bid=100.0, ask=105.0, max_spread_pct=0.01) is False
