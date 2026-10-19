import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fcr_fvg.data import build_sessions, load_ohlcv
from fcr_fvg.instruments import ES, SPY
from fcr_fvg.params import StrategyParams
from tests.synth import bday_range, random_walk_frame

P = StrategyParams()


def write(tmp_path, df, name="x.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_utc_timestamps_map_to_0930_new_york_across_dst(tmp_path):
    # 2024-03-08 is EST (UTC-5), 2024-03-11 is EDT (UTC-4): same wall-clock 09:30 -> different UTC hour
    df = random_walk_frame([dt.date(2024, 3, 8), dt.date(2024, 3, 11)], extended=False)
    f = load_ohlcv(write(tmp_path, df))
    for d in ("2024-03-08", "2024-03-11"):
        day = f[f.index.strftime("%Y-%m-%d") == d]
        assert day.index[0].strftime("%H:%M") == "09:30" and day.index[-1].strftime("%H:%M") == "15:59"
    raw = pd.read_csv(tmp_path / "x.csv")
    assert raw["timestamp"].iloc[0].startswith("2024-03-08 14:30") and raw["timestamp"].iloc[390].startswith("2024-03-11 13:30")


def test_naive_timestamps_without_tz_are_refused(tmp_path):
    df = random_walk_frame([dt.date(2024, 3, 8)], extended=False)
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    p = write(tmp_path, df)
    with pytest.raises(ValueError):
        load_ohlcv(p)
    f = load_ohlcv(p, input_tz="America/New_York")
    assert f.index[0].strftime("%H:%M") == "09:30"


def test_end_stamped_bars_are_shifted_back_one_minute(tmp_path):
    df = random_walk_frame([dt.date(2024, 3, 8)], extended=False)
    df["timestamp"] = df["timestamp"] + pd.Timedelta(minutes=1)
    f = load_ohlcv(write(tmp_path, df), ts_convention="end")
    assert f.index[0].strftime("%H:%M") == "09:30"


@pytest.mark.parametrize("unit,div", [("s", 1), ("ms", 1000), ("ns", 10**9)])
def test_epoch_timestamps_supported(tmp_path, unit, div):
    df = random_walk_frame([dt.date(2024, 3, 8)], extended=False)
    secs = (df["timestamp"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    df["timestamp"] = secs * {"s": 1, "ms": 1000, "ns": 10**9}[unit]
    f = load_ohlcv(write(tmp_path, df))
    assert f.index[0].strftime("%H:%M") == "09:30" and f.index[-1].strftime("%H:%M") == "15:59"


def test_extended_hours_holiday_and_weekend_are_removed(tmp_path):
    days = bday_range("2024-03-25", "2024-03-29")                    # Mar 29 2024 = Good Friday (NYSE closed)
    df = random_walk_frame(days, extended=True)
    f = load_ohlcv(write(tmp_path, df))
    sessions, qc = build_sessions(f, SPY, P)
    assert [s.date for s in sessions] == [dt.date(2024, 3, d) for d in (25, 26, 27, 28)]
    assert qc.sessions_excluded.get("not_an_nyse_trading_day") == 1
    assert qc.rows_outside_rth_dropped > 0
    for s in sessions:
        assert np.isfinite(s.o[:390]).all()


def test_half_day_session_has_210_minutes(tmp_path):
    days = [dt.date(2024, 11, 29)]                                   # day after Thanksgiving: 13:00 close
    df = random_walk_frame(days, extended=False)
    df["ts_ny"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df[df["ts_ny"].dt.hour * 60 + df["ts_ny"].dt.minute < 13 * 60].drop(columns="ts_ny")   # vendor stops at 13:00
    f = load_ohlcv(write(tmp_path, df))
    sessions, qc = build_sessions(f, SPY, P)
    assert len(sessions) == 1 and sessions[0].expected_n == 210 and qc.early_close_sessions == 1


def test_incomplete_opening_range_excludes_the_session(tmp_path):
    days = bday_range("2024-03-04", "2024-03-06")
    df = random_walk_frame(days, extended=False)
    ny = df["timestamp"].dt.tz_convert("America/New_York")
    drop = (ny.dt.date == dt.date(2024, 3, 5)) & (ny.dt.hour == 9) & (ny.dt.minute == 33)
    f = load_ohlcv(write(tmp_path, df[~drop]))
    sessions, qc = build_sessions(f, SPY, P)
    assert [s.date.day for s in sessions] == [4, 6] and qc.sessions_excluded["incomplete_opening_range"] == 1


def test_missing_minutes_are_never_filled(tmp_path):
    days = bday_range("2024-03-04", "2024-03-05")
    df = random_walk_frame(days, extended=False)
    ny = df["timestamp"].dt.tz_convert("America/New_York")
    drop = (ny.dt.date == dt.date(2024, 3, 5)) & (ny.dt.hour == 14) & (ny.dt.minute == 10)   # outside signal window
    f = load_ohlcv(write(tmp_path, df[~drop]))
    sessions, qc = build_sessions(f, SPY, P)
    assert len(sessions) == 2
    s = sessions[1]
    assert np.isnan(s.o[(14 - 9) * 60 + 10 - 30]) and qc.missing_minutes_in_kept_sessions == 1


def test_missing_minute_inside_signal_window_excludes_session(tmp_path):
    df = random_walk_frame([dt.date(2024, 3, 5)], extended=False)
    ny = df["timestamp"].dt.tz_convert("America/New_York")
    drop = (ny.dt.hour == 10) & (ny.dt.minute == 15)
    f = load_ohlcv(write(tmp_path, df[~drop]))
    sessions, qc = build_sessions(f, SPY, P)
    assert sessions == [] and qc.sessions_excluded["missing_minutes_in_signal_window"] == 1


def test_truncated_session_excluded(tmp_path):
    df = random_walk_frame([dt.date(2024, 3, 5)], extended=False)
    ny = df["timestamp"].dt.tz_convert("America/New_York")
    f = load_ohlcv(write(tmp_path, df[ny.dt.hour < 15]))             # data stops at 15:00
    sessions, qc = build_sessions(f, SPY, P)
    assert sessions == [] and qc.sessions_excluded["truncated_session_end"] == 1


def test_duplicates_and_bad_ohlc_are_dropped(tmp_path):
    df = random_walk_frame([dt.date(2024, 3, 5)], extended=False)
    dup = pd.concat([df, df.iloc[[100]]])
    bad = df.iloc[[7]].copy()
    bad_idx = df.index[7]
    df2 = pd.concat([df.drop(index=bad_idx), bad.assign(high=bad["low"] - 1.0)])   # high < low
    f = load_ohlcv(write(tmp_path, dup, "d.csv"))
    _, qc = build_sessions(f, SPY, P)
    assert qc.rows_duplicate_dropped == 1
    f2 = load_ohlcv(write(tmp_path, df2, "b.csv"))
    sess, qc2 = build_sessions(f2, SPY, P)
    assert qc2.rows_bad_ohlc_dropped == 1 and sess == []      # bar 7 is inside the opening range window / signal window


def test_futures_one_contract_per_session_most_traded(tmp_path):
    d = [dt.date(2024, 3, 5)]
    a = random_walk_frame(d, seed=1, start=5000, extended=False, tick=0.25).assign(contract="ESH4", volume=500)
    b = random_walk_frame(d, seed=2, start=5100, extended=False, tick=0.25).assign(contract="ESM4", volume=900)
    f = load_ohlcv(write(tmp_path, pd.concat([a, b])))
    sessions, qc = build_sessions(f, ES, P)
    assert len(sessions) == 1 and sessions[0].contract == "ESM4"
    assert abs(sessions[0].o[0] - 5100) < 150               # prices come from ESM4 only
