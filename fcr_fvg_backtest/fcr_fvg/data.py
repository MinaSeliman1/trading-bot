"""Loading, quality control and session construction for 1-minute OHLCV data.

Principles
----------
* Real data only. Nothing is interpolated, forward-filled or reconstructed from
  coarser bars. A missing minute stays missing (NaN) and the strategy logic
  treats it as "no usable bar".
* All timestamps are converted to America/New_York *wall-clock* time, so EST/EDT
  is handled by the tz database. A bar is identified by its OPEN time (bar
  "09:30" covers 09:30:00-09:30:59). Use ts_convention="end" for vendors that
  stamp the bar close.
* Only the regular session 09:30-16:00 (or the shortened close from the NYSE
  calendar) is kept. Overnight / pre-market / post-market bars are dropped
  BEFORE anything else looks at the data.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .instruments import Instrument
from .params import OPEN_MINUTE, TZ, StrategyParams

RTH_MINUTES = 390                      # 09:30-16:00
CLOSE_MINUTE = OPEN_MINUTE + RTH_MINUTES

_TS_CANDIDATES = ("timestamp", "datetime", "date", "time", "ts_event", "ts", "t", "dt")
_COL_ALIASES = {
    "open": ("open", "o"), "high": ("high", "h"), "low": ("low", "l"),
    "close": ("close", "c"), "volume": ("volume", "v", "vol"),
}
_CONTRACT_CANDIDATES = ("contract", "symbol", "instrument", "raw_symbol", "ticker")


# ----------------------------------------------------------------------------
# Session container
# ----------------------------------------------------------------------------
@dataclass
class Session:
    date: dt.date
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    expected_n: int = RTH_MINUTES
    contract: str | None = None

    def bar_open_time(self, idx: int) -> pd.Timestamp:
        base = pd.Timestamp(f"{self.date} 09:30", tz=TZ)
        return base + pd.Timedelta(minutes=int(idx))

    def present(self, idx: int) -> bool:
        return 0 <= idx < RTH_MINUTES and bool(np.isfinite(self.o[idx]))

    @property
    def last_present_idx(self) -> int:
        ok = np.flatnonzero(np.isfinite(self.o))
        return int(ok[-1]) if len(ok) else -1


@dataclass
class QCReport:
    symbol: str
    source_rows: int = 0
    rows_nan_dropped: int = 0
    rows_not_minute_aligned_dropped: int = 0
    rows_duplicate_dropped: int = 0
    rows_outside_rth_dropped: int = 0
    rows_after_official_close_dropped: int = 0
    rows_bad_ohlc_dropped: int = 0
    bad_ohlc_examples: list = field(default_factory=list)
    data_start: str | None = None
    data_end: str | None = None
    calendar_used: bool = False
    calendar_sessions_expected: int = 0
    sessions_in_data: int = 0
    calendar_days_without_data: int = 0
    sessions_kept: int = 0
    sessions_excluded: dict = field(default_factory=dict)
    early_close_sessions: int = 0
    missing_minutes_in_kept_sessions: int = 0
    expected_minutes_in_kept_sessions: int = 0
    contract_changes: int = 0
    notes: list = field(default_factory=list)

    def exclude(self, reason: str):
        self.sessions_excluded[reason] = self.sessions_excluded.get(reason, 0) + 1

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _find_col(cols: list[str], candidates) -> str | None:
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def _parse_timestamps(s: pd.Series, input_tz: str | None) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(s):
        m = float(s.dropna().abs().median())
        unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
        return pd.DatetimeIndex(pd.to_datetime(s, unit=unit, utc=True))
    parsed = pd.to_datetime(s, errors="coerce", utc=False, format="mixed")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return pd.DatetimeIndex(parsed)
    if parsed.dt.tz is None:
        if input_tz is None:
            raise ValueError(
                "Timestamps are timezone-naive. Pass input_tz (e.g. 'UTC' or "
                "'America/New_York'); guessing a timezone silently would corrupt the 09:30 alignment."
            )
        return pd.DatetimeIndex(parsed.dt.tz_localize(input_tz, ambiguous="NaT", nonexistent="NaT"))
    return pd.DatetimeIndex(parsed)


def load_ohlcv(path: str | Path, input_tz: str | None = None, ts_convention: str = "start") -> pd.DataFrame:
    """Read a CSV/CSV.GZ/Parquet file of 1-minute bars into a New-York-indexed frame.

    Returns columns open/high/low/close/volume (+ 'contract' if the file has one).
    """
    if ts_convention not in ("start", "end"):
        raise ValueError("ts_convention must be 'start' or 'end'")
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    cols = list(raw.columns)

    ts_col = _find_col(cols, _TS_CANDIDATES)
    if ts_col is None and raw.index.name and str(raw.index.name).lower() in _TS_CANDIDATES + ("index",):
        raw = raw.reset_index()
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        cols = list(raw.columns)
        ts_col = _find_col(cols, _TS_CANDIDATES + (str(raw.columns[0]),))
    if ts_col is None:
        raise ValueError(f"No timestamp column found among {_TS_CANDIDATES} in {cols}")

    out = {}
    for name, aliases in _COL_ALIASES.items():
        col = _find_col(cols, aliases)
        if col is None:
            if name == "volume":
                out[name] = np.nan
                continue
            raise ValueError(f"Missing column '{name}' in {path} (have {cols})")
        out[name] = pd.to_numeric(raw[col], errors="coerce")

    idx = _parse_timestamps(raw[ts_col], input_tz).tz_convert(TZ)
    df = pd.DataFrame(out)
    df.index = idx
    ccol = _find_col(cols, _CONTRACT_CANDIDATES)
    if ccol is not None:
        df["contract"] = raw[ccol].astype(str).values
    df = df[~df.index.isna()]
    if ts_convention == "end":
        df.index = df.index - pd.Timedelta(minutes=1)
    return df.sort_index(kind="stable")


# ----------------------------------------------------------------------------
# Quality control + session construction
# ----------------------------------------------------------------------------
def _nyse_schedule(start: dt.date, end: dt.date) -> pd.DataFrame | None:
    try:
        import pandas_market_calendars as pmc
    except Exception:
        return None
    sched = pmc.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    sched = sched.copy()
    sched["close_ny"] = sched["market_close"].dt.tz_convert(TZ)
    return sched


def build_sessions(df: pd.DataFrame, inst: Instrument, params: StrategyParams,
                   use_calendar: bool = True) -> tuple[list[Session], QCReport]:
    """Clean `df` and cut it into per-day Session objects. Never fills gaps."""
    qc = QCReport(symbol=inst.symbol, source_rows=len(df))
    if df.empty:
        qc.notes.append("empty input")
        return [], qc

    df = df.copy()
    # 1. numeric / NaN
    n0 = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    qc.rows_nan_dropped = n0 - len(df)

    # 2. minute alignment (bars must start on a whole minute)
    aligned = (df.index.second == 0) & (df.index.microsecond == 0) & (df.index.nanosecond == 0)
    qc.rows_not_minute_aligned_dropped = int((~aligned).sum())
    df = df[aligned]

    # 3. regular session only (wall-clock New York) -- BEFORE contract choice or dedupe
    mod = df.index.hour * 60 + df.index.minute
    in_rth = (mod >= OPEN_MINUTE) & (mod < CLOSE_MINUTE)
    qc.rows_outside_rth_dropped = int((~in_rth).sum())
    df = df[in_rth]
    mod = mod[in_rth]

    # 4. OHLC consistency (drop the bar; it becomes a missing minute, never repaired)
    bad = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1) - 1e-9)
        | (df["low"] > df[["open", "close", "high"]].min(axis=1) + 1e-9)
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    qc.rows_bad_ohlc_dropped = int(bad.sum())
    if bad.any():
        qc.bad_ohlc_examples = [str(t) for t in df.index[bad][:5]]
    df = df[~bad]
    mod = mod[~bad.values]
    if df.empty:
        qc.notes.append("no regular-session rows left after cleaning")
        return [], qc

    dates = df.index.date
    df = df.assign(_date=dates, _idx=mod - OPEN_MINUTE)

    # 5. futures: ONE contract per session = the most traded one that day
    if "contract" in df.columns:
        vol = df.groupby(["_date", "contract"])["volume"].sum(min_count=1).fillna(0.0)
        count = df.groupby(["_date", "contract"]).size()
        score = vol + count * 1e-9          # tie-break on bar count when volume is absent
        chosen = score.groupby(level=0).idxmax().map(lambda t: t[1])
        keep = df["contract"].values == df["_date"].map(chosen).values
        df = df[keep]
    else:
        chosen = None
        if inst.kind == "future":
            qc.notes.append(
                "No contract column: a continuous futures series was assumed. Intraday "
                "roll integrity cannot be verified. Provide per-contract bars with a 'contract' "
                "column (unadjusted) for a clean run."
            )

    # 6. duplicate minutes inside a session (keep the first, count the rest)
    dup = df.duplicated(subset=["_date", "_idx"], keep="first")
    qc.rows_duplicate_dropped = int(dup.sum())
    df = df[~dup]

    qc.data_start = str(df.index.min())
    qc.data_end = str(df.index.max())
    day_list = sorted(set(df["_date"]))
    qc.sessions_in_data = len(day_list)

    sched = _nyse_schedule(day_list[0], day_list[-1]) if use_calendar else None
    qc.calendar_used = sched is not None
    if sched is None and use_calendar:
        qc.notes.append("pandas_market_calendars unavailable: holidays/early closes inferred from the data only.")
    cal_days = None
    if sched is not None:
        cal_days = {d.date() for d in sched.index}
        qc.calendar_sessions_expected = len(cal_days)
        qc.calendar_days_without_data = len(cal_days - set(day_list))

    sessions: list[Session] = []
    for d, g in df.groupby("_date", sort=True):
        # expected number of RTH minutes for that day
        if cal_days is not None:
            if d not in cal_days:
                qc.exclude("not_an_nyse_trading_day")
                continue
            close_ny = sched.loc[pd.Timestamp(d), "close_ny"]
            exp_n = int(close_ny.hour * 60 + close_ny.minute - OPEN_MINUTE)
        else:
            if d.weekday() >= 5:
                qc.exclude("weekend")
                continue
            last = int(g["_idx"].max()) + 1
            exp_n = 210 if last == 210 else RTH_MINUTES
        if exp_n < RTH_MINUTES:
            qc.early_close_sessions += 1

        idx = g["_idx"].to_numpy()
        valid = idx < exp_n          # ignore anything stamped after the official close
        if not valid.all():
            qc.rows_after_official_close_dropped += int((~valid).sum())
            g = g[valid]
            idx = idx[valid]
        arrs = {}
        for name in ("open", "high", "low", "close", "volume"):
            a = np.full(RTH_MINUTES, np.nan)
            a[idx] = g[name].to_numpy(dtype=float)
            arrs[name] = a
        s = Session(d, arrs["open"], arrs["high"], arrs["low"], arrs["close"], arrs["volume"],
                    expected_n=exp_n, contract=(chosen.get(d) if chosen is not None else None))

        # --- session-level exclusion policy (fixed before running) ---
        if not np.isfinite(s.o[:5]).all():
            qc.exclude("incomplete_opening_range")
            continue
        win = min(params.entry_cutoff_idx, exp_n)
        if int(np.isnan(s.o[:win]).sum()) > params.max_missing_signal_window:
            qc.exclude("missing_minutes_in_signal_window")
            continue
        if s.last_present_idx < exp_n - 1 - params.end_tolerance_min:
            qc.exclude("truncated_session_end")
            continue
        sessions.append(s)
        qc.missing_minutes_in_kept_sessions += int(np.isnan(s.o[:exp_n]).sum())
        qc.expected_minutes_in_kept_sessions += exp_n

    qc.sessions_kept = len(sessions)
    if chosen is not None:
        kept = [s.contract for s in sessions]
        qc.contract_changes = sum(1 for a, b in zip(kept, kept[1:]) if a != b)
    return sessions, qc
