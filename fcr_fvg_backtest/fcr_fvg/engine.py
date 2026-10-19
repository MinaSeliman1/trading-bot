"""Backtest loop: one instrument, one parameter set, sessions in chronological order."""
from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np
import pandas as pd

from .data import Session
from .execution import Rejection, Trade, simulate_trade
from .instruments import Instrument
from .params import StrategyParams
from .signals import Signal, scan_session


class SignalCache:
    """Signals do not depend on the stop variant, slippage, sizing or capital, so
    they are computed once per (session, ttl, confirm window, cutoff) and reused."""

    def __init__(self):
        self._d: dict = {}

    def get(self, s: Session, p: StrategyParams) -> list[Signal]:
        key = (s.date, id(s), p.fvg_ttl_bars, p.confirm_extra_bars, p.entry_cutoff_idx)
        if key not in self._d:
            self._d[key] = scan_session(s, p)
        return self._d[key]


def session_features(sessions: Sequence[Session]) -> pd.DataFrame:
    """Per-session descriptors that are known BEFORE the trade (no lookahead):
    opening-range width at 09:35, and the mean daily range of the previous 20 sessions."""
    rows = []
    for s in sessions:
        h5, l5 = float(np.max(s.h[:5])), float(np.min(s.l[:5]))
        mid = (h5 + l5) / 2.0
        day_hi, day_lo, day_cl = np.nanmax(s.h), np.nanmin(s.l), s.c[s.last_present_idx]
        rows.append((s.date, (h5 - l5) / mid * 100.0, (day_hi - day_lo) / day_cl * 100.0))
    f = pd.DataFrame(rows, columns=["date", "or_width_pct", "day_range_pct"]).set_index("date")
    f["prev20_range_pct"] = f["day_range_pct"].shift(1).rolling(20, min_periods=10).mean()
    return f.drop(columns="day_range_pct")


def run_backtest(sessions: Sequence[Session], inst: Instrument, p: StrategyParams,
                 cache: SignalCache | None = None) -> dict:
    """Returns dict(trades=DataFrame, rejections=DataFrame, daily=DataFrame, features=DataFrame)."""
    cache = cache or SignalCache()
    equity = p.initial_capital
    trades: list[Trade] = []
    rejections: list[Rejection] = []
    daily_rows = []

    for s in sessions:
        pnl = 0.0
        traded = False
        n_signals = 0
        for sig in cache.get(s, p):
            n_signals += 1
            res = simulate_trade(s, sig, inst, p, equity)
            if isinstance(res, Rejection):
                rejections.append(res)
                continue
            trades.append(res)
            pnl = res.net_pnl
            equity += pnl
            traded = True
            break                                  # max ONE trade per day and instrument
        daily_rows.append((s.date, pnl, equity, traded, n_signals))

    tdf = pd.DataFrame([asdict(t) for t in trades])
    if not tdf.empty:
        tdf = tdf.drop(columns=["extra"])
    rdf = pd.DataFrame([asdict(r) for r in rejections])
    ddf = pd.DataFrame(daily_rows, columns=["date", "pnl", "equity", "traded", "n_signals"]).set_index("date")
    feats = session_features(sessions)
    return {"trades": tdf, "rejections": rdf, "daily": ddf, "features": feats,
            "params": p, "instrument": inst}
