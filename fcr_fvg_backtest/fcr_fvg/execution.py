"""Order sizing and trade simulation on 1-minute bars.

Execution model (all conservative, none tuned):
  * Entry: market order sent at the close of the engulfing bar, filled at the
    OPEN of the next bar, worse by `slippage_ticks`.
  * Stop: stop-market. Hit when the bar's low (LONG) / high (SHORT) trades through
    the stop. Fill = stop level, or the bar's open if it gapped beyond it, then
    worse by `slippage_ticks`.
  * Take-profit: treated as market-if-touched. Hit when the bar trades through
    the target; fill = target worse by `slippage_ticks`. A gap through the target
    is NOT credited with a better price.
  * If stop and target are both inside one bar, the STOP is assumed first.
  * Still open at the last bar of the session: market exit at that bar's close,
    worse by `slippage_ticks`.
  * The entry bar itself is checked for stop/target (the position exists from its open)
    unless params.check_exit_on_entry_bar is False.
Prices are handled in integer ticks to avoid float drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .data import Session
from .instruments import Instrument
from .params import StrategyParams
from .signals import Signal

_EPS = 1e-9


@dataclass
class Rejection:
    date: object
    symbol: str
    variant: str
    reason: str
    signal_idx: int
    direction: int
    detail: str = ""


@dataclass
class Trade:
    date: object
    symbol: str
    variant: str
    direction: str
    fvg_time: object
    retest_time: object
    signal_time: object        # close of the engulfing bar (= when the order is sent)
    entry_time: object         # open of the fill bar
    entry_price: float
    stop: float
    target: float
    exit_time: object
    exit_price: float
    exit_reason: str
    units: int
    risk_ticks: int
    risk_usd_nominal: float    # units * |entry-stop| * point value (excl. costs)
    risk_pct_equity: float     # nominal risk / equity before the trade
    equity_before: float
    notional: float
    exposure_pct_equity: float
    margin_pct_equity: float
    size_capped: bool
    commissions: float
    slippage_cost: float
    gross_pnl_ideal: float     # before slippage and commissions
    net_pnl: float
    r_multiple: float
    duration_min: int
    or_high: float
    or_low: float
    zone_lo: float
    zone_hi: float
    stop_pierced_before_entry: bool
    entry_idx: int = 0
    exit_idx: int = 0
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
def stop_ticks_for(sig: Signal, inst: Instrument, p: StrategyParams) -> int:
    tick = inst.tick
    if p.stop_variant == "A":
        if sig.direction == 1:
            return int(round(sig.ext_low / tick)) - p.stop_buffer_ticks
        return int(round(sig.ext_high / tick)) + p.stop_buffer_ticks
    if sig.direction == 1:
        return int(round(sig.zone_lo / tick)) - p.stop_buffer_ticks
    return int(round(sig.zone_hi / tick)) + p.stop_buffer_ticks


def size_position(inst: Instrument, equity: float, entry_price: float, risk_ticks: int,
                  p: StrategyParams) -> tuple[int, bool, str]:
    """Whole units such that a stop-out (incl. exit slippage and commissions) costs <= risk_pct*equity.

    Returns (units, size_capped, reject_reason). units == 0 means the signal is rejected.
    """
    budget = p.risk_pct * equity
    tv = inst.tick_value
    per_unit = (risk_ticks + p.slippage_ticks) * tv + inst.round_trip_commission_per_unit()
    units_budget = int(math.floor(budget / per_unit + _EPS)) if per_unit > 0 else 0

    def worst_case(u: int) -> float:
        stop_px = entry_price  # commission price scale only matters for the ETF % cap
        return u * (risk_ticks + p.slippage_ticks) * tv + inst.commission(u, entry_price) + inst.commission(u, stop_px)

    while units_budget > 0 and worst_case(units_budget) > budget + _EPS:
        units_budget -= 1
    if units_budget < 1:
        return 0, False, "risk_budget_below_one_unit"

    if inst.kind == "etf":
        cap = int(math.floor(p.max_leverage_etf * equity / (entry_price * inst.point_value)))
    else:
        cap = int(math.floor(p.max_margin_usage * equity / inst.margin_per_unit))
    if cap < 1:
        return 0, False, "exposure_cap_below_one_unit"
    units = min(units_budget, cap)
    return units, units < units_budget, ""


def simulate_trade(s: Session, sig: Signal, inst: Instrument, p: StrategyParams,
                   equity: float) -> Trade | Rejection:
    tick = inst.tick
    d = sig.direction
    variant = p.stop_variant

    def rej(reason, detail=""):
        return Rejection(s.date, inst.symbol, variant, reason, sig.signal_idx, d, detail)

    ei = sig.signal_idx + 1
    if not s.present(ei):
        return rej("entry_bar_missing")
    if ei >= p.entry_cutoff_idx:                                   # defensive; scan already guarantees it
        return rej("after_entry_cutoff")

    slip = p.slippage_ticks
    open_t = float(s.o[ei]) / tick
    entry_t = int(round(open_t)) + d * slip
    stop_t = stop_ticks_for(sig, inst, p)
    risk_t = d * (entry_t - stop_t)
    if risk_t <= 0:
        return rej("invalid_stop", f"entry_ticks={entry_t} stop_ticks={stop_t}")
    tp_dist = int(math.ceil(p.rr * risk_t - _EPS))
    tp_t = entry_t + d * tp_dist

    entry_price = entry_t * tick
    units, capped, why = size_position(inst, equity, entry_price, risk_t, p)
    if units < 1:
        return rej(why)

    # Was the (future) stop level already traded through between retest and signal? (diagnostic only)
    pierced = (sig.ext_low <= stop_t * tick + _EPS) if d == 1 else (sig.ext_high >= stop_t * tick - _EPS)

    # ------------------------------------------------------------------ exits
    last = min(s.last_present_idx, s.expected_n - 1)
    exit_t = None
    reason = ""
    exit_idx = last
    for j in range(ei, last + 1):
        if not np.isfinite(s.o[j]):
            continue                      # no trade printed in that minute -> nothing can be touched
        if j == ei and not p.check_exit_on_entry_bar:
            continue
        ot, ht, lt = s.o[j] / tick, s.h[j] / tick, s.l[j] / tick
        if d == 1:
            stop_hit = lt <= stop_t + _EPS
            tp_hit = ht >= tp_t - _EPS
            if stop_hit:                                              # stop wins every ambiguity
                exit_t = int(round(min(stop_t, ot))) - slip
                reason = "gap_stop" if ot < stop_t - _EPS else "stop"
            elif tp_hit:
                exit_t = tp_t - slip
                reason = "tp"
        else:
            stop_hit = ht >= stop_t - _EPS
            tp_hit = lt <= tp_t + _EPS
            if stop_hit:
                exit_t = int(round(max(stop_t, ot))) + slip
                reason = "gap_stop" if ot > stop_t + _EPS else "stop"
            elif tp_hit:
                exit_t = tp_t + slip
                reason = "tp"
        if reason:
            exit_idx = j
            break
    if not reason:
        exit_t = int(round(s.c[last] / tick)) - d * slip
        reason = "eod"
        exit_idx = last

    exit_price = exit_t * tick
    pv = inst.point_value
    gross_fill = d * (exit_t - entry_t) * tick * pv * units
    comm = inst.commission(units, entry_price) + inst.commission(units, exit_price)
    net = gross_fill - comm
    risk_usd = units * risk_t * tick * pv
    slip_cost = 2 * slip * tick * pv * units
    notional = units * entry_price * pv
    margin_pct = (units * inst.margin_per_unit / equity) if inst.kind == "future" else 0.0

    exit_clock = (s.bar_open_time(exit_idx) + (np.timedelta64(1, "m") if reason == "eod" else np.timedelta64(0, "m")))
    return Trade(
        date=s.date, symbol=inst.symbol, variant=variant, direction="LONG" if d == 1 else "SHORT",
        fvg_time=s.bar_open_time(sig.fvg_idx) + np.timedelta64(1, "m"),
        retest_time=s.bar_open_time(sig.retest_idx),
        signal_time=s.bar_open_time(sig.signal_idx) + np.timedelta64(1, "m"),
        entry_time=s.bar_open_time(ei), entry_price=entry_price,
        stop=stop_t * tick, target=tp_t * tick,
        exit_time=exit_clock, exit_price=exit_price, exit_reason=reason,
        units=units, risk_ticks=risk_t, risk_usd_nominal=risk_usd,
        risk_pct_equity=risk_usd / equity, equity_before=equity,
        notional=notional, exposure_pct_equity=notional / equity, margin_pct_equity=margin_pct,
        size_capped=capped, commissions=comm, slippage_cost=slip_cost,
        gross_pnl_ideal=net + comm + slip_cost, net_pnl=net, r_multiple=net / risk_usd,
        duration_min=int(exit_idx - ei + (1 if reason == "eod" else 0)),
        or_high=sig.or_high, or_low=sig.or_low, zone_lo=sig.zone_lo, zone_hi=sig.zone_hi,
        stop_pierced_before_entry=bool(pierced), entry_idx=ei, exit_idx=exit_idx,
    )
