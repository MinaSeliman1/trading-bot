"""Performance, risk and robustness statistics. Every number is computed from the
trade list / daily series produced by the engine; nothing is estimated by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- primitives
def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def max_drawdown(equity: pd.Series, start_equity: float) -> tuple[float, float]:
    """(max drawdown in USD, max drawdown in % of the running peak). Equity includes the start value."""
    e = np.concatenate([[start_equity], equity.to_numpy(dtype=float)])
    peak = np.maximum.accumulate(e)
    dd = peak - e
    pct = dd / peak
    return float(dd.max()), float(pct.max() * 100.0)


def drawdown_series(equity: pd.Series, start_equity: float) -> pd.Series:
    """Drawdown in % below the running peak (<= 0), same index as `equity`."""
    peak = np.maximum.accumulate(np.maximum(equity.to_numpy(dtype=float), start_equity))
    return pd.Series((equity.to_numpy(dtype=float) - peak) / peak * 100.0, index=equity.index)


def longest_streak(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def split_dates(all_dates: list, frac: float = 0.7):
    """Chronological cut: the first `frac` of the sorted unique session dates are IS."""
    d = sorted(set(all_dates))
    k = int(np.floor(len(d) * frac))
    return d[k]          # first OOS date


# ------------------------------------------------------------------ summary
def summarize(trades: pd.DataFrame, daily: pd.DataFrame, initial_capital: float) -> dict:
    """Core metrics for ONE (instrument, variant[, segment]). `daily` is restricted to the same sessions."""
    n_sessions = len(daily)
    out = {"sessions": n_sessions, "trades": int(len(trades))}
    if n_sessions:
        start_eq = float(daily["equity"].iloc[0] - daily["pnl"].iloc[0])
    else:
        start_eq = initial_capital
    out["start_equity"] = start_eq
    if trades.empty:
        out.update(dict(wins=0, losses=0, win_rate=np.nan, profit_factor=np.nan, net_pnl=0.0, net_return_pct=0.0,
                        expectancy_r=np.nan, avg_win_r=np.nan, avg_loss_r=np.nan, payoff_ratio=np.nan,
                        max_dd_usd=0.0, max_dd_pct=0.0, longest_loss_streak=0, longest_win_streak=0,
                        avg_risk_pct_equity=np.nan, max_risk_pct_equity=np.nan, max_exposure_pct_equity=np.nan,
                        max_margin_pct_equity=np.nan, trades_per_week=0.0, avg_duration_min=np.nan,
                        commissions=0.0, slippage_cost=0.0, size_capped_trades=0))
        return out
    pnl, r = trades["net_pnl"], trades["r_multiple"]
    wins, losses = int((pnl > 0).sum()), int((pnl < 0).sum())
    avg_win = float(r[pnl > 0].mean()) if wins else np.nan
    avg_loss = float(r[pnl < 0].mean()) if losses else np.nan
    dd_usd, dd_pct = max_drawdown(daily["equity"], start_eq)
    out.update(
        wins=wins, losses=losses, win_rate=wins / len(trades) * 100.0,
        profit_factor=profit_factor(pnl), net_pnl=float(pnl.sum()),
        net_return_pct=float(pnl.sum() / start_eq * 100.0),
        expectancy_r=float(r.mean()), avg_win_r=avg_win, avg_loss_r=avg_loss,
        payoff_ratio=(avg_win / abs(avg_loss)) if wins and losses else np.nan,
        max_dd_usd=dd_usd, max_dd_pct=dd_pct,
        longest_loss_streak=longest_streak((pnl < 0).to_numpy()),
        longest_win_streak=longest_streak((pnl > 0).to_numpy()),
        avg_risk_pct_equity=float(trades["risk_pct_equity"].mean() * 100.0),
        max_risk_pct_equity=float(trades["risk_pct_equity"].max() * 100.0),
        max_exposure_pct_equity=float(trades["exposure_pct_equity"].max() * 100.0),
        max_margin_pct_equity=float(trades["margin_pct_equity"].max() * 100.0),
        trades_per_week=len(trades) / (n_sessions / 5.0) if n_sessions else 0.0,
        avg_duration_min=float(trades["duration_min"].mean()),
        commissions=float(trades["commissions"].sum()), slippage_cost=float(trades["slippage_cost"].sum()),
        size_capped_trades=int(trades["size_capped"].sum()),
    )
    return out


def group_stats(t: pd.DataFrame) -> pd.Series:
    if t.empty:
        return pd.Series({"trades": 0, "win_rate": np.nan, "expectancy_r": np.nan, "profit_factor": np.nan, "net_pnl": 0.0})
    return pd.Series({
        "trades": len(t), "win_rate": (t["net_pnl"] > 0).mean() * 100.0,
        "expectancy_r": t["r_multiple"].mean(), "profit_factor": profit_factor(t["net_pnl"]),
        "net_pnl": t["net_pnl"].sum(),
    })


# --------------------------------------------------------------- breakdowns
def by_month(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    m = pd.to_datetime(trades["date"]).dt.to_period("M").astype(str)
    return trades.groupby(m).apply(group_stats, include_groups=False)


def by_period(trades: pd.DataFrame, freq: str = "Y") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    if freq == "H":                                   # half-years
        d = pd.to_datetime(trades["date"])
        key = d.dt.year.astype(str) + "-H" + ((d.dt.month - 1) // 6 + 1).astype(str)
    else:
        key = pd.to_datetime(trades["date"]).dt.to_period(freq).astype(str)
    return trades.groupby(key).apply(group_stats, include_groups=False)


def by_direction(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("direction").apply(group_stats, include_groups=False)


def by_entry_slot(trades: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    et = pd.to_datetime(trades["entry_time"])
    mod = et.dt.hour * 60 + et.dt.minute
    lo = (mod // minutes) * minutes
    label = lo.map(lambda x: f"{x // 60:02d}:{x % 60:02d}")
    return trades.groupby(label).apply(group_stats, include_groups=False)


def by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("exit_reason").apply(group_stats, include_groups=False)


def by_tercile(trades: pd.DataFrame, feats: pd.DataFrame, col: str) -> pd.DataFrame:
    """Descriptive: terciles of a pre-trade feature computed over ALL sessions of the instrument.
    Thresholds use the whole sample, so this is a post-hoc description, not a tradable filter."""
    if trades.empty or col not in feats:
        return pd.DataFrame()
    f = feats[col].dropna()
    if len(f) < 9:
        return pd.DataFrame()
    q1, q2 = f.quantile([1 / 3, 2 / 3])
    v = trades["date"].map(feats[col])
    lab = pd.cut(v, [-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])
    res = trades.groupby(lab, observed=True).apply(group_stats, include_groups=False)
    res.attrs["thresholds"] = (float(q1), float(q2))
    return res


# --------------------------------------------------------- uncertainty / luck
def daily_r_series(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    d = daily[["pnl"]].copy()
    d["r"] = 0.0
    d["trade"] = 0
    if not trades.empty:
        tr = trades.set_index("date")
        d.loc[tr.index, "r"] = tr["r_multiple"]
        d.loc[tr.index, "trade"] = 1
    return d


def bootstrap_sessions(trades: pd.DataFrame, daily: pd.DataFrame, n_boot: int = 5000, seed: int = 12345) -> dict:
    """Resample SESSIONS (days, including days with no trade) with replacement.
    Keeps the day as the independent unit; trades are never resampled individually."""
    d = daily_r_series(trades, daily)
    n = len(d)
    if n < 20 or d["trade"].sum() < 5:
        return {"n_boot": 0, "note": "sample too small for a meaningful bootstrap"}
    r = d["r"].to_numpy(); tr = d["trade"].to_numpy(); pnl = d["pnl"].to_numpy()
    rng = np.random.default_rng(seed)
    exp_r, tot_r, pf = [], [], []
    done = 0
    while done < n_boot:
        b = min(500, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        rs, ts, ps = r[idx].sum(1), tr[idx].sum(1), pnl[idx]
        ok = ts > 0
        exp_r.append(np.where(ok, rs / np.maximum(ts, 1), np.nan))
        tot_r.append(rs)
        g, lo = np.where(ps > 0, ps, 0).sum(1), -np.where(ps < 0, ps, 0).sum(1)
        pf.append(np.where(lo > 0, g / np.where(lo > 0, lo, 1), np.nan))
        done += b
    exp_r, tot_r, pf = np.concatenate(exp_r), np.concatenate(tot_r), np.concatenate(pf)
    q = lambda a: tuple(np.nanpercentile(a, [2.5, 50, 97.5]).round(4))
    # naive t-stat on daily R (days with no trade count as 0) for reference
    sd = r.std(ddof=1)
    return {
        "n_boot": n_boot, "expectancy_r_ci95": q(exp_r), "total_r_ci95": q(tot_r), "profit_factor_ci95": q(pf),
        "p_expectancy_le_0": float(np.nanmean(exp_r <= 0)),
        "daily_r_t_stat": float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan"),
    }


def concentration(trades: pd.DataFrame, top_k: int = 5) -> dict:
    """How much of the result depends on a handful of winning days."""
    if trades.empty:
        return {}
    pnl = trades["net_pnl"]
    r = trades["r_multiple"].sort_values(ascending=False)
    gains = pnl[pnl > 0].sum()
    top = pnl.sort_values(ascending=False).head(top_k)
    return {
        "top_k": top_k,
        "top_k_share_of_gross_gains_pct": float(top.sum() / gains * 100.0) if gains > 0 else float("nan"),
        "net_pnl_without_top_k": float(pnl.sum() - top.sum()),
        "total_r": float(r.sum()),
        "total_r_without_top_k": float(r.iloc[top_k:].sum()),
        "expectancy_r_without_top_k": float(r.iloc[top_k:].mean()) if len(r) > top_k else float("nan"),
    }
