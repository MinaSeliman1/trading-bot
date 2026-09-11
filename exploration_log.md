# Exploration log

Overnight exploration run following the A-vs-B comparison. Goal: determine
whether the poor baseline results are a parameter problem or a base-logic
problem, using strict anti-overfitting rules (reject OOS Sharpe deviating
>30% from IS Sharpe; prefer stable parameter plateaus over isolated peaks;
never select/rank on win rate).

Methodology notes carried over from the main comparison: same cached data
(2022-06 to 2026-07), same walk-forward schedule (12mo train / 3mo test),
same cost scenarios, indicators warmed up on train-period bars before each
OOS test window (fixes the earlier GLD/USO zero-trade bug).

---

## Bug fixes applied before exploration

1. **Parquet cache backfill bug**: `DataClient.get_bars` only extended the
   cache forward from its cached tail, never backward -- a narrow cache
   seeded by an earlier test silently starved a later, wider request. Fixed
   to backfill both directions. Regression test added.
2. **Indicator warmup bug**: OOS test windows (3 months) were too short for
   a 200-period EMA on 4-hour GLD/USO bars to ever finish warming up when
   computed from a cold start at the window boundary -- GLD/USO showed 0 OOS
   trades, not because there was no signal but because the test literally
   couldn't start. Fixed via `active_start`: indicators now compute over
   train+test bars, entries are gated to fire only from test_start onward.
   Regression test added.
3. **Strategy A re-entry cooldown**: added a 5-bar cooldown after any
   stop-loss exit, fixing the diagnosed whipsaw failure mode (immediate
   re-entry into a still-adverse move). Fixed at 5 bars, not walk-forward
   optimized -- this is a fairness correction (Strategy B gets equivalent
   protection from regime gating), not a tunable edge. Regression test added.
4. **Uncontrolled position-stacking bug (found via the mandated sanity
   check, see below)**: capped Strategy A's aggregate gross notional at 4x
   equity, matching Alpaca's real margin multiplier.

---

## Sanity check (priority result -- run before any further exploration)

**Sizing recalculation**: confirmed correct. Pulled `equity_at_entry` across
15 sampled trades spanning the full OOS period (2023-06 to 2026-03): it
tracks the account's actual running balance (100000 -> 87724 -> 78052 ->
91152 -> 84149 -> ... -> 82818), never stuck at the fixed starting value.
`running_equity()` is called fresh at each entry decision and correctly sums
starting capital + realized P&L + unrealized P&L of open positions. No bug.

**Position stacking: real bug found and fixed.** Located the steepest 5-day
equity decline in the OOS curve (2023-07-09 -> 2023-07-14, -$7,951). Dumped
the 20 trades around it and reconstructed simultaneous open positions day by
day. Gross notional reached **200-600% of equity** with up to 5 instruments
open at once (e.g. 2023-07-20: BTCUSD+GLD+QQQ+SPY+USO all open simultaneously,
gross notional $564k against $95k equity = 594%). Root cause: Strategy A's
original spec caps each position individually at 100% of equity notional but
never caps gross exposure across instruments -- with several positions
breaching their entry threshold on the same day (common during a broad
market move), the position count stacks with no ceiling. No real account
carries anywhere near that much leverage; Alpaca's actual standard margin
multiplier is 4x (confirmed live via `TradingClient.get_account().multiplier`).

**Fix applied**: capped Strategy A's aggregate gross notional at 4x equity
(`STRATEGY_A_MAX_GROSS_EXPOSURE` in backtest_engine.py) -- new positions are
now sized down (or blocked entirely) once the portfolio is already near that
ceiling, matching what a real margin account could actually execute.
Regression test added confirming gross notional never exceeds the cap even
when 4 instruments simultaneously breach their entry threshold on the same
bar. This was the top-priority fix; the previous comparison_report.md's
Strategy A drawdown figures (max DD 99.8%, near-total account wipeout) are
almost certainly overstated by this unmodeled leverage and need to be
re-generated -- see the full re-run later in this log.

---

## Round 1: base logic, isolated, win-rate-agnostic ranking

- **SPY / baseline**: 60 (param,window) combos tested, 9 rejected by the 30% IS/OOS deviation rule. 5 distinct param sets survived.
  best plateau params={'sma_period': 20, 'std_dev_mult': 2.0}: mean_oos_sharpe=-8.543 std=0.550 -> stitched Sharpe=-8.371 PF=0.219 MaxDD=100.0% MC_DD95=100.0% trades=1797 cost0.15%=-100.0%
- **SPY / cooldown5**: 60 (param,window) combos tested, 23 rejected by the 30% IS/OOS deviation rule. 5 distinct param sets survived.
  best plateau params={'sma_period': 20, 'std_dev_mult': 2.0}: mean_oos_sharpe=-8.790 std=0.754 -> stitched Sharpe=-8.170 PF=0.235 MaxDD=100.0% MC_DD95=100.0% trades=909 cost0.15%=-100.0%
- **SPY / trendfilter200**: 60 (param,window) combos tested, 23 rejected by the 30% IS/OOS deviation rule. 5 distinct param sets survived.
  best plateau params={'sma_period': 20, 'std_dev_mult': 2.0, 'trend_sma_period': 200}: mean_oos_sharpe=-4.185 std=0.750 -> stitched Sharpe=-4.288 PF=0.235 MaxDD=99.9% MC_DD95=100.0% trades=495 cost0.15%=-100.0%
- **QQQ / baseline**: 60 (param,window) combos tested, 29 rejected by the 30% IS/OOS deviation rule. 5 distinct param sets survived.
  best plateau params={'sma_period': 20, 'std_dev_mult': 2.4}: mean_oos_sharpe=-5.995 std=0.590 -> stitched Sharpe=-5.967 PF=0.333 MaxDD=100.0% MC_DD95=100.0% trades=1065 cost0.15%=-100.0%
- **QQQ / cooldown5**: 60 (param,window) combos tested, 30 rejected by the 30% IS/OOS deviation rule. 5 distinct param sets survived.
  best plateau params={'sma_period': 20, 'std_dev_mult': 2.4}: mean_oos_sharpe=-4.885 std=1.044 -> stitched Sharpe=-5.517 PF=0.343 MaxDD=99.9% MC_DD95=100.0% trades=635 cost0.15%=-100.0%
- **QQQ / trendfilter200**: 60 (param,window) combos tested, 35 rejected by the 30% IS/OOS deviation rule. 5 distinct param sets survived.
  best plateau params={'sma_period': 20, 'std_dev_mult': 2.4, 'trend_sma_period': 200}: mean_oos_sharpe=-3.310 std=0.337 -> stitched Sharpe=-2.877 PF=0.354 MaxDD=96.5% MC_DD95=98.1% trades=293 cost0.15%=-99.4%
- **BTCUSD / baseline breakout**: 48 combos, 43 rejected by anti-overfitting rule. 1 param sets survived.
  best params={'lookback': 20, 'trailing_atr_mult': 3.0, 'volume_mult': 1.5}: Sharpe=-0.500 PF=0.830 MaxDD=83.4% MC_DD95=94.3% trades=545
- **BTCUSD / MTF-confirmed breakout**: 48 combos, 46 rejected. 0 param sets survived.

## Round 2: GLD/USO plateau-based grid tightening + volatility filter

- **GLD / trend_following wide-grid plateau search**: 72 combos, 68 rejected. Plateau ranking (mean_oos_sharpe - std):
- **GLD / trend_following_volfilter**: 72 combos, 69 rejected. 0 param sets survived.
- **USO / trend_following wide-grid plateau search**: 72 combos, 70 rejected. Plateau ranking (mean_oos_sharpe - std):
- **USO / trend_following_volfilter**: 72 combos, 66 rejected. 1 param sets survived.
  best params={'fast_ema': 50, 'slow_ema': 200, 'trailing_atr_mult': 4.0, 'vol_percentile_min': 0.25}: Sharpe=-0.324 PF=0.278 MaxDD=15.0% MC_DD95=15.2% trades=10 cost0.15%=-12.6%

## Round 1+2 complete: 8 candidates collected. See exploration_candidates.json.

**Note on GLD/USO plateau search**: the wide grid (1.5-4.0) found NO parameter
value forming a stable multi-window plateau after the 30% anti-overfitting
filter -- both baseline and volfilter variants show 0-1 survivors, and where
1 survives it's typically a single (param, window) pair, not a genuine
plateau across multiple windows. This is very likely a DATA-VOLUME problem,
not a "no edge" verdict: GLD/USO only produce 7-15 trades total over 4 years
on 4-hour bars, so per-window OOS Sharpe is dominated by sampling noise and
routinely swings past the 30% deviation threshold even when the underlying
signal hasn't changed. Contrast with the earlier looser-filtered isolated
diagnostic (before this stricter re-test), which found USO's plain baseline
at Sharpe=0.422 / PF=2.225 -- the single most promising number found all
night. That number does NOT survive the strict anti-overfitting filter
applied here, so it should be read as "a fragile, statistically thin signal
that couldn't be confirmed," not as a validated edge. GLD/USO's PARAM_GRIDS
in compare_strategies.py are left UNCHANGED (2.0/2.5/3.0/3.5) rather than
narrowed to a single value cherry-picked from a search that legitimately
found no stable winner -- doing that would itself be the exact overfitting
this exercise exists to avoid.

---

## Final ranking: top candidates found overnight, ranked by OOS Sharpe

None of the 8 tested combinations meet all three viability criteria
(OOS Sharpe >= 0.5, MC 95th-pct drawdown <= 20%, edge survives 0.15%
slippage). Ranked by OOS Sharpe (least-bad first), win rate NOT used in
ranking or selection anywhere in this exploration:

| Rank | Combination | Sharpe OOS | Profit factor | MC DD95 | Trades | Overfitting risk |
|---|---|---|---|---|---|---|
| 1 | USO trend_following + vol filter | -0.32 | 0.28 | 15.2% | 10 | HIGH -- only 10 trades total, single-window survivor, not a real plateau |
| 2 | BTCUSD momentum_breakout baseline | -0.50 | 0.83 | 94.3% | 545 | LOW -- 545 trades, stable std=0.17 across surviving windows, but drawdown alone disqualifies it |
| 3 | QQQ mean_reversion + SMA200 trend filter | -2.88 | 0.35 | 98.1% | 293 | LOW-MODERATE -- survived 5/12 windows, consistent direction of improvement vs baseline |
| 4 | SPY mean_reversion + SMA200 trend filter | -4.29 | 0.24 | 100.0% | 495 | LOW-MODERATE -- survived 9/12 windows, same consistent pattern as QQQ |
| 5 | QQQ mean_reversion + 5-bar cooldown | -5.52 | 0.34 | 100.0% | 635 | MODERATE -- marginal improvement over baseline, cooldown alone is a weak fix |

**Verdict: none viable.** The SMA200 trend filter is the one intervention
that produced a real, consistent, large effect (roughly HALVES the Sharpe
disaster on both SPY and QQQ, cuts trade count sharply, reduces drawdown on
QQQ from 100% to 96.5%) -- but it improves a catastrophic strategy into a
merely very bad one, not into a viable one. Cooldown alone is a much weaker
fix by comparison. BTC's breakout is the least-bad of the untouched base
logics but is disqualified purely on tail risk (94% Monte-Carlo drawdown).
GLD/USO's trend following could not be validated either way with the
available trade volume under a strict anti-overfitting standard.

