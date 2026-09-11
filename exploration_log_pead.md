# PEAD (post-earnings announcement drift) exploration log

## Data source investigation

- **Alpaca**: NOT VIABLE. Verified directly via the SDK (`alpaca.data.enums.CorporateActionsType`)
  -- the "Corporate Actions" API covers splits, dividends, mergers, spin-offs,
  name changes only. No earnings surprise / analyst estimate data anywhere
  in the Alpaca API surface.
- **Financial Modeling Prep**: has an Earnings Surprises endpoint (claims up
  to 30yr depth) but confirmed via search that it requires a paid plan --
  the free 250 req/day tier does not include it. Could not verify a live
  sample (site returns 403 to automated fetches).
- **EODHD**: same field structure (epsEstimate/epsDifference/surprisePercent)
  but free tier is only 20 req/day (weaker than Alpha Vantage's 25) and
  fundamentals specifically requires a $59.99/mo add-on plan. Not verified live.
- **Polygon.io / Massive** (recently rebranded): added "unified fundamentals
  endpoints" very recently per their own announcement; pricing page is
  JS-rendered and inaccessible to fetch tools. Not evaluated further.
- **Alpha Vantage: CONFIRMED, verified with real data.** EARNINGS endpoint
  returns `reportedDate`, `reportedEPS`, `estimatedEPS`, `surprise`,
  `surprisePercentage`, `reportTime` (pre/post-market) per quarter, one
  API call per symbol returns the FULL history. Verified via the public
  `demo` key (IBM, 121 quarters, 1996-2026) and then via the user's own
  free key (JNJ, same structure, same depth). Free tier: 25 requests/day,
  5/minute. Real key: `<REDACTED-SEE-ENV>`, stored in `.env` as
  `ALPHA_VANTAGE_API_KEY`.

## Sample universe (23 tickers, deliberately NOT mega-cap tech)

Financials: JPM, BAC, GS, V | Healthcare: JNJ, UNH, PFE, ABBV | Staples: PG,
KO, WMT, COST | Discretionary: MCD, HD, DIS | Industrials: CAT, HON, UPS |
Energy: XOM, CVX | Materials: LIN | Utilities: DUK | Communications: VZ

All 23 fetched successfully (54-121 quarters each depending on listing
history), cached in `earnings_cache/`. Daily price bars (Alpaca, 2020-07-27
to now) fetched for all 23, consistent depth.

---

## Preliminary PEAD backtest: 23 tickers, tradeable window 2020-07-27 to 2026-05-28

(First run attempt crashed here on a real bug: PEAD hold periods of up to 60
trading days routinely spill past a walk-forward window's test_end boundary,
so consecutive windows' equity curves overlap in calendar time -- stitching
them the same way the bar-based backtest does produced duplicate timestamps
and crashed `compute_metrics`. Fixed by building one equity curve from the
complete combined OOS trade list instead of stitching per-window curves --
the underlying trades were already correctly non-duplicated across windows.
Regression-tested with a synthetic overlapping-hold scenario before re-running.)

- 544 total quarterly earnings events across all tickers in the tradeable window (price data available for the hold period).
- Signal: long-only, raw surprise_pct >= threshold (grid: [3.0, 5.0, 7.0, 10.0]), hold [20, 40, 60] trading days (grid search per window, walk-forward optimized). Fixed $10,000 notional per trade (no compounding/portfolio sizing yet -- preliminary edge-existence check only, as requested).
- Walk-forward: 24mo train / 6mo test (wider than the usual 12/3 given quarterly event frequency -- ensures a meaningful event count per window). 7 windows generated.
- Window 0 (train 2020-07-27-2022-07-27, test 2022-07-27-2023-01-27): best params thresh=10.0% hold=60d, IS Sharpe=8.127, OOS Sharpe=-4.297, OOS trades=9, FLAG (>30% IS/OOS deviation)
- Window 1 (train 2021-01-27-2023-01-27, test 2023-01-27-2023-07-27): best params thresh=3.0% hold=60d, IS Sharpe=4.499, OOS Sharpe=-5.404, OOS trades=27, FLAG (>30% IS/OOS deviation)
- Window 2 (train 2021-07-27-2023-07-27, test 2023-07-27-2024-01-27): best params thresh=3.0% hold=20d, IS Sharpe=0.351, OOS Sharpe=5.633, OOS trades=21, FLAG (>30% IS/OOS deviation)
- Window 3 (train 2022-01-27-2024-01-27, test 2024-01-27-2024-07-27): best params thresh=3.0% hold=20d, IS Sharpe=0.725, OOS Sharpe=5.271, OOS trades=24, FLAG (>30% IS/OOS deviation)
- Window 4 (train 2022-07-27-2024-07-27, test 2024-07-27-2025-01-27): best params thresh=3.0% hold=60d, IS Sharpe=5.358, OOS Sharpe=1.378, OOS trades=20, FLAG (>30% IS/OOS deviation)
- Window 5 (train 2023-01-27-2025-01-27, test 2025-01-27-2025-07-27): best params thresh=7.0% hold=60d, IS Sharpe=5.108, OOS Sharpe=6.650, OOS trades=16, FLAG (>30% IS/OOS deviation)
- Window 6 (train 2023-07-27-2025-07-27, test 2025-07-27-2026-01-27): best params thresh=7.0% hold=40d, IS Sharpe=7.347, OOS Sharpe=5.893, OOS trades=14, OK

### Combined out-of-sample result across all 7 windows

- Sharpe=2.932  Sortino=4.189  MaxDD=7.5%  MC_DD95=80.1%  PF=1.458  WinRate=56.5%  Trades=131
- Cost sensitivity: 0.05%->19.9%, 0.10%->18.6%, 0.15%->17.3%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: False | edge survives 0.15% slippage: True | 6/7 windows flagged for IS/OOS deviation >30%**

**PRELIMINARY VERDICT: DOES NOT pass all 3 viability criteria**

## Part 1: empirical convergence -- how many pooled trades to get MC_DD95 under 20%?

- Base sample: 131 observed OOS trade returns (mean=1.42%, std=9.88%, min=-22.0%, max=37.7%).
- Method: resample AT the target size N with replacement FROM this same empirical return distribution (1000 resamples per N), track how the resulting MC 95th-pct drawdown moves. **Explicit assumption**: this assumes tickers added later produce trades with a statistically similar per-trade return distribution to the current 23 -- reasonable but not guaranteed; it is the best estimate possible without the additional data itself.
    N=  131 trades -> MC_DD95= 80.1%
    N=  150 trades -> MC_DD95= 82.1%
    N=  200 trades -> MC_DD95= 84.4%
    N=  250 trades -> MC_DD95= 86.6%
    N=  300 trades -> MC_DD95= 88.0%
    N=  400 trades -> MC_DD95= 89.0%
    N=  500 trades -> MC_DD95= 91.3%
    N=  700 trades -> MC_DD95= 92.5%
    N= 1000 trades -> MC_DD95= 93.9%
    N= 1500 trades -> MC_DD95= 95.0%
    N= 2000 trades -> MC_DD95= 95.4%
    N= 3000 trades -> MC_DD95= 95.9%
    N= 4000 trades -> MC_DD95= 96.7%
    N= 5000 trades -> MC_DD95= 97.1%

- **Does NOT cross below 20% even at N=5000 trades** under this resampling -- the tail-risk problem may be structural (a genuinely fat-tailed per-trade return distribution) rather than a sample-size problem alone. More data would still narrow the CONFIDENCE INTERVAL around the drawdown estimate, but may not lower the point estimate itself.

- Observed rate: 131 trades / 23 tickers = 5.70 trades/ticker over this walk-forward setup.

## Part 2: finer parameter grid search for a stable plateau

- Grid: threshold in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0] (9 values) x hold_days in [10, 15, 20, 30, 40, 50, 60, 80] (8 values) = 72 combos x 7 windows = 504 (param,window) evaluations. Same 30% IS/OOS deviation rule, same viability bar.
- 504 total (param,window) combos evaluated. 420 rejected by the 30% anti-overfitting rule.
- Param combos with a plateau across >=3/7 windows: 4
    thresh=2.0% hold=80d: n_windows=4 mean_oos_sharpe=10.554 std=0.659 total_trades=114
    thresh=4.0% hold=80d: n_windows=3 mean_oos_sharpe=9.223 std=1.221 total_trades=64
    thresh=5.0% hold=80d: n_windows=3 mean_oos_sharpe=9.526 std=1.817 total_trades=55
    thresh=8.0% hold=50d: n_windows=3 mean_oos_sharpe=4.514 std=1.832 total_trades=37

**Read with strong skepticism, not as a lead**: the top "plateau" (thresh=2%,
hold=80d) shows mean OOS Sharpe=10.55 on only 114 trades from a 72-combo
search. A Sharpe over 10 is not a plausible genuine trading edge at any
real-world scale -- it is the classic signature of a small sample surviving
a wide grid search by chance (72 combos x 7 windows = 504 draws; even a
fairly strict ">=3 window" survival bar will let a few spuriously "stable"
combos through). Also notable: all 4 surviving combos cluster at hold=50-80
days, OUTSIDE the originally-requested 20-60 day range -- another sign this
is the search finding noise at the edge of the grid rather than a real
signal. Not treated as a validated finding.

## Diagnostic: root cause of the non-converging drawdown (verified, not guessed)

Part 1's finding (MC_DD95 WORSENS with more trades, from 80.1% at N=131 to
97.1% at N=5000) rules out "just needs a bigger sample" and points at
something structural. Reconstructed the actual concurrent-position timeline
from the 131 real OOS trades: **up to 17 positions open simultaneously,
$170,000 notional exposure against $100,000 starting capital (1.7x) on
2023-05-02.** Root cause: `build_event_trades` sizes every trade at a fixed
$10,000 regardless of how many other positions are already open, and
earnings reports cluster heavily in "earnings season" weeks (many
companies of different sectors report within days of each other each
quarter) -- so the portfolio can and does stack well beyond its notional
capital during busy weeks. This is the same PATTERN as the uncontrolled
gross-exposure bug found and fixed in Strategy A during round 1 (there:
no cap on simultaneously-open ATR-sized positions; here: no cap on
simultaneously-open fixed-notional event trades), just via a different
mechanism. It plausibly explains the persistent tail risk directly: during
a bad earnings season, many of the ~17 concurrent positions can be exposed
to correlated risk (broad market conditions affecting many reporting
companies at once) at the same time, and Monte Carlo resampling surfaces
unlucky orderings where that clustering coincides with underperformance --
a risk a single historical path doesn't have to reveal, but that becomes
more (not less) likely to matter as trade count grows, since more trades
means more opportunities for stacking during any given busy period.

**This means the productive next step is NOT (only) collecting more tickers
on the free tier -- it's adding a portfolio-level exposure cap to the PEAD
engine and re-testing on the data already in hand, which costs zero
additional API calls and could resolve the drawdown problem directly.**

## Re-test with portfolio-level exposure cap (max_gross_exposure=1.5x, max_new_positions_per_day=3, priority by surprise magnitude)

- Identical setup to the original run: same 23 tickers, same grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], same 7 walk-forward windows (24mo train / 6mo test). Only change: trades now go through `build_portfolio_event_trades` instead of independent per-ticker sizing.
- Window 0: params thresh=10.0% hold=60d, IS Sharpe=8.172, OOS Sharpe=-4.297, OOS trades=9, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=7.0% hold=20d, IS Sharpe=4.472, OOS Sharpe=-8.842, OOS trades=16, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=3.0% hold=20d, IS Sharpe=0.273, OOS Sharpe=5.633, OOS trades=21, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=3.0% hold=20d, IS Sharpe=0.662, OOS Sharpe=5.271, OOS trades=24, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=3.0% hold=60d, IS Sharpe=5.572, OOS Sharpe=1.378, OOS trades=20, FLAG (>30% IS/OOS deviation)
- Window 5: params thresh=7.0% hold=60d, IS Sharpe=5.108, OOS Sharpe=6.650, OOS trades=16, FLAG (>30% IS/OOS deviation)
- Window 6: params thresh=7.0% hold=40d, IS Sharpe=7.347, OOS Sharpe=5.893, OOS trades=14, OK

### Combined out-of-sample result (exposure-capped)

- Sharpe=2.409  Sortino=3.234  MaxDD=9.2%  **MC_DD95=13.0%**  PF=1.435  WinRate=55.0%  Trades=120  (6/7 windows flagged)
- Cost sensitivity (checked separately, same window params, no re-optimization): 0.05%->+17.2%, 0.10%->+15.9%, 0.15%->+14.7% (120 trades throughout)

**Comparison: uncapped MC_DD95 was 80.1% (131 trades) -> capped MC_DD95 is 13.0% (120 trades).**

**CONCLUSION: the drawdown problem was almost entirely structural/sizing-driven, not the underlying
PEAD signal.** Capping gross exposure at 1.5x with a 3-new-positions/day limit and surprise-magnitude
priority dropped MC_DD95 by 67 percentage points while barely touching Sharpe (2.93->2.41), profit
factor (1.46->1.44), win rate (56.5%->55.0%), or trade count (131->120, only ~8% of trades skipped
to enforce the cap). **All three formal viability criteria now pass: Sharpe>=0.5 (2.41), MC_DD95<=20%
(13.0%), edge survives 0.15% slippage (+14.7%).**

**Important caveat, not resolved by this fix**: 6 of 7 windows are STILL flagged for >30% IS/OOS
Sharpe deviation -- identical to before capping. The exposure cap fixed the TAIL-RISK/drawdown
problem but did NOT fix the underlying parameter instability; the best (threshold, hold_days) choice
still swings a lot window to window, and OOS Sharpe still deviates sharply from IS Sharpe in most
windows. This is the same category of warning sign that predicted round 2's finding wouldn't survive
a genuinely fresh time period. Unlike round 2, there is no unused earlier period left to test against
here -- 2020-07-27 (the earliest available daily price history) is already the start of window 0's
training data, so the walk-forward already uses the full available data frontier. The next
independent check available is a genuinely fresh, independently-drawn TICKER set (continuing the
free-tier drip-feed with the NOW-FIXED capped engine), not a fresh time window.

---

## Regime pattern investigation: what distinguishes the 2 failing windows from the 5 passing ones

No new API calls -- reused SPY daily bars (already cached from round 2) and
`classify_regime_periods` (already built and tested in round 2) to label
each day of every PEAD window's TEST period as BULLISH/CORRECTION/
VOL_SHOCK/NEUTRAL, then cross-referenced against each window's OOS Sharpe.

| Window | Test period | OOS Sharpe | Dominant regime | Avg realized vol |
|---|---|---|---|---|
| 0 | 2022-07-27 to 2023-01-27 | **-4.30** | CORRECTION (100%) | 22.8% |
| 1 | 2023-01-27 to 2023-07-27 | **-8.84** | CORRECTION (74%) | 13.9% |
| 2 | 2023-07-27 to 2024-01-27 | +5.63 | BULLISH (72%) | 11.5% |
| 3 | 2024-01-27 to 2024-07-27 | +5.27 | BULLISH (100%) | 10.7% |
| 4 | 2024-07-27 to 2025-01-27 | +1.38 | BULLISH (98%) | 14.1% |
| 5 | 2025-01-27 to 2025-07-27 | +6.65 | BULLISH (49%, mixed) | 20.1% |
| 6 | 2025-07-27 to 2026-01-27 | +5.89 | BULLISH (100%) | 10.6% |

**Clean, not-random pattern**: both windows with NEGATIVE OOS Sharpe (the
true failures, not just IS/OOS-flagged-but-positive) are CORRECTION-
dominated. All five windows with positive OOS Sharpe are BULLISH-dominated.
0 of 5 positive windows are correction-dominated; 2 of 2 negative windows
are. Window 1's test period includes the March 2023 regional banking
crisis (SVB/Signature/Credit Suisse) specifically.

**Interpretation**: PEAD as tested (buy positive earnings surprises,
hold 20-60 days) appears to depend on a broadly rising/trending market to
work -- during a correction, even genuinely good earnings surprises seem to
get dragged down by negative market beta, overwhelming the stock-specific
signal. This is a real, if small-sample (n=7, 2 failures), regime pattern,
not random noise scattered across periods.

**Explicit caveat requested and worth repeating: this does NOT mean the
signal is regime-robust just because a fresh ticker universe validates it.**
Testing on new tickers (the drip-feed now underway) validates generalization
ACROSS TICKERS -- it says nothing about generalization ACROSS MARKET
REGIMES, since every ticker's history, old or new, is drawn from the same
overall 2020-2026 window (5 of 7 sub-periods bullish). A stable, positive
result on 100 fresh tickers would still be consistent with "this works
in bull markets and would fail again in the next correction" -- the two
forms of validation are independent and this analysis only speaks to the
regime question, not the drip-feed.

## Universe expansion drip-feed (started)

Objective random draw (same rigor as round 2's independent-universe check):
`bot/sp500_pool.py`'s 278-ticker pool, minus the 24 tickers already tested
for PEAD (23 original + AXP used as today's quota probe), drawn via
`np.random.default_rng(seed=20260703)`, target batch = 75 new tickers
(documented in `drip_feed_pead.py`, fully reproducible).

**Day 1 result: 0 fetched.** Today's 25-request quota was already consumed
by the original 23-ticker fetch + the JNJ real-key verification + the AXP
quota probe (25 total) before the drip-feed started. The target list (75
tickers) is committed and will resume automatically from where it left off
on subsequent days -- nothing needs to be re-fetched. At 25/day: ~3 more
days to complete this batch.

## Regime-filtered re-test: block new entries when SPY is in CORRECTION

- Identical setup to the capped-only run (same 23 tickers, same grid, same 7 windows, same 1.5x exposure cap / 3-per-day / surprise-priority). Only addition: no new position opens on a day SPY's regime label (via `classify_regime_periods`, same classifier used for the diagnostic) is CORRECTION.
- Window 0: params thresh=10.0% hold=60d, IS Sharpe=9.820, OOS Sharpe=nan, OOS trades=0, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=10.0% hold=40d, IS Sharpe=7.244, OOS Sharpe=-21.736, OOS trades=2, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=10.0% hold=40d, IS Sharpe=0.795, OOS Sharpe=-3.295, OOS trades=4, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=7.0% hold=40d, IS Sharpe=-0.606, OOS Sharpe=14.753, OOS trades=14, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=7.0% hold=40d, IS Sharpe=6.799, OOS Sharpe=2.393, OOS trades=12, FLAG (>30% IS/OOS deviation)
- Window 5: params thresh=7.0% hold=20d, IS Sharpe=5.650, OOS Sharpe=2.710, OOS trades=12, FLAG (>30% IS/OOS deviation)
- Window 6: params thresh=7.0% hold=40d, IS Sharpe=6.263, OOS Sharpe=5.893, OOS trades=14, OK

### Combined out-of-sample result (exposure-capped + regime-filtered)

- Sharpe=4.700  Sortino=7.484  MaxDD=2.8%  MC_DD95=6.2%  PF=1.875  WinRate=62.1%  Trades=58  (6/7 windows flagged)
- Cost sensitivity: 0.05%->12.9%, 0.10%->12.3%, 0.15%->11.7%

**Comparison table**
| | Uncapped | Capped only | Capped + regime-filtered |
|---|---|---|---|
| Sharpe | 2.93 | 2.41 | 4.70 |
| MC_DD95 | 80.1% | 13.0% | 6.2% |
| Trades | 131 | 120 | 58 |
| Windows flagged | 6/7 | 6/7 | 6/7 |
| Cost@0.15% survival | +17.3% | +14.7% | +11.7% |

**Honest read: NOT the clean "eliminates losses, preserves gains" story hoped
for.** Per-window OOS Sharpe, capped-only -> capped+filtered:

| Window (test period regime) | Before filter | After filter |
|---|---|---|
| 0 (CORRECTION 100%) | -4.30 | N/A (0 trades -- filter blocked ALL entries) |
| 1 (CORRECTION 74%) | -8.84 | **-21.74** (worse, but only 2 trades left) |
| 2 (BULLISH 72%) | +5.63 | **-3.30** (a previously-GOOD window got WORSE) |
| 3 (BULLISH 100%) | +5.27 | +14.75 (much better) |
| 4 (BULLISH 98%) | +1.38 | +2.39 |
| 5 (BULLISH 49%, mixed) | +6.65 | +2.71 (worse) |
| 6 (BULLISH 100%) | +5.89 | +5.89 (unchanged -- filter never bound here) |

What actually happened: the filter didn't cleanly excise the correction-
period losses and leave everything else intact. Window 0 avoided its loss
by simply not trading at all (58% fewer trades project-wide, 131->58) --
which is a form of risk avoidance, but also means the strategy is now
largely idle during exactly the periods it would need to prove itself
robust in. Window 1 is still bad, now on an almost-uninterpretable 2-trade
sample. Window 2 -- a window the diagnostic did NOT flag as a failure --
got WORSE after filtering, showing the regime label isn't a perfect
proxy for "will this trade work"; it's removing some good trades along
with bad ones. Windows-flagged count is UNCHANGED (6/7) -- the filter
improved aggregate drawdown/Sharpe substantially but did not resolve the
underlying window-to-window parameter instability, and the much smaller
sample (58 trades, down from 120) makes every remaining window's Sharpe
estimate noisier, not more reliable.

**Bottom line**: this is a genuine improvement on the aggregate viability
numbers (all 3 criteria now pass more comfortably: Sharpe 4.70, MC_DD95
6.2%, cost-survival +11.7%) and confirms the regime hypothesis has some
real predictive content (window 0's total avoidance, window 3's big
improvement). But it is NOT the clean confirmation that would let this be
called "the most solid result of the project" -- the mechanism is messier
than "filter out corrections, keep everything else," the sample shrank
by more than half, and parameter instability persists unchanged. Treat as
a genuine improvement worth carrying forward, not as proof the fragility
is resolved.

---

## Forensic drill-down: what exactly did the regime filter remove in windows 1 and 2?

Day-by-day regime sequence check first: Window 1's test period has only 1
regime transition (CORRECTION for ~10 weeks, then a clean switch to
BULLISH) -- this is a genuine, unambiguous market period (the 2022-2023
bear market bottoming in March 2023 amid the banking crisis, then a real
bull turn), NOT classifier hesitation. Window 2 has 6 transitions
(BULLISH -> brief NEUTRAL/CORRECTION dip -> BULLISH) -- this matches the
real Aug-Oct 2023 ~10% S&P pullback within an otherwise-intact bull run,
a real but short-lived corrective episode, not classifier noise per se.

**Trade-level reconstruction (params thresh=10%, hold=40d, as actually
chosen for both windows by the regime-filtered run):**

Window 1 -- 8 candidates, 6 removed (CORRECTION), 2 kept (BULLISH):
- Removed: DIS -14.8%, JPM **+4.2%**, BAC -5.6%, MCD -0.6%, CAT **+12.2%**, PFE -8.7%
- Kept: JPM -5.0%, MCD -7.1% (BOTH losers)

Window 2 -- 6 candidates, 2 removed (CORRECTION), 4 kept (BULLISH/NEUTRAL):
- Removed: PFE -5.1%, CAT **+29.8%**
- Kept: PFE -10.8%, CAT -0.1%, DIS +1.9%, GS +3.4%

**This directly answers the question, and not in the filter's favor.** In
both windows, the filter removed substantial WINNERS (JPM +4.2%, CAT +12.2%
in window 1; CAT +29.8% in window 2) right alongside losers, while KEEPING
substantial LOSERS that happened to fall on days labeled BULLISH (JPM -5.0%,
MCD -7.1% in window 1; PFE -10.8% in window 2). There is no clean
separation between "removed" and "lost money" -- the entry-day regime
label is a weak-to-nonexistent predictor of an individual trade's actual
40-day-forward outcome in these two windows.

**Root cause, most likely**: a PEAD trade's entry-day regime label is a
poor proxy for its outcome because the REGIME CAN CHANGE during the 20-60
day hold period, and the filter only looks at entry-day state. CAT's
window-2 trade entered 2023-10-31 (during the brief correction dip) but
exited 2023-12-28, by which point the market had fully resumed its bull
run -- hence +29.8%. The entry-day gate has no way to anticipate this;
it's throwing away information (or rather, guessing on incomplete
information) about a state that will evolve substantially over the
following 2 months.

Window 1 is different in character: it's a case of small-sample luck
after aggressive filtering (8 candidates -> 2), not a regime-driven
outcome specifically -- the 2 kept trades both happening to lose is
close to what you'd expect from randomly keeping 2 of 8 mixed-outcome
trades, not evidence the filter is doing something wrong per se, just
evidence that 2-trade samples are not informative.

## Honest verdict on the regime filter construction

**This is very likely over-engineering relative to what the underlying
signal and data can support, not a validated improvement.** The aggregate
numbers (Sharpe 2.41->4.70, MC_DD95 13.0%->6.2%) look better, but the
trade-level evidence shows the mechanism is not doing what it's supposed
to do in 2 of the 3 windows where it actually binds (window 0 is the
exception -- a genuine, sustained, unambiguous bear market where avoiding
everything was plausibly correct). In windows 1 and 2, the filter is
functionally closer to "remove a somewhat arbitrary subset of trades,
shrinking the sample" than "surgically avoid negative-beta trades" --
and shrinking an already-thin sample (120 -> 58 trades) while getting
credit for the resulting lower apparent variance is exactly the kind of
artifact the project's anti-overfitting rules exist to catch, even though
this particular manifestation (day-level regime gating removing winners
and keeping losers near-randomly) isn't directly caught by the IS/OOS
Sharpe deviation rule.

**Recommendation: do not carry this specific entry-day regime gate
forward to the larger ticker universe as currently built.** If regime-
awareness is worth pursuing further, a more principled version would need
to account for the regime trajectory across the WHOLE hold period (e.g.
exit early if regime flips to CORRECTION mid-hold, rather than gating
only at entry) or use a smoother, less binary signal (e.g. SPY vs its own
SMA200, which changes slowly and wouldn't have flipped 6 times in a
10-week window) -- but that is a new construction requiring its own
validation, not "more data on the same filter."

---

## DECISION: regime-filter path rejected -- do not retry without revisiting this section

**Status: REJECTED (2026-07-04).** The binary entry-day regime filter
(block new entries when SPY is in CORRECTION per `classify_regime_periods`)
is NOT being carried forward. Reason: the trade-level forensic check above
showed the mechanism is not causal -- in the 2 of 3 windows where it
actually bound (windows 1 and 2), it removed substantial winning trades
(JPM +4.2%, CAT +12.2%, CAT +29.8%) right alongside losers, and kept
substantial losing trades (JPM -5.0%, MCD -7.1%, PFE -10.8%) that happened
to fall on BULLISH-labeled days. The aggregate improvement it produced
(Sharpe 2.41->4.70, MC_DD95 13.0%->6.2%) is very likely driven mostly by
sample-size reduction (120->58 trades) rather than genuinely avoiding
negative-beta trades. Only window 0 (a sustained, unambiguous bear market)
showed the filter behaving as intended.

Separately, with only 7 walk-forward windows total, there is not enough
data to distinguish a genuine regime-aware improvement from a new
overfitting artifact, no matter how principled the construction looks on
paper -- explicitly the user's reasoning for not pursuing a 4th iteration
(e.g. a hold-period-aware dynamic exit, or a smoother SMA200-based gate)
right now.

**If this path is revisited later**, it should not restart from scratch --
start from this conclusion: any future regime-aware construction needs a
mechanism that accounts for the regime trajectory across the ENTIRE hold
period (not just entry-day state), and should not be evaluated as
"validated" based on aggregate Sharpe/drawdown improvement alone --
trade-level attribution (does it remove trades for a defensible reason?)
is required before trusting it, exactly as done here.

**Going forward: the baseline carried into the universe-expansion
drip-feed is the CAPPED-ONLY version (exposure cap 1.5x, max 3 new
positions/day, priority by surprise magnitude, NO regime filter) --
Sharpe=2.41, MC_DD95=13.0%, PF=1.44, WinRate=55.0%, 120 trades, cost-15bps
survival=+14.7%. This is the least-engineered, most interpretable result
produced so far and the one being tested for generalization to a larger
ticker universe.**

---

## Bug found and fixed: malformed API responses were being permanently cached as "no data"

While running today's drip-feed batch, BK's fetch returned an empty `{}`
JSON object with NONE of the expected "Information"/"Note"/"Error Message"
keys Alpha Vantage normally uses for rate-limit/error responses. Because
`earnings_data.py`'s cache-write only skipped on those specific keys, this
malformed response got written to `earnings_cache/BK_earnings.json` as a
genuine (but empty) cache entry -- meaning BK would have been silently and
PERMANENTLY treated as "confirmed no earnings data" on every future
drip-feed run, never retried. (Exact root cause of the malformed shape is
unclear -- possibly the same daily quota limit surfacing differently,
possibly unrelated -- but the caching bug is real regardless of cause.)

**Fixed**: `get_quarterly_surprises` now also refuses to cache when
`quarterlyEarnings` is missing or empty, regardless of which keys are
present in the response. Added 4 regression tests in
`tests/test_earnings_data.py` (successful response caches correctly,
rate-limit response doesn't cache, malformed/empty response without the
documented error keys doesn't cache -- the regression case -- and cached
files are read without a new API call). Deleted the bad `BK_earnings.json`
so it gets retried on the next drip-feed run. 44/44 tests pass.

## Drip-feed progress (capped-only baseline, no regime filter)

- Day 1 (2026-07-03): 0 fetched (today's quota already spent on setup/verification calls).
- Day 2 (2026-07-04): 6 fetched -- AME, AMGN, AON, AXON, BAX, BBWI. BK's malformed response
  (see bug above) incorrectly counted as "quota exhausted" and stopped the batch early; the
  real remaining quota for today is unknown, but the bug fix means no data is lost either way.
- Running total: 23 (original) + AXP (quota probe) + 6 new = 30 tickers with real cached data.
- 69 tickers remaining in the 75-ticker target batch (BK included, will retry).
- Pace has been slower than the assumed 25/day (6/day observed on day 2) -- possible the
  effective daily cap is tighter than documented, or day 2's early stop was itself partly an
  artifact of the now-fixed bug. Will reassess the days-remaining estimate after 1-2 more days
  of real data.

## Bug found and fixed: batch was stopping on individual bad symbols, not just quota exhaustion

Confirmed empirically: BK's malformed response was NOT a quota issue --
fetched a different ticker (CCL) immediately after and it succeeded with
122 quarters, proving today's quota was fine. `drip_feed_pead.py` was
treating ANY empty result as "quota exhausted, stop the whole batch,"
which meant a single permanently-bad symbol early in the alphabetical
target list (BK) would silently block progress on all 68 remaining
tickers, every single day, forever.

**Fixed properly this time**: `earnings_data.py` now raises a dedicated
`AlphaVantageRateLimitError` only when the response text specifically
mentions "rate limit"/"per day"/"per minute" (Alpha Vantage's actual
quota-exhaustion wording) -- any other error/empty response (invalid
symbol, malformed data, etc.) just returns an empty DataFrame and is
logged as a permanent no-data symbol, without raising. `drip_feed_pead.py`
now catches the specific exception to stop the day's batch, but skips
past a plain no-data symbol and keeps going. Added 2 more regression tests
(rate-limit response raises the specific exception; a non-rate-limit error
message does not raise, just returns empty) -- 45/45 tests pass total.

**Result with the fix**: day 3 re-run went from 0 useful fetches (stopped
dead at BK) to 14 useful fetches (CDW, CLX, CMI, CPRT, CTAS, CTSH, DGX,
DHI, DPZ, ELV, EMR, EOG, EQT, ETN) before correctly hitting the REAL daily
rate limit at FDS. BK and DFS confirmed as genuine no-data symbols (won't
be retried). Running total: 45 tickers with real cached data (23 original
+ AXP + CCL + 14 new). 52 remaining in the 75-ticker target batch. At
~14/day (today's real observed rate): ~4 more days.

**Autonomy note**: continuing per the user's standing rule to act without
per-step confirmation on technical/analysis decisions. This session's
tools cannot bridge a full-day wait (scheduled wakeup is capped at 1 hour,
and no persistent external scheduler is authorized) -- the drip-feed will
resume on the next invocation of this script, whether triggered by the
user's next check-in or a future session.

## Data quality check: 45 cached tickers

- Earnings data: 20/45 tickers flagged:
    AXON: 8/99 rows with NaN surprise_pct
    AXON: 1 extreme surprise_pct values (>|500.0%|): [533.3]
    AXON: 3 quarter gaps < 60d (possible duplicate/restated report): [54.0, 58.0, 57.0]
    AXON: 1 quarter gaps > 200d (possible missed quarter): [204.0]
    AXP: 1 extreme surprise_pct values (>|500.0%|): [583.3]
    BAC: 2 extreme surprise_pct values (>|500.0%|): [-900.0, -613.3]
    BAX: 1/121 rows with NaN surprise_pct
    BAX: 1 quarter gaps < 60d (possible duplicate/restated report): [44.0]
    BBWI: 7/109 rows with NaN surprise_pct
    BBWI: 1 extreme surprise_pct values (>|500.0%|): [1030.0]
    CAT: 2 extreme surprise_pct values (>|500.0%|): [1850.0, 1500.0]
    CCL: 1/122 rows with NaN surprise_pct
    CCL: 1 extreme surprise_pct values (>|500.0%|): [650.0]
    CDW: 1/52 rows with NaN surprise_pct
    COST: 1 duplicate report dates: [Timestamp('2007-05-31 00:00:00+0000', tz='UTC')]
    COST: 13/102 rows with NaN surprise_pct
    COST: 3 quarter gaps < 60d (possible duplicate/restated report): [26.0, 21.0, 0.0]
    COST: 3 quarter gaps > 200d (possible missed quarter): [211.0, 217.0, 203.0]
    CTSH: 4/112 rows with NaN surprise_pct
    DHI: 2 extreme surprise_pct values (>|500.0%|): [-546.9, 1010.0]
    DPZ: 1/88 rows with NaN surprise_pct
    DPZ: 4 quarter gaps < 60d (possible duplicate/restated report): [49.0, 58.0, 58.0]
    EOG: 3/121 rows with NaN surprise_pct
    EQT: 2 extreme surprise_pct values (>|500.0%|): [7500.0, 800.0]
    GS: 1 extreme surprise_pct values (>|500.0%|): [-740.0]
    JPM: 1 extreme surprise_pct values (>|500.0%|): [-2900.0]
    JPM: 1 quarter gaps < 60d (possible duplicate/restated report): [45.0]
    KO: 1 quarter gaps < 60d (possible duplicate/restated report): [56.0]
    LIN: 1 quarter gaps < 60d (possible duplicate/restated report): [53.0]
    WMT: 5 quarter gaps < 60d (possible duplicate/restated report): [48.0, 50.0, 48.0]
    XOM: 1 extreme surprise_pct values (>|500.0%|): [960.0]
- Price bars: no issues found across all 45 tickers (no non-positive OHLC, no impossible high/low, no large gaps, sufficient history).

## Data quality check: 45 cached tickers

- Earnings data: 23/45 tickers flagged:
    AXON: 42/99 rows with NaN surprise_pct
    AXON: 3 quarter gaps < 60d (possible duplicate/restated report): [54.0, 58.0, 57.0]
    AXON: 1 quarter gaps > 200d (possible missed quarter): [204.0]
    AXP: 1 extreme surprise_pct values (>|500.0%|): [583.3]
    BAC: 1/121 rows with NaN surprise_pct
    BAC: 2 extreme surprise_pct values (>|500.0%|): [-900.0, -613.3]
    BAX: 1/121 rows with NaN surprise_pct
    BAX: 1 quarter gaps < 60d (possible duplicate/restated report): [44.0]
    BBWI: 15/109 rows with NaN surprise_pct
    BBWI: 1 extreme surprise_pct values (>|500.0%|): [1030.0]
    CAT: 2/121 rows with NaN surprise_pct
    CCL: 6/122 rows with NaN surprise_pct
    CDW: 1/52 rows with NaN surprise_pct
    CMI: 5/121 rows with NaN surprise_pct
    COST: 12/101 rows with NaN surprise_pct
    COST: 2 quarter gaps < 60d (possible duplicate/restated report): [26.0, 21.0]
    COST: 3 quarter gaps > 200d (possible missed quarter): [211.0, 217.0, 203.0]
    CPRT: 61/121 rows with NaN surprise_pct
    CTSH: 26/112 rows with NaN surprise_pct
    DGX: 6/118 rows with NaN surprise_pct
    DHI: 4/121 rows with NaN surprise_pct
    DHI: 2 extreme surprise_pct values (>|500.0%|): [-546.9, 1010.0]
    DPZ: 1/88 rows with NaN surprise_pct
    DPZ: 4 quarter gaps < 60d (possible duplicate/restated report): [49.0, 58.0, 58.0]
    EOG: 13/121 rows with NaN surprise_pct
    EQT: 13/121 rows with NaN surprise_pct
    GS: 1 extreme surprise_pct values (>|500.0%|): [-740.0]
    JPM: 1/121 rows with NaN surprise_pct
    JPM: 1 quarter gaps < 60d (possible duplicate/restated report): [45.0]
    KO: 1 quarter gaps < 60d (possible duplicate/restated report): [56.0]
    LIN: 1 quarter gaps < 60d (possible duplicate/restated report): [53.0]
    WMT: 5 quarter gaps < 60d (possible duplicate/restated report): [48.0, 50.0, 48.0]
    XOM: 1/121 rows with NaN surprise_pct
    XOM: 1 extreme surprise_pct values (>|500.0%|): [960.0]
- Price bars: no issues found across all 45 tickers (no non-positive OHLC, no impossible high/low, no large gaps, sufficient history).

## Data quality fixes applied (bot/earnings_data.py, data hygiene not strategy logic)

Two real issues found and fixed in `_parse`, both applying retroactively
to all 45 already-cached tickers with no new API calls needed (the fix is
in how cached raw JSON is interpreted, not in what's stored):

1. **Near-zero-estimate percentage blowup**: surprise_pct = (actual-est)/est
   is mathematically unstable when the consensus estimate is near zero --
   found real cases up to +-7500% (EQT 2018, estimated_eps=$0.01). Rows
   with `|estimated_eps| < $0.05` now have surprise_pct nulled (row kept,
   just excluded from any surprise_pct>=threshold signal), exactly like
   the already-existing handling for old pre-analyst-coverage quarters.
   Confirmed this did NOT affect the already-reported capped-only baseline
   result (Sharpe=2.41, MC_DD95=13.0%, 120 trades) -- all affected dates
   found were 2009-2010, well before the walk-forward's 2020-07-27 start.
2. **Duplicate report dates**: COST had two quarters (2007-02-28 and
   2007-05-31 fiscal periods) both stamped with reportedDate=2007-05-31 --
   deduplicated, preferring the row with a usable surprise_pct.

Both fixes have dedicated regression tests in `tests/test_earnings_data.py`
(7 tests total for this module now). 47/47 tests pass project-wide.

**Remaining extreme surprise_pct values reviewed, not further filtered**:
AXP, BAC, BBWI, DHI, GS, XOM still show |surprise_pct|>500% on quarters
with estimated_eps in the $0.05-0.64 range. Checked each: these trace to
real crisis-period earnings collapses/recoveries (BAC/GS 2009 financial
crisis, GS 2011 eurozone crisis, DHI 2008 housing collapse / 2012
recovery, XOM/AXP 2020 COVID), not division artifacts -- the underlying
EPS numbers are genuinely extreme, not just the percentage. Chasing a
higher exclusion threshold to eliminate these too would drift from
"fix a numerical artifact" toward "tune away inconvenient real data
points," which is out of scope for a data hygiene pass. Left as-is.

**Quarter-gap anomalies reviewed, no fix needed**: tight gaps (21-58 days
between reports, several tickers) are normal reporting-calendar variation.
Wide gaps (200+ days, COST specifically in 2016/2017/2020) trace to
genuinely MISSING quarters in Alpha Vantage's historical coverage for
that ticker, not a parsing bug -- a real data completeness limitation of
the free tier's historical depth, not something fixable without a second
data source. Minor (a few missing quarters out of 100+ for one ticker),
noted for transparency, does not warrant action.

**Price bars: zero issues found** across all 45 tickers (no non-positive
OHLC, no impossible high/low relationships, no gaps >10 calendar days,
all with sufficient history).

## Viability check: capped-only, all cached tickers (45 tickers)

- Tickers: ['ABBV', 'AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAC', 'BAX', 'BBWI', 'CAT', 'CCL', 'CDW', 'CLX', 'CMI', 'COST', 'CPRT', 'CTAS', 'CTSH', 'CVX', 'DGX', 'DHI', 'DIS', 'DPZ', 'DUK', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'GS', 'HD', 'HON', 'JNJ', 'JPM', 'KO', 'LIN', 'MCD', 'PFE', 'PG', 'UNH', 'UPS', 'V', 'VZ', 'WMT', 'XOM']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 24mo train / 6mo test, 5 windows over 2021-08-03 to 2026-06-23.
- Window 0: params thresh=3.0% hold=20d, IS Sharpe=0.200, OOS Sharpe=4.964, OOS trades=34, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=7.0% hold=60d, IS Sharpe=3.000, OOS Sharpe=11.204, OOS trades=29, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=7.086, OOS Sharpe=4.544, OOS trades=26, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=7.0% hold=60d, IS Sharpe=5.614, OOS Sharpe=9.198, OOS trades=26, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=7.0% hold=60d, IS Sharpe=8.857, OOS Sharpe=11.666, OOS trades=27, FLAG (>30% IS/OOS deviation)

### Combined out-of-sample result (capped-only, all cached tickers)

- Sharpe=8.080  Sortino=10.938  MaxDD=6.4%  MC_DD95=11.2%  PF=2.142  WinRate=62.0%  Trades=142  (5/5 windows flagged)
- Cost sensitivity: 0.05%->50.9%, 0.10%->49.4%, 0.15%->47.9%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 5/5 windows flagged for IS/OOS deviation >30%**

**VERDICT (capped-only, all cached tickers): PASSES all 3 viability criteria**

## Preparatory work while the drip-feed quota recharges (no strategy changes)

**1. Generalized viability-check script ready**: `run_pead_viability_check.py`
dynamically discovers all cached tickers via `earnings_cache/*.json` glob
(no hardcoded list) and runs the exact capped-only methodology (exposure
cap 1.5x, max 3/day, priority by surprise magnitude, NO regime filter) --
ready to run immediately once the full 98-ticker batch completes, no
changes needed. Test-run on the current 45-ticker interim set confirms it
works end-to-end (Sharpe=8.08, MC_DD95=11.2%, 142 trades, 5 windows -- but
this number should NOT be read as a signal: only 45/98 tickers collected,
window count differs from the eventual full run since it's derived from
min/max across whichever tickers happen to be cached so far, and this was
explicitly a functional test, not a checkpoint result).

**2. Data quality check on all 45 cached tickers**: found and fixed two
real issues (near-zero-estimate percentage blowup, duplicate report dates
-- see above), confirmed neither affected the already-reported baseline
result. Price bars: zero issues. Remaining extreme surprise values and
quarter-gap variations reviewed and traced to genuine crisis-period events
or normal reporting variation, not bugs -- documented, not filtered further.

**3. Test coverage strengthened**: added 5 tests for previously-uncovered,
risk-relevant logic -- slippage direction correctness (a silent bug here
would make the backtest look better than reality without symptoms),
NaN surprise_pct exclusion at the event-detection layer (not just the
parsing layer), events with insufficient future data for the hold period,
and `daily_returns_from_equity` basic sanity + empty-series handling.
52/52 tests pass project-wide (up from 45).

No new strategy variant or filter was built, per standing instruction --
all of the above is data hygiene, tooling, and test coverage, not new
trading logic.

## Viability check: capped-only, all cached tickers (97 tickers)

- Tickers: ['ABBV', 'AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAC', 'BAX', 'BBWI', 'CAT', 'CCL', 'CDW', 'CLX', 'CMI', 'COST', 'CPRT', 'CTAS', 'CTSH', 'CVX', 'DGX', 'DHI', 'DIS', 'DPZ', 'DUK', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'GS', 'HBAN', 'HCA', 'HD', 'HON', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'JNJ', 'JPM', 'KLAC', 'KO', 'KR', 'L', 'LH', 'LIN', 'LRCX', 'MCD', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PFE', 'PG', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'UNH', 'UPS', 'USB', 'V', 'VRSK', 'VZ', 'WDC', 'WHR', 'WMT', 'XOM', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 24mo train / 6mo test, 5 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=1.527, OOS Sharpe=7.272, OOS trades=34, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=5.0% hold=60d, IS Sharpe=5.806, OOS Sharpe=8.562, OOS trades=35, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=8.685, OOS Sharpe=2.851, OOS trades=35, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=10.0% hold=60d, IS Sharpe=6.040, OOS Sharpe=9.598, OOS trades=30, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=7.0% hold=60d, IS Sharpe=9.498, OOS Sharpe=13.743, OOS trades=35, FLAG (>30% IS/OOS deviation)

### Combined out-of-sample result (capped-only, all cached tickers)

- Sharpe=8.095  Sortino=10.721  MaxDD=10.9%  MC_DD95=15.5%  PF=1.988  WinRate=61.5%  Trades=169  (5/5 windows flagged)
- Cost sensitivity: 0.05%->72.1%, 0.10%->70.4%, 0.15%->68.6%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 5/5 windows flagged for IS/OOS deviation >30%**

**VERDICT (capped-only, all cached tickers): PASSES all 3 viability criteria**

## Viability check: independent check: 74 NEW tickers only, disjoint from original 23 (74 tickers)

- Tickers: ['AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAX', 'BBWI', 'CCL', 'CDW', 'CLX', 'CMI', 'CPRT', 'CTAS', 'CTSH', 'DGX', 'DHI', 'DPZ', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'HBAN', 'HCA', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'KLAC', 'KR', 'L', 'LH', 'LRCX', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'USB', 'VRSK', 'WDC', 'WHR', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 24mo train / 6mo test, 5 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=1.772, OOS Sharpe=9.199, OOS trades=33, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=3.0% hold=60d, IS Sharpe=5.868, OOS Sharpe=2.279, OOS trades=36, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=9.985, OOS Sharpe=-0.515, OOS trades=29, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=10.0% hold=60d, IS Sharpe=6.837, OOS Sharpe=12.815, OOS trades=26, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=10.0% hold=60d, IS Sharpe=10.465, OOS Sharpe=11.807, OOS trades=23, OK

### Combined out-of-sample result (independent check: 74 NEW tickers only, disjoint from original 23)

- Sharpe=7.462  Sortino=10.553  MaxDD=10.2%  MC_DD95=16.9%  PF=1.863  WinRate=60.5%  Trades=147  (4/5 windows flagged)
- Cost sensitivity: 0.05%->59.8%, 0.10%->58.2%, 0.15%->56.7%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 4/5 windows flagged for IS/OOS deviation >30%**

**VERDICT (independent check: 74 NEW tickers only, disjoint from original 23): PASSES all 3 viability criteria**

---

## CRITICAL BUG FOUND: Sharpe was inflated by 10-50x due to linear-interpolation mark-to-market

While reviewing the full 97-ticker viability result (Sharpe=8.095 -- an
implausibly high number for any real strategy), traced the cause to
`equity_curve_from_trades`: unrealized P&L during a trade's holding period
was marked via a LINEAR RAMP from entry to exit price, not the real daily
closing price -- even though real daily price data was already being
fetched and available. A straight-line ramp to a known endpoint has almost
zero day-to-day variance, which severely understates the true volatility
feeding into Sharpe = mean/std * sqrt(252), and hides intra-trade
drawdown entirely (a ramp to a positive endpoint never dips below its
start).

**Verified empirically** with a synthetic single trade (same entry/exit
price, same final P&L, real ~20%-annualized-vol daily price path):
linear interpolation gives Sharpe=25.05, MaxDD=0.0%; marking to the real
daily close gives Sharpe=0.55, MaxDD=2.1%. Same final equity either way
(101,727.24) -- the bug only distorts the PATH, not the endpoint, which
is exactly why cost-sensitivity (compares final equity across slippage
scenarios) and Monte Carlo drawdown (resamples trade-level returns
directly, never touches this equity curve) are NOT affected -- only
Sharpe and the (non-MC) historical MaxDD are.

**This means every Sharpe number reported anywhere in this PEAD
investigation (2.93 preliminary, 2.41 capped-only, 4.70 regime-filtered
[already rejected], 8.10 full-97, 7.46 new-only-74) is inflated by an
unknown but likely large factor and should not be trusted as reported.**
MC_DD95 and cost-survival numbers from those same runs remain valid.

**Fixed**: `equity_curve_from_trades` now accepts an optional
`bars_by_ticker` parameter; when provided, unrealized P&L is marked to
the real daily close price (correct). Without it, falls back to the old
linear interpolation (kept only for tests that don't need valid risk
metrics) -- callers computing anything used for a real decision MUST pass
bars_by_ticker. `run_pead_viability_check.py` updated at all 4 call sites.
Added a dedicated regression test (`test_real_price_marking_gives_realistic_sharpe...`)
asserting the fallback overstates Sharpe by >10x and hides drawdown
entirely versus the real-price version, and that the real-price Sharpe
lands in a plausible range (0.1-2.0) for a single noisy trade. 53/53
tests pass.

Re-running the full-97 and new-only-74 viability checks now with the
corrected engine before reporting any verdict.

## Viability check: capped-only, all cached tickers (97 tickers)

- Tickers: ['ABBV', 'AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAC', 'BAX', 'BBWI', 'CAT', 'CCL', 'CDW', 'CLX', 'CMI', 'COST', 'CPRT', 'CTAS', 'CTSH', 'CVX', 'DGX', 'DHI', 'DIS', 'DPZ', 'DUK', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'GS', 'HBAN', 'HCA', 'HD', 'HON', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'JNJ', 'JPM', 'KLAC', 'KO', 'KR', 'L', 'LH', 'LIN', 'LRCX', 'MCD', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PFE', 'PG', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'UNH', 'UPS', 'USB', 'V', 'VRSK', 'VZ', 'WDC', 'WHR', 'WMT', 'XOM', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 24mo train / 6mo test, 5 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=0.240, OOS Sharpe=1.093, OOS trades=34, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=5.0% hold=60d, IS Sharpe=0.628, OOS Sharpe=0.511, OOS trades=35, OK
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.119, OOS Sharpe=0.416, OOS trades=35, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=10.0% hold=20d, IS Sharpe=0.960, OOS Sharpe=1.383, OOS trades=39, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=10.0% hold=20d, IS Sharpe=1.556, OOS Sharpe=1.291, OOS trades=32, OK

### Combined out-of-sample result (capped-only, all cached tickers)

- Sharpe=0.884  Sortino=1.002  MaxDD=18.5%  MC_DD95=14.2%  PF=1.891  WinRate=62.3%  Trades=175  (3/5 windows flagged)
- Cost sensitivity: 0.05%->56.4%, 0.10%->54.6%, 0.15%->52.8%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 3/5 windows flagged for IS/OOS deviation >30%**

**VERDICT (capped-only, all cached tickers): PASSES all 3 viability criteria**

## Viability check: independent check: 74 NEW tickers only, disjoint from original 23 (corrected engine) (74 tickers)

- Tickers: ['AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAX', 'BBWI', 'CCL', 'CDW', 'CLX', 'CMI', 'CPRT', 'CTAS', 'CTSH', 'DGX', 'DHI', 'DPZ', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'HBAN', 'HCA', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'KLAC', 'KR', 'L', 'LH', 'LRCX', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'USB', 'VRSK', 'WDC', 'WHR', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 24mo train / 6mo test, 5 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=0.284, OOS Sharpe=1.278, OOS trades=33, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=3.0% hold=60d, IS Sharpe=0.673, OOS Sharpe=0.164, OOS trades=36, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.176, OOS Sharpe=-0.030, OOS trades=29, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=10.0% hold=60d, IS Sharpe=1.080, OOS Sharpe=1.096, OOS trades=26, OK
- Window 4: params thresh=10.0% hold=20d, IS Sharpe=1.601, OOS Sharpe=0.957, OOS trades=23, FLAG (>30% IS/OOS deviation)

### Combined out-of-sample result (independent check: 74 NEW tickers only, disjoint from original 23 (corrected engine))

- Sharpe=0.718  Sortino=0.808  MaxDD=28.7%  MC_DD95=16.3%  PF=1.782  WinRate=60.5%  Trades=147  (4/5 windows flagged)
- Cost sensitivity: 0.05%->49.5%, 0.10%->48.0%, 0.15%->46.5%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 4/5 windows flagged for IS/OOS deviation >30%**

**VERDICT (independent check: 74 NEW tickers only, disjoint from original 23 (corrected engine)): PASSES all 3 viability criteria**

## Re-test with portfolio-level exposure cap (max_gross_exposure=1.5x, max_new_positions_per_day=3, priority by surprise magnitude)

- Identical setup to the original run: same 23 tickers, same grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], same 7 walk-forward windows (24mo train / 6mo test). Only change: trades now go through `build_portfolio_event_trades` instead of independent per-ticker sizing.
- Window 0: params thresh=10.0% hold=20d, IS Sharpe=1.210, OOS Sharpe=0.580, OOS trades=9, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=7.0% hold=20d, IS Sharpe=0.958, OOS Sharpe=-1.816, OOS trades=16, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=3.0% hold=20d, IS Sharpe=0.087, OOS Sharpe=1.214, OOS trades=21, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=3.0% hold=20d, IS Sharpe=0.192, OOS Sharpe=1.145, OOS trades=24, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=3.0% hold=60d, IS Sharpe=0.663, OOS Sharpe=0.144, OOS trades=20, FLAG (>30% IS/OOS deviation)
- Window 5: params thresh=7.0% hold=40d, IS Sharpe=0.703, OOS Sharpe=1.449, OOS trades=16, FLAG (>30% IS/OOS deviation)
- Window 6: params thresh=7.0% hold=40d, IS Sharpe=1.365, OOS Sharpe=0.701, OOS trades=14, FLAG (>30% IS/OOS deviation)

### Combined out-of-sample result (exposure-capped)

- Sharpe=0.494  Sortino=0.495  MaxDD=10.9%  **MC_DD95=10.2%**  PF=1.543  WinRate=56.7%  Trades=120  (7/7 windows flagged)

**Comparison: uncapped MC_DD95 was 80.1% (131 trades) -> capped MC_DD95 is 10.2% (120 trades).**

**CONCLUSION: the drawdown problem was substantially structural/sizing-driven -- capping exposure meaningfully reduced tail risk. This points at engineering (position sizing discipline), not the underlying PEAD signal, as the primary issue.**

---

## FINAL SYNTHESIS: complete conclusion after 98-ticker collection + critical bug fix

**Corrected comparison table (all figures now use real-price mark-to-market, not the flawed linear interpolation):**

| Version | Sharpe | MC_DD95 | Actual MaxDD | Trades | Windows flagged | Formal verdict |
|---|---|---|---|---|---|---|
| 23-ticker original baseline | 0.494 | 10.2% | 10.9% | 120 | 7/7 | **FAILS** (Sharpe<0.5, by 0.006) |
| 97-ticker full universe | 0.884 | 14.2% | 18.5% | 175 | 3/5 | PASSES |
| 74-ticker new-only (independent) | 0.718 | 16.3% | 28.7% | 147 | 4/5 | PASSES |

**What changed and why it matters**: the number this entire multi-day
investigation was anchored on (Sharpe=2.41, later corrected to 2.93
uncapped) was computed with a mark-to-market bug that inflated Sharpe by
roughly 5x on the original data. The TRUE original-23-ticker Sharpe is
0.494 -- a coin-flip away from failing the pre-registered viability bar
entirely, with EVERY SINGLE window (7/7) showing unstable IS/OOS Sharpe.
Had this bug not been found, the project would have been operating on a
badly wrong picture of how strong this signal actually is.

**Independent validation performed**: fresh, disjoint 74-ticker universe
(never part of the original discovery, objectively identified as
"cached minus the original 23") independently clears all 3 viability
criteria (Sharpe=0.718, MC_DD95=16.3%, cost-15bps survival=+46.5%). This
is real evidence against pure overfitting to the original 23 -- a
classic overfit would be expected to show reversal or much weaker
performance on a disjoint set, similar to what happened to round 2's
signal on a fresh time window. No fresh TIME window validation was
possible here (the data frontier -- back to 2020-07-27, the earliest
available daily price history -- is already fully used inside the
walk-forward; there is no unseen earlier period left to test against).

**Persistent, unresolved concern**: window-level instability never went
away. Even in the best-looking result (97-ticker full universe), 3 of 5
windows still show >30% IS/OOS Sharpe deviation -- the walk-forward's own
parameter selection is still noisy and not clearly converging on a
stable, generalizable (threshold, hold_days) choice. Actual historical
max drawdown also GREW as the universe grew (10.9% -> 18.5%/28.7%), even
though the Monte Carlo 95th-percentile estimate stayed under the 20% bar
throughout.

**HONEST VERDICT: le signal PEAD passe la barre formelle sur un univers
plus large et une validation indépendante par tickers frais, mais de
justesse et avec une instabilité de paramètres qui persiste dans la
majorité des fenêtres testées. Ce n'est pas un résultat solide et
convaincant -- c'est un passage marginal, obtenu seulement après avoir
corrigé un bug sérieux qui aurait autrement fait paraître le signal
beaucoup plus fort qu'il ne l'est réellement.**

## Sensitivity analysis: walk-forward train-window width

- Sweeping train_months in [12, 18, 24, 30, 36] (test_months=6 fixed throughout -- unchanged since the very first PEAD run). Not chosen post-hoc to improve any number -- a symmetric bracket around the original 24mo choice. Corrected (real-price mark-to-market) engine throughout.
- Also confirmed: BK and DFS (the 2 no-data symbols) never produced a cache file and are therefore ALREADY excluded from every run by construction -- there is no data to 'include', so this dimension is not separately testable, and doesn't affect any result below.

## Viability check: sensitivity: full universe, train=12mo/test=6mo (97 tickers)

- Tickers: ['ABBV', 'AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAC', 'BAX', 'BBWI', 'CAT', 'CCL', 'CDW', 'CLX', 'CMI', 'COST', 'CPRT', 'CTAS', 'CTSH', 'CVX', 'DGX', 'DHI', 'DIS', 'DPZ', 'DUK', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'GS', 'HBAN', 'HCA', 'HD', 'HON', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'JNJ', 'JPM', 'KLAC', 'KO', 'KR', 'L', 'LH', 'LIN', 'LRCX', 'MCD', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PFE', 'PG', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'UNH', 'UPS', 'USB', 'V', 'VRSK', 'VZ', 'WDC', 'WHR', 'WMT', 'XOM', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 12mo train / 6mo test, 7 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=3.0% hold=20d, IS Sharpe=0.221, OOS Sharpe=0.088, OOS trades=55, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=10.0% hold=60d, IS Sharpe=0.323, OOS Sharpe=-0.537, OOS trades=31, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=0.402, OOS Sharpe=1.726, OOS trades=33, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=5.0% hold=40d, IS Sharpe=1.088, OOS Sharpe=0.752, OOS trades=45, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=10.0% hold=60d, IS Sharpe=1.870, OOS Sharpe=0.484, OOS trades=29, FLAG (>30% IS/OOS deviation)
- Window 5: params thresh=10.0% hold=20d, IS Sharpe=1.477, OOS Sharpe=1.383, OOS trades=39, OK
- Window 6: params thresh=10.0% hold=20d, IS Sharpe=1.474, OOS Sharpe=1.291, OOS trades=32, OK

### Combined out-of-sample result (sensitivity: full universe, train=12mo/test=6mo)

- Sharpe=0.641  Sortino=0.688  MaxDD=32.1%  MC_DD95=21.8%  PF=1.568  WinRate=58.7%  Trades=264  (5/7 windows flagged)
- Cost sensitivity: 0.05%->62.5%, 0.10%->59.8%, 0.15%->57.1%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: False | edge survives 0.15% slippage: True | 5/7 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: full universe, train=12mo/test=6mo): DOES NOT pass all 3 viability criteria**

## Viability check: sensitivity: new-only 74, train=12mo/test=6mo (74 tickers)

- Tickers: ['AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAX', 'BBWI', 'CCL', 'CDW', 'CLX', 'CMI', 'CPRT', 'CTAS', 'CTSH', 'DGX', 'DHI', 'DPZ', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'HBAN', 'HCA', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'KLAC', 'KR', 'L', 'LH', 'LRCX', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'USB', 'VRSK', 'WDC', 'WHR', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 12mo train / 6mo test, 7 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=3.0% hold=20d, IS Sharpe=0.200, OOS Sharpe=-0.151, OOS trades=52, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=10.0% hold=60d, IS Sharpe=0.323, OOS Sharpe=-0.363, OOS trades=31, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=0.808, OOS Sharpe=1.744, OOS trades=33, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=3.0% hold=40d, IS Sharpe=1.202, OOS Sharpe=0.230, OOS trades=45, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=10.0% hold=60d, IS Sharpe=1.737, OOS Sharpe=0.679, OOS trades=25, FLAG (>30% IS/OOS deviation)
- Window 5: params thresh=10.0% hold=20d, IS Sharpe=1.448, OOS Sharpe=1.648, OOS trades=35, OK
- Window 6: params thresh=10.0% hold=20d, IS Sharpe=1.535, OOS Sharpe=0.957, OOS trades=23, FLAG (>30% IS/OOS deviation)

### Combined out-of-sample result (sensitivity: new-only 74, train=12mo/test=6mo)

- Sharpe=0.592  Sortino=0.630  MaxDD=29.3%  MC_DD95=23.9%  PF=1.500  WinRate=56.6%  Trades=244  (6/7 windows flagged)
- Cost sensitivity: 0.05%->57.1%, 0.10%->54.6%, 0.15%->52.1%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: False | edge survives 0.15% slippage: True | 6/7 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: new-only 74, train=12mo/test=6mo): DOES NOT pass all 3 viability criteria**

## Viability check: sensitivity: full universe, train=18mo/test=6mo (97 tickers)

- Tickers: ['ABBV', 'AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAC', 'BAX', 'BBWI', 'CAT', 'CCL', 'CDW', 'CLX', 'CMI', 'COST', 'CPRT', 'CTAS', 'CTSH', 'CVX', 'DGX', 'DHI', 'DIS', 'DPZ', 'DUK', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'GS', 'HBAN', 'HCA', 'HD', 'HON', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'JNJ', 'JPM', 'KLAC', 'KO', 'KR', 'L', 'LH', 'LIN', 'LRCX', 'MCD', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PFE', 'PG', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'UNH', 'UPS', 'USB', 'V', 'VRSK', 'VZ', 'WDC', 'WHR', 'WMT', 'XOM', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 18mo train / 6mo test, 6 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=3.0% hold=20d, IS Sharpe=0.210, OOS Sharpe=-0.511, OOS trades=60, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=5.0% hold=60d, IS Sharpe=0.370, OOS Sharpe=1.093, OOS trades=34, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.023, OOS Sharpe=0.358, OOS trades=35, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=5.0% hold=60d, IS Sharpe=1.197, OOS Sharpe=0.085, OOS trades=36, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=10.0% hold=20d, IS Sharpe=1.584, OOS Sharpe=1.383, OOS trades=39, OK
- Window 5: params thresh=10.0% hold=20d, IS Sharpe=1.473, OOS Sharpe=1.291, OOS trades=32, OK

### Combined out-of-sample result (sensitivity: full universe, train=18mo/test=6mo)

- Sharpe=0.583  Sortino=0.662  MaxDD=23.2%  MC_DD95=19.6%  PF=1.461  WinRate=55.5%  Trades=236  (4/6 windows flagged)
- Cost sensitivity: 0.05%->43.1%, 0.10%->40.7%, 0.15%->38.3%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 4/6 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: full universe, train=18mo/test=6mo): PASSES all 3 viability criteria**

## Viability check: sensitivity: new-only 74, train=18mo/test=6mo (74 tickers)

- Tickers: ['AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAX', 'BBWI', 'CCL', 'CDW', 'CLX', 'CMI', 'CPRT', 'CTAS', 'CTSH', 'DGX', 'DHI', 'DPZ', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'HBAN', 'HCA', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'KLAC', 'KR', 'L', 'LH', 'LRCX', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'USB', 'VRSK', 'WDC', 'WHR', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 18mo train / 6mo test, 6 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=10.0% hold=60d, IS Sharpe=0.206, OOS Sharpe=-0.363, OOS trades=31, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=7.0% hold=60d, IS Sharpe=0.344, OOS Sharpe=1.744, OOS trades=33, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.176, OOS Sharpe=0.226, OOS trades=34, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=7.0% hold=60d, IS Sharpe=1.166, OOS Sharpe=-0.030, OOS trades=29, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=10.0% hold=20d, IS Sharpe=1.573, OOS Sharpe=1.648, OOS trades=35, OK
- Window 5: params thresh=10.0% hold=20d, IS Sharpe=1.526, OOS Sharpe=0.957, OOS trades=23, FLAG (>30% IS/OOS deviation)

### Combined out-of-sample result (sensitivity: new-only 74, train=18mo/test=6mo)

- Sharpe=0.672  Sortino=0.744  MaxDD=28.2%  MC_DD95=18.6%  PF=1.600  WinRate=58.9%  Trades=185  (5/6 windows flagged)
- Cost sensitivity: 0.05%->50.8%, 0.10%->48.9%, 0.15%->47.0%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 5/6 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: new-only 74, train=18mo/test=6mo): PASSES all 3 viability criteria**

## Viability check: sensitivity: full universe, train=24mo/test=6mo (97 tickers)

- Tickers: ['ABBV', 'AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAC', 'BAX', 'BBWI', 'CAT', 'CCL', 'CDW', 'CLX', 'CMI', 'COST', 'CPRT', 'CTAS', 'CTSH', 'CVX', 'DGX', 'DHI', 'DIS', 'DPZ', 'DUK', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'GS', 'HBAN', 'HCA', 'HD', 'HON', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'JNJ', 'JPM', 'KLAC', 'KO', 'KR', 'L', 'LH', 'LIN', 'LRCX', 'MCD', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PFE', 'PG', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'UNH', 'UPS', 'USB', 'V', 'VRSK', 'VZ', 'WDC', 'WHR', 'WMT', 'XOM', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 24mo train / 6mo test, 5 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=0.240, OOS Sharpe=1.093, OOS trades=34, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=5.0% hold=60d, IS Sharpe=0.628, OOS Sharpe=0.511, OOS trades=35, OK
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.119, OOS Sharpe=0.416, OOS trades=35, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=10.0% hold=20d, IS Sharpe=0.960, OOS Sharpe=1.383, OOS trades=39, FLAG (>30% IS/OOS deviation)
- Window 4: params thresh=10.0% hold=20d, IS Sharpe=1.556, OOS Sharpe=1.291, OOS trades=32, OK

### Combined out-of-sample result (sensitivity: full universe, train=24mo/test=6mo)

- Sharpe=0.884  Sortino=1.002  MaxDD=18.5%  MC_DD95=14.2%  PF=1.891  WinRate=62.3%  Trades=175  (3/5 windows flagged)
- Cost sensitivity: 0.05%->56.4%, 0.10%->54.6%, 0.15%->52.8%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 3/5 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: full universe, train=24mo/test=6mo): PASSES all 3 viability criteria**

## Viability check: sensitivity: new-only 74, train=24mo/test=6mo (74 tickers)

- Tickers: ['AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAX', 'BBWI', 'CCL', 'CDW', 'CLX', 'CMI', 'CPRT', 'CTAS', 'CTSH', 'DGX', 'DHI', 'DPZ', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'HBAN', 'HCA', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'KLAC', 'KR', 'L', 'LH', 'LRCX', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'USB', 'VRSK', 'WDC', 'WHR', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 24mo train / 6mo test, 5 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=0.284, OOS Sharpe=1.278, OOS trades=33, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=3.0% hold=60d, IS Sharpe=0.673, OOS Sharpe=0.164, OOS trades=36, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.176, OOS Sharpe=-0.030, OOS trades=29, FLAG (>30% IS/OOS deviation)
- Window 3: params thresh=10.0% hold=60d, IS Sharpe=1.080, OOS Sharpe=1.096, OOS trades=26, OK
- Window 4: params thresh=10.0% hold=20d, IS Sharpe=1.601, OOS Sharpe=0.957, OOS trades=23, FLAG (>30% IS/OOS deviation)

### Combined out-of-sample result (sensitivity: new-only 74, train=24mo/test=6mo)

- Sharpe=0.718  Sortino=0.808  MaxDD=28.7%  MC_DD95=16.3%  PF=1.782  WinRate=60.5%  Trades=147  (4/5 windows flagged)
- Cost sensitivity: 0.05%->49.5%, 0.10%->48.0%, 0.15%->46.5%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 4/5 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: new-only 74, train=24mo/test=6mo): PASSES all 3 viability criteria**

## Viability check: sensitivity: full universe, train=30mo/test=6mo (97 tickers)

- Tickers: ['ABBV', 'AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAC', 'BAX', 'BBWI', 'CAT', 'CCL', 'CDW', 'CLX', 'CMI', 'COST', 'CPRT', 'CTAS', 'CTSH', 'CVX', 'DGX', 'DHI', 'DIS', 'DPZ', 'DUK', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'GS', 'HBAN', 'HCA', 'HD', 'HON', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'JNJ', 'JPM', 'KLAC', 'KO', 'KR', 'L', 'LH', 'LIN', 'LRCX', 'MCD', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PFE', 'PG', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'UNH', 'UPS', 'USB', 'V', 'VRSK', 'VZ', 'WDC', 'WHR', 'WMT', 'XOM', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 30mo train / 6mo test, 4 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=0.521, OOS Sharpe=0.511, OOS trades=35, OK
- Window 1: params thresh=5.0% hold=60d, IS Sharpe=0.732, OOS Sharpe=0.085, OOS trades=36, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=10.0% hold=60d, IS Sharpe=0.884, OOS Sharpe=0.758, OOS trades=30, OK
- Window 3: params thresh=7.0% hold=60d, IS Sharpe=1.073, OOS Sharpe=1.215, OOS trades=35, OK

### Combined out-of-sample result (sensitivity: full universe, train=30mo/test=6mo)

- Sharpe=0.653  Sortino=0.688  MaxDD=37.9%  MC_DD95=16.6%  PF=1.775  WinRate=58.8%  Trades=136  (1/4 windows flagged)
- Cost sensitivity: 0.05%->48.3%, 0.10%->46.9%, 0.15%->45.5%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 1/4 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: full universe, train=30mo/test=6mo): PASSES all 3 viability criteria**

## Viability check: sensitivity: new-only 74, train=30mo/test=6mo (74 tickers)

- Tickers: ['AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAX', 'BBWI', 'CCL', 'CDW', 'CLX', 'CMI', 'CPRT', 'CTAS', 'CTSH', 'DGX', 'DHI', 'DPZ', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'HBAN', 'HCA', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'KLAC', 'KR', 'L', 'LH', 'LRCX', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'USB', 'VRSK', 'WDC', 'WHR', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 30mo train / 6mo test, 4 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=0.515, OOS Sharpe=0.595, OOS trades=34, OK
- Window 1: params thresh=3.0% hold=60d, IS Sharpe=0.749, OOS Sharpe=0.035, OOS trades=35, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.008, OOS Sharpe=0.926, OOS trades=33, OK
- Window 3: params thresh=10.0% hold=20d, IS Sharpe=1.097, OOS Sharpe=0.957, OOS trades=23, OK

### Combined out-of-sample result (sensitivity: new-only 74, train=30mo/test=6mo)

- Sharpe=0.613  Sortino=0.620  MaxDD=39.2%  MC_DD95=17.5%  PF=1.731  WinRate=60.0%  Trades=125  (1/4 windows flagged)
- Cost sensitivity: 0.05%->40.9%, 0.10%->39.6%, 0.15%->38.3%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 1/4 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: new-only 74, train=30mo/test=6mo): PASSES all 3 viability criteria**

## Viability check: sensitivity: full universe, train=36mo/test=6mo (97 tickers)

- Tickers: ['ABBV', 'AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAC', 'BAX', 'BBWI', 'CAT', 'CCL', 'CDW', 'CLX', 'CMI', 'COST', 'CPRT', 'CTAS', 'CTSH', 'CVX', 'DGX', 'DHI', 'DIS', 'DPZ', 'DUK', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'GS', 'HBAN', 'HCA', 'HD', 'HON', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'JNJ', 'JPM', 'KLAC', 'KO', 'KR', 'L', 'LH', 'LIN', 'LRCX', 'MCD', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PFE', 'PG', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'UNH', 'UPS', 'USB', 'V', 'VRSK', 'VZ', 'WDC', 'WHR', 'WMT', 'XOM', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 36mo train / 6mo test, 3 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=0.602, OOS Sharpe=0.085, OOS trades=36, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=10.0% hold=60d, IS Sharpe=0.592, OOS Sharpe=0.758, OOS trades=30, OK
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.112, OOS Sharpe=1.215, OOS trades=35, OK

### Combined out-of-sample result (sensitivity: full universe, train=36mo/test=6mo)

- Sharpe=0.656  Sortino=0.674  MaxDD=39.7%  MC_DD95=16.0%  PF=1.857  WinRate=56.4%  Trades=101  (1/3 windows flagged)
- Cost sensitivity: 0.05%->41.9%, 0.10%->40.9%, 0.15%->39.8%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 1/3 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: full universe, train=36mo/test=6mo): PASSES all 3 viability criteria**

## Viability check: sensitivity: new-only 74, train=36mo/test=6mo (74 tickers)

- Tickers: ['AME', 'AMGN', 'AON', 'AXON', 'AXP', 'BAX', 'BBWI', 'CCL', 'CDW', 'CLX', 'CMI', 'CPRT', 'CTAS', 'CTSH', 'DGX', 'DHI', 'DPZ', 'ELV', 'EMR', 'EOG', 'EQT', 'ETN', 'FDS', 'FIS', 'GILD', 'GL', 'GRMN', 'HBAN', 'HCA', 'HSIC', 'HUM', 'INCY', 'INTU', 'IVZ', 'J', 'KLAC', 'KR', 'L', 'LH', 'LRCX', 'MCHP', 'MHK', 'MKC', 'MKTX', 'MOH', 'MPWR', 'NDAQ', 'NVR', 'NWL', 'PANW', 'PCAR', 'PGR', 'PHM', 'PODD', 'PPL', 'REGN', 'RF', 'RMD', 'ROP', 'SJM', 'SNPS', 'STZ', 'TER', 'TFC', 'TPR', 'TROW', 'TT', 'ULTA', 'USB', 'VRSK', 'WDC', 'WHR', 'ZBH', 'ZTS']
- Same methodology throughout: exposure cap 1.5x, max 3 new positions/day, priority by surprise magnitude, NO regime filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). Grid [3.0, 5.0, 7.0, 10.0] x [20, 40, 60], walk-forward 36mo train / 6mo test, 3 windows over 2021-08-03 to 2026-07-01.
- Window 0: params thresh=5.0% hold=60d, IS Sharpe=0.597, OOS Sharpe=-0.328, OOS trades=33, FLAG (>30% IS/OOS deviation)
- Window 1: params thresh=10.0% hold=60d, IS Sharpe=0.636, OOS Sharpe=1.096, OOS trades=26, FLAG (>30% IS/OOS deviation)
- Window 2: params thresh=7.0% hold=60d, IS Sharpe=1.104, OOS Sharpe=1.451, OOS trades=29, FLAG (>30% IS/OOS deviation)

### Combined out-of-sample result (sensitivity: new-only 74, train=36mo/test=6mo)

- Sharpe=0.669  Sortino=0.748  MaxDD=41.2%  MC_DD95=16.8%  PF=1.879  WinRate=55.7%  Trades=88  (3/3 windows flagged)
- Cost sensitivity: 0.05%->41.1%, 0.10%->40.2%, 0.15%->39.3%

**Viability check: Sharpe>=0.5: True | MC_DD95<=20%: True | edge survives 0.15% slippage: True | 3/3 windows flagged for IS/OOS deviation >30%**

**VERDICT (sensitivity: new-only 74, train=36mo/test=6mo): PASSES all 3 viability criteria**

### Sensitivity summary table

| train_months | Full-97 Sharpe | Full-97 MC_DD95 | Full-97 windows flagged | New-74 Sharpe | New-74 MC_DD95 | New-74 windows flagged |
|---|---|---|---|---|---|---|
| 12 | 0.641 | 21.8% | 5/7 | 0.592 | 23.9% | 6/7 |
| 18 | 0.583 | 19.6% | 4/6 | 0.672 | 18.6% | 5/6 |
| 24 | 0.884 | 14.2% | 3/5 | 0.718 | 16.3% | 4/5 |
| 30 | 0.653 | 16.6% | 1/4 | 0.613 | 17.5% | 1/4 |
| 36 | 0.656 | 16.0% | 1/3 | 0.669 | 16.8% | 3/3 |

- Full-97 Sharpe range across train-window widths: 0.583 to 0.884 (4/5 pass)
- New-only-74 Sharpe range across train-window widths: 0.592 to 0.718 (4/5 pass)

**CONCLUSION: the pass verdict is ROBUST to train-window width -- Sharpe stays above the 0.5 bar across all 5 widths tested on both universes, not just the originally-chosen 24mo.**

**Correction/completion to the auto-generated conclusion above** (which only
checked the Sharpe criterion): the FULL 3-criteria pass is NOT robust to
train-window width. At train_months=12, **MC_DD95 breaches the 20% bar on
both universes (21.8% full-97, 23.9% new-only-74)** -- Sharpe alone staying
above 0.5 does not mean the complete viability check passes at every width.
So one of five tested widths (12mo) fails outright on the drawdown
criterion specifically, even though it passes on Sharpe.

**Window-level instability was requested to be documented as a serious
reservation, not a footnote -- doing so explicitly here.** Across all 5
train-window widths and both universes (10 total runs), the fraction of
walk-forward windows flagged for >30% IS/OOS Sharpe deviation ranges from
33% (1/3) to 100% (3/3, 6/7) and is NEVER below 1/3. This has been true
in literally every version of this backtest run across the entire PEAD
investigation, from the original 23-ticker baseline (7/7 flagged) through
every subsequent expansion and correction. **The walk-forward's own
parameter selection has never once converged on a stable, generalizable
(threshold, hold_days) choice across a majority of its windows, in any
configuration tested.** This means the reported aggregate Sharpe, even at
its best (0.884, 24mo/97-tickers), is being achieved despite -- not
because of -- a well-behaved, predictable parameter-selection process.
It should be read as "the aggregate direction is probably positive" and
NOT as "a specific, identified (threshold, hold_days) rule reliably
predicts future performance," which is a meaningfully weaker claim.

**Complete, corrected verdict**: Sharpe robustly clears 0.5 across all 5
train-window widths tested (0.583-0.884 full-97, 0.592-0.718 new-only-74)
-- this part of the result is NOT fragile to reasonable methodology
choices. But the full 3-criteria pass is fragile (MC_DD95 fails at the
shortest tested window), and window-level parameter instability -- the
single most consistent concern across this entire project -- was never
resolved at any point, in any configuration. BK/DFS inclusion/exclusion
is a non-issue: they never cached (zero usable data) and are excluded
from every run by construction, not by choice.

---

## Root-cause investigation: why do (threshold, hold_days) never converge across windows?

**Method**: `diagnose_pead_param_instability.py` dumps the FULL 12-cell grid
(all 4 thresholds x 3 hold_days) per window on the full-97 universe --
IS Sharpe, IS trade count, OOS Sharpe, OOS trade count for every combo, not
just the argmax -- plus each window's train/test regime mix. Two hypotheses
tested: (a) per-cell trade counts too thin for the IS Sharpe estimate to be
reliable, (b) the true optimum genuinely shifts with market regime.

**Finding 1 -- not a small-sample problem in the usual sense**: IS trade
counts per grid cell range 115-225 (median 148), not tiny.

**Finding 2 -- the IS-argmax selection is actively anti-informative, not
just noisy**: ranking each window's 12 combos by OOS Sharpe and checking
where the IS-argmax pick lands: windows 0-4 ranked #7, #12, #6, #10, #8
of 12 -- mean rank 8.6/12, WORSE than the ~6.5 expected from picking at
random. The top 3 IS Sharpes per window are separated by only 0.07-0.26 --
statistical ties -- so the "winner" is close to an arbitrary draw among
near-equal candidates, and that draw is biased the wrong way.

**Finding 3 -- root cause identified**: thresh=10%/hold=20d had the WORST
or near-worst IS Sharpe rank in windows 0-1 (#11, #12) yet was the best OOS
performer in EVERY window (mean OOS Sharpe 1.50, range 1.29-1.73, zero
negative windows) -- the single most stable combo on the grid. Longer
holds (40-60d) systematically won the IS-argmax instead, because over any
one ~24mo training window they capture more of that window's OWN
idiosyncratic trend (whatever direction the market happened to drift over
40-60 days), which inflates in-sample Sharpe in a way specific to that
window's realized path and does not carry into the next window. This
matches the academic PEAD literature (drift concentrated shortly after a
large surprise, contaminated by unrelated price action the longer the
hold) and explains why the argmax mechanism is worse than random: it's
systematically selecting for trend-riding within the training window, not
for the generalizable part of the signal.

**Independent cross-check (not the same cherry-pick restated)**:
`diagnose_pead_param_instability_crosscheck.py` reruns the pooled-OOS-by-combo
table on the DISJOINT 74-new-only ticker universe. Same standout: thresh=10%/
hold=20d again ranks #1 (mean OOS Sharpe 1.451, std 0.275, worst window
still +0.957 -- zero negative windows), while low-threshold/mid-hold combos
(3%/40d, 5%/40d) again cluster at the bottom with high variance and
negative windows. Confirms the pattern is a real attribute of the signal
across an independent ticker sample, not noise specific to the original 97.

**Regime dependency reconfirmed as NOT the primary driver of *parameter*
instability** (consistent with the earlier entry-day regime-filter
rejection): dominant train-period regime does not cleanly predict which
combo the IS-argmax picks (both CORRECTION-dominant windows 0-1 and
BULLISH-dominant windows 3-4 land on different, not regime-patterned,
choices). The instability is explained by Finding 3's selection-mechanism
bias, not by a clean regime split.

## Structural fix tested: freeze params, stop re-optimizing per window

`run_pead_frozen_params.py`: thresh=10%/hold=20d frozen globally (chosen
for being the top-mean, lowest-variance, only-ever-positive combo on BOTH
independent universes above -- not picked from a single test), no
per-window re-optimization at all.

- **Does resolve the parameter-instability artifact itself**: by
  construction there is no more window-to-window swinging choice, and the
  frozen combo is the most stable OOS performer found across every slice
  examined on both universes.
- **Does NOT resolve overall viability through a full cycle**: run as ONE
  continuous simulation over the full available history (2021-08-03 to
  2026-07-01, i.e. including the 2021-2022 bear market that none of the 5
  walk-forward TEST windows happen to cover -- they only span 2023-08 to
  2026-02, a period that is 48-100% BULLISH-labeled in every window per
  the earlier regime breakdown):
  - Full-97: Sharpe=0.479 (FAILS the >=0.5 bar), MaxDD=29.1% (real,
    historical), MC_DD95=15.9%, cost-15bps survival=+45.1%.
  - New-only-74: Sharpe=0.505 (barely passes), MaxDD=27.5% (real),
    MC_DD95=15.7%, cost-15bps survival=+47.0%.
  - The real historical MaxDD (27.5-29.1%) is nearly double the
    Monte-Carlo-estimated 95th-percentile drawdown (15.7-15.9%) -- MC
    resampling (i.i.d. trade order) understates the actual clustered loss
    that occurred in the real 2021-2022 sequence.

**Conclusion**: the parameter-instability question now has a real,
evidenced, structural answer (selection-mechanism bias toward
trend-contaminated long holds, confirmed on two independent ticker sets) --
this is a genuine finding, not "still a mystery." But fixing it is not
sufficient to clear this project's own viability bar: the moment the frozen
rule is tested through an actual correction (2021-2022, absent from every
walk-forward test window used elsewhere in this investigation), Sharpe
falls to the failing/barely-passing boundary and real drawdown roughly
doubles versus its Monte-Carlo estimate. This reconfirms and sharpens the
earlier "Regime pattern investigation" finding (positive OOS windows were
all BULLISH-dominant, negative ones all CORRECTION-dominant) at the level
of a single frozen rule rather than a shifting per-window choice, and the
one regime-aware fix attempted for that problem (entry-day filter) was
already rejected as non-causal. **No validated correction-robust
construction exists yet.** Per the project's viability bar, this is not a
signal solid enough to freeze into a single rule for paper trading --
aggregate Sharpe alone (even a stable 0.88-1.5 across bullish-only test
windows) was masking this, exactly the failure mode the full-cycle
continuous check above was built to catch.

---

## FINAL ITERATION (last one, per explicit decision): mid-hold early exit

**Mechanism**: frozen thresh=10%/hold=20d (established above as the only
combo stable across every window on both universes) + a market-wide
mid-hold early exit -- if SPY's close drops below its own 200-day SMA on
any day during a position's hold, exit that day instead of waiting for the
full hold_days. This is the log's own previously-suggested alternative to
the rejected entry-day filter ("account for the regime trajectory across
the WHOLE hold period ... a smoother, less binary signal e.g. SPY vs its
own SMA200"). No parameters tuned on this data -- 200d SMA is a standard
convention. Implemented in `bot/pead_backtest.py` (`early_exit_by_date`
param on `_raw_candidates`/`build_portfolio_event_trades`); tested via
`run_pead_early_exit.py`. SMA200 flags 19.6% of days (2020-2026) as
below-trend.

Evaluated as ONE continuous run over the full available history
(2021-08-03 to 2026-07-01, including the 2021-2022 correction), baseline
vs early-exit, on both the full-97 and independent new-only-74 universes:

| Universe | Variant | Sharpe | MC_DD95 | Real MaxDD | Cost-15bps | Verdict |
|---|---|---|---|---|---|---|
| Full-97 | baseline | 0.479 | 15.9% | 29.1% | +45.1% | FAILS (Sharpe) |
| Full-97 | + early exit | **0.545** | 16.0% | **22.3%** | +34.7% | **PASSES all 3** |
| New-only-74 | baseline | 0.505 | 15.7% | 27.5% | +47.0% | PASSES (barely) |
| New-only-74 | + early exit | **0.555** | 16.2% | **20.3%** | +34.1% | **PASSES all 3** |

**Regime breakdown confirms the mechanism is causal, not coincidental**:
CORRECTION-period losses roughly HALVE on both universes (full-97:
-23.5%->-11.3%; new-only-74: -24.8%->-11.3%), exactly the failure mode
this fix targets, reproduced independently on the disjoint ticker set.
Real historical MaxDD (the metric that previously ran ~2x the
Monte-Carlo estimate, exposing that MC understated true clustered risk)
now sits much closer to its MC estimate on both universes -- the
tail-clustering gap that flagged the earlier frozen-only version as
unreliable has visibly narrowed.

**Honest costs of this fix, not hidden**: NEUTRAL-regime performance gets
WORSE (full-97: sharpe 0.67->-1.68; new-only-74: 1.10->-1.43) -- the SMA200
signal is blunt enough to cut some genuinely good non-correction trades
short too, a real tradeoff, not a free lunch. Cost-sensitivity decay is
somewhat steeper (more turnover from extra exits) though survival stays
comfortably positive. And critically: both variants are still evaluated
against the SAME single historical correction (2021-2022) -- cross-
sectionally independent (74 disjoint tickers corroborate the same result)
but NOT time-independent (there is only one bear market in the available
data to validate against, a limitation repeatedly flagged throughout this
investigation and not resolved by this fix).

**VERDICT: PASSES all 3 viability criteria on both universes, with the
improvement concentrated exactly where the previously-diagnosed failure
mode was, reproduced independently on disjoint tickers.** Margins above
the Sharpe>=0.5 bar are positive but not large (0.045-0.055) -- flagged
plainly, not overstated. This is the result of the final iteration as
agreed; no further iteration planned regardless of outcome.

---

## SECOND-RECESSION TEST (final iteration, per explicit pre-commitment): 2016-2020

**Data coverage investigation (requested, documented honestly)**: 2007-2009
is confirmed NOT reachable -- bisected year-by-year against the live Alpaca
API (JPM and SPY, SIP feed): zero rows for every year 2005-2015, 252 real
rows starting exactly 2016-01-04. Hard account/API floor, not a coverage
nuance. Alpha Vantage earnings data is not the constraint (92/97 tickers
have earnings history well before 2010). Separately discovered along the
way: this project's DEFAULT feed (`config.ALPACA_DATA_FEED="iex"`) has an
even later floor for individual equities -- IEX-sourced bars for JPM return
ZERO rows before ~2020-07 even though SIP covers the same symbol back to
2016-01-04. All earlier PEAD runs in this investigation were unknowingly
constrained to 2020-07-27 onward by this IEX floor, not by a true absence
of earlier data -- worth knowing for any future investigation that revisits
this data source, though it does not change any verdict already reached
(the 2021-2026 tests remain valid; they just could have started up to ~4
years earlier had SIP been requested from the start).

Using SIP explicitly (isolated fetch, `use_cache=False`, no shared-cache
contamination) for 2016-11-01 (leaves ~200 trading days for SPY's own
SMA200 to be valid at window start) to 2020-07-26 (no overlap with the
already-tested 2020-2026 period): **95 of 97 tickers have full price
coverage** (BBWI and LIN excluded -- genuinely IPO'd/spun off after window
start). This window contains two independent stress episodes never used to
build this mechanism: the Q4 2018 correction and the Feb-Mar 2020 COVID
crash.

**Result: thresh=10%/hold=20d + SPY-SMA200 mid-hold early exit -- exact
same frozen mechanism, zero new parameters -- FAILS clearly on this second
recession, on both universes:**

| Universe | Variant | Sharpe | MC_DD95 | Cost-15bps | Verdict |
|---|---|---|---|---|---|
| Full-95 | baseline | 0.483 | 16.4% | +25.9% | FAILS (Sharpe, barely) |
| Full-95 | + early exit | **0.121** | 19.8% | **+0.9%** | **FAILS clearly** |
| New-only-73 | baseline | 0.589 | 15.0% | +30.7% | PASSES all 3 |
| New-only-73 | + early exit | **0.175** | 18.7% | **+3.3%** | **FAILS clearly** |

The early-exit mechanism made things WORSE on both universes, not better --
Sharpe collapsed by 75-70%, and the cost-15bps survival margin nearly
vanished (0.9%/3.3%, i.e. realistic trading costs would likely erase the
edge entirely). This is the opposite of what happened in 2021-2022.

**Why, diagnosed from the sub-period and regime breakdown**: the mechanism
helped in the COVID crash specifically (pnl -5060->+495, full-97) but badly
hurt the Q4-2018 correction (pnl +11543->+2371, an ~80% cut to a genuinely
profitable episode) and cut BULLISH-regime returns nearly in half
(19.2%->10.1%) and NEUTRAL-regime returns from +2.2% to -10.1%. SPY's
200-day SMA is a slow, lagging filter -- it works when a correction is a
prolonged grind (2021-2022, where SPY spent extended periods below its
SMA200) but is actively harmful in a fast, V-shaped correction (Q4 2018:
~3 months top to bottom) because it triggers well after the damage is
already done and then keeps positions out during the sharp recovery that
typically follows a V-shaped bottom -- cutting the recovery, not just the
drawdown. The COVID crash partially validated the mechanism only because
its recovery, unusually, stayed muted long enough for the lagging signal
to still be flagged "below trend" through part of the rebound.

**CONCLUSION -- PEAD investigation retired, per explicit pre-commitment
(no further iteration regardless of outcome).** The mid-hold early-exit
fix that cleared all 3 viability criteria on 2021-2022 does NOT generalize
to a second, independent recession (Q4 2018) with a different shape (fast
V-shaped vs. slow grinding bear market) -- confirmed on two independent
ticker universes, with solid data coverage (95/97 tickers), not a thin-data
artifact. This is exactly the kind of result the two-recession test was
designed to catch, and it caught it: the earlier "clear pass" on
2021-2022 was itself a form of overfitting to that specific recession's
shape, not evidence of a mechanism that structurally protects PEAD through
any downturn. **No further work on PEAD is planned.** `bot/main.py`
remains disabled (Strategies A/B). PEAD stays as a validated-negative
research result, documented here for future reference, not deployed to
paper trading or live trading in any form.

---

## LESSON LEARNED -- read this first before attempting any future regime/trend filter

**PEAD is retired for good (2026-07-18). No further iteration planned or
authorized.** The reusable lesson from this entire investigation, for any
future regime-aware construction on ANY strategy in this repo, not just
PEAD:

**A slow/lagging trend filter (SMA200, or anything with a similar
multi-month reaction lag) is not a general-purpose "avoid bad markets"
tool. It specifically helps against SLOW, GRINDING drawdowns (2021-2022:
SPY spent extended periods below its own SMA200, so the filter was
correctly "on" for most of the damage) and specifically HURTS during FAST,
V-SHAPED corrections (Q4 2018: ~3 months top-to-bottom) because it
triggers well after the drop has already happened and then stays "on"
through the sharp recovery that typically follows a V-bottom -- cutting
the recovery, not the drawdown. Tested and confirmed empirically here:
same exact mechanism, zero new parameters, cleared all 3 viability
criteria on the slow 2021-2022 bear market and then failed clearly (Sharpe
down 70-75%, cost-margin nearly wiped out) on the fast 2018-2020 window
containing Q4 2018 + the COVID crash.

**Implication for later attempts**: a viable regime filter would need to
distinguish the SHAPE of the downturn (grinding vs. V-shaped), not just
detect "market is down" -- a single lagging indicator cannot do both. A
result that clears all viability criteria on ONE historical recession is
not evidence of a working regime filter; it is, at best, evidence the
filter's specific lag happened to match that one recession's specific
shape. Any future attempt MUST be checked against at least one
structurally different downturn (different speed/shape, not just a
different date range) before being trusted -- exactly the check that
caught this one. This project's default 5-window walk-forward (2023-2026)
would NOT have caught this failure mode on its own -- it never contained a
fast V-shaped correction; the failure only surfaced by deliberately
reaching back to 2016-2020 for a structurally different stress episode.
