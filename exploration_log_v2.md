# SYNTHESIS -- read this first (updated after independent validation)

**One-sentence verdict: the signal passes the pre-declared numeric gate on
a fresh ticker universe, but fails badly on a fresh time period using the
exact same configuration -- the pattern (works across many ticker draws
within 2022-2026, reverses sign entirely on 2020-2022) is the signature of
a period-specific effect, not a stable, generalizable edge, and I would NOT
proceed toward live capital on this basis.**

## What was tested

Two independent checks on Direction 2's finding (mean reversion + SMA200
trend filter, daily bars, broad equity universe; original 56-ticker result:
7 passers, combined Sharpe 1.90):

**Check 1 -- fresh, objectively-drawn universe, same 2022-2026 period.**
56 tickers drawn via `np.random.default_rng(seed=20260703)` from a
documented 278-ticker S&P 500 reference pool (`bot/sp500_pool.py`) minus
the 56 already tested -- not hand-picked. Same exact grid, same 30%
anti-overfitting rule, same viability bar.
- Result: 2 of 56 passed (FANG, GRMN) vs. 7 of 56 originally (3.6% vs
  12.5% hit rate). Combined Sharpe = **1.140** vs. original 1.896.
- **Passes the pre-declared gate** (must stay > 0.5 AND > 0.948 [50% of
  1.896]) -- but the signal clearly weakened, both in hit rate and in
  combined Sharpe, even though it didn't cross the 50% failure line.

**Step 2A -- cautious expansion, unchanged criteria, triggered because
check 1 passed.** All 222 remaining tickers in the same S&P 500 pool
(56 original + 56 validation draw already excluded), same grid, same
filter, same bar -- no parameter changes.
- Result: 29 of 222 passed (13.1%). Combined Sharpe of this batch alone =
  **3.184**, MaxDD 0.4%, PF 2.532, 519 trades.
- **Cumulative accounting across all three batches: 334 tickers tested
  total, 38 passed all 3 criteria (11.4% overall hit rate)** -- consistent
  across batches 1 and 3 (12.5%, 13.1%) with batch 2 lower (3.6%). The
  combined Sharpe increasing from 1.90 (7 names) to 3.18 (29 names) is
  expected under standard diversification math (~sqrt(N) scaling) when
  combining many largely-independent, similarly-signed small edges -- it
  is NOT independent additional evidence beyond what the per-ticker hit
  rate already showed; it's a mathematical consequence of that hit rate
  being real and roughly uncorrelated across names, not a new data point.

**Check 2 -- fresh time window, SAME 7 original tickers, SAME
already-chosen parameters (no re-fitting), 2020-07-27 to 2022-06-03.** This
period is entirely before the earliest train_start (2022-06-03) used
anywhere in the original walk-forward -- genuinely never touched by any
train or test split before now. This is the more decisive check because it
holds BOTH the tickers and the parameters fixed and only changes the time
period, isolating exactly the question "does this specific, already-proven
configuration generalize across time."
- Result: **combined Sharpe = -0.969** (vs. +1.896 in 2022-2026). Individual
  tickers: META -0.92 (2 trades), NVDA -1.00, MRK undefined (0 trades),
  COP +0.81, NKE -0.64, PYPL -0.56, GE +0.62. Only 2 of 7 stayed positive.

## How to reconcile these two results

Check 1 and step 2A show the effect is NOT an artifact of the specific 56
tickers originally picked -- a blind draw from a much larger pool, and then
a full sweep of everything remaining, both reproduce a similar ~3-13% hit
rate and the same diversification-amplified combined Sharpe, within the
SAME 2022-2026 period. That's real evidence against "these 7 tickers were
just lucky."

But check 2 shows the effect does NOT survive moving to a different market
period with the identical setup. Put together, the most honest reading is:
**something about the 2022-2026 period specifically (dominated by a broad,
sustained mega-cap/growth rally with periodic pullbacks -- exactly the
shape "buy dips above the 200-day SMA" is built to catch) made this signal
work across many names drawn from that period, but the signal is reading
the character of ONE market regime, not a timeless mean-reversion
phenomenon.** 2020-2022 (COVID crash, V-shaped recovery, 2022 bear market)
has a very different character, and the same rule fails there. A real,
stable edge should have degraded gracefully across time periods, not
flipped sign entirely.

## Verdict on directions 1, 3, 4 (unchanged from before validation)

No viable signal in any of them -- direction 1 (wider universe, same
intraday logic) reproduces round 1's SPY/QQQ whipsaw catastrophe on
mid-caps; direction 3 (cross-sectional momentum) never found a stable
parameter plateau; direction 4 (portfolio macro gate) made no measurable
difference vs. each instrument's own trend filter.

## Recommendation

**Do not proceed to live capital on the Direction 2 finding as it stands.**
The independent-universe check passing was necessary but not sufficient --
the fresh-time-window check is the one that most directly simulates "would
this have worked if discovered before 2022," and it says no. This doesn't
necessarily mean SMA200-filtered mean reversion has zero merit in all
conditions, but it means the specific configuration validated here is most
likely capturing 2022-2026's particular market character rather than a
portable edge, and presenting it as validated would be overselling the
result. Per your instruction not to launch a further exploration round
without explicit sign-off: stopping here. If you want to pursue this
further, the natural next step would be testing on additional independent
historical periods as they become available (the current data frontier is
~2020-07 backward and "now" forward), or testing whether the signal is
IS_OOS-recoverable if 2022-2026-specific features (e.g. explicit reference
to prevailing multi-year trend regime) are added explicitly rather than
implicitly assumed -- but that would be new-hypothesis testing, not
validation of what's already been proposed, and needs your go-ahead.

---

# Exploration log v2 -- structurally different directions

Following last night's conclusion (classic mean-reversion/breakout/trend
signals have no exploitable edge on SPY/QQQ/BTC/GLD/USO after costs), this
round tests structurally different directions rather than continuing to
tune the same signal families:

1. Same validated logic (mean reversion + SMA200 trend filter; breakout) on
   a wider, less-arbitraged instrument universe (20 mid-cap equities, 10
   liquid crypto alt-coins).
2. Same logic on daily bars across a broad 50+ ticker equity universe
   (different timeframe -- less microstructure noise, less HFT competition).
3. Cross-sectional momentum (rank a basket by 3/6/12-month trailing return,
   hold the top quintile, rebalance monthly) -- a fundamentally different
   strategy family with real academic support (Jegadeesh & Titman), not a
   time-series signal on a single instrument.
4. Portfolio-level SPY-SMA200 macro gate on top of the existing mean
   reversion + trend filter, extending the one intervention that showed a
   real effect in round 1.

Rules carried over unchanged: never optimize/rank on win rate; reject any
OOS Sharpe deviating >30% from IS Sharpe; viability = OOS Sharpe >= 0.5 AND
MC 95th-pct drawdown <= 20% AND edge survives 0.15% slippage. Scope is
capped at these 4 directions -- if none pass, stop and report honestly
rather than inventing a 5th.

---

## Direction 1: wider instrument universe, same validated intraday logic

- Tested 20 mid-cap equities with mean_reversion_trendfilter. 16 produced a surviving plateau. Top 5 by OOS Sharpe:
    APA: Sharpe=-1.166 PF=0.731 MaxDD=78.5% MC_DD95=91.5% trades=457 cost0.15%=-92.0%
    KSS: Sharpe=-1.184 PF=0.703 MaxDD=88.8% MC_DD95=95.7% trades=439 cost0.15%=-92.0%
    ANF: Sharpe=-1.292 PF=0.731 MaxDD=97.5% MC_DD95=99.5% trades=772 cost0.15%=-99.3%
    ALK: Sharpe=-1.320 PF=0.739 MaxDD=89.8% MC_DD95=96.0% trades=456 cost0.15%=-96.7%
    URBN: Sharpe=-1.608 PF=0.664 MaxDD=90.9% MC_DD95=95.5% trades=448 cost0.15%=-96.6%
- Tested 10 crypto alt-coins with momentum_breakout. Top 5 by OOS Sharpe:
    AVAX/USD: Sharpe=0.378 PF=1.124 MaxDD=42.7% MC_DD95=68.4% trades=480 cost0.15%=-15.4%
    ETH/USD: Sharpe=0.085 PF=0.987 MaxDD=58.7% MC_DD95=84.1% trades=563 cost0.15%=-59.1%
    LINK/USD: Sharpe=-0.442 PF=0.848 MaxDD=58.3% MC_DD95=85.6% trades=445 cost0.15%=-71.9%
    AAVE/USD: Sharpe=-0.700 PF=0.809 MaxDD=79.5% MC_DD95=90.3% trades=498 cost0.15%=-77.8%
    CRV/USD: Sharpe=-0.866 PF=0.760 MaxDD=57.4% MC_DD95=78.4% trades=240 cost0.15%=-56.7%

## Direction 2: daily bars, broad liquid equity universe

- Tested 56 liquid equities on DAILY bars with mean_reversion_trendfilter. 34 produced a surviving plateau. Top 5 by OOS Sharpe:
    META: Sharpe=1.200 PF=inf MaxDD=2.3% MC_DD95=0.0% trades=8 cost0.15%=14.1%
    NVDA: Sharpe=0.975 PF=2.338 MaxDD=4.0% MC_DD95=8.7% trades=31 cost0.15%=19.4%
    COP: Sharpe=0.931 PF=2.406 MaxDD=4.8% MC_DD95=7.2% trades=23 cost0.15%=16.2%
    NKE: Sharpe=0.661 PF=1.689 MaxDD=4.9% MC_DD95=11.8% trades=30 cost0.15%=11.6%
    GE: Sharpe=0.625 PF=1.909 MaxDD=6.5% MC_DD95=10.6% trades=19 cost0.15%=10.1%

### Direction 2 deep-dive: full 56-ticker breakdown + multiple-testing check

This is the first result across BOTH exploration rounds to show multiple
candidates clearing all three viability criteria, so per the "iterate once
more on a promising direction" rule, it got one additional validation pass
before being trusted.

**Full breakdown**: of the 56 tickers tested, 34 survived the anti-overfitting
plateau filter, and **7 of 56 (12.5%) passed all three viability criteria**
(Sharpe OOS >= 0.5, MC 95th-pct drawdown <= 20%, edge survives 0.15%
slippage): META, NVDA, MRK, COP, NKE, PYPL, GE.

**Multiple-testing concern (the obvious objection)**: testing 56 independent
tickers and reporting the winners is a textbook multiple-comparisons setup.
A rough back-of-envelope null-rate estimate (if the combined viability bar
had, hypothetically, a ~10% chance of being cleared by pure luck for an
edge-less strategy) would predict ~5-6 false positives out of 56 trials by
chance alone -- uncomfortably close to the 7 actually found. On its own,
"7/56 passed" is NOT strong evidence of a real, common edge.

**Decisive check: does the edge diversify or regress to noise?** Combined
the 7 passing tickers into an equal-weighted daily-rebalanced portfolio
(averaging their independently-tested OOS daily returns, concatenating their
trade lists for PF/win-rate). If the 7 hits are independent lucky draws with
no shared underlying signal, combining them should show the Sharpe REGRESS
TOWARD ZERO (averaging real edge with noise dilutes both). If there's a real,
common phenomenon ("buying a dip above the 200-day SMA has positive
expectancy"), combining largely-uncorrelated names (tech, pharma, energy,
consumer, fintech, industrials) should show a Sharpe AS GOOD OR BETTER than
any individual leg, via standard diversification.

Result: **the combined portfolio's Sharpe (1.90) is HIGHER than every
individual ticker's Sharpe (max individually was META at 1.20)**, with a
much smaller max drawdown (1.2% vs each individual ticker's own 2-7%) and
136 total trades. This is the diversification signature, not the noise
signature -- strong evidence this is a real (if modest, thin-sample) common
effect rather than 7 independent false positives:

| | Combined portfolio (7 tickers, equal-weight) |
|---|---|
| Sharpe (0.10% slippage) | 1.896 |
| Max drawdown | 1.2% |
| MC 95th-pct drawdown | 12.5% |
| Profit factor | 2.283 |
| Total trades | 136 |
| Total return @ 0.05% slippage | +14.0% |
| Total return @ 0.10% slippage | +13.2% |
| Total return @ 0.15% slippage | +12.6% |

Edge survives all three cost scenarios comfortably (Sharpe stays >1.8 even
at 0.15% slippage). **This combined portfolio passes all three viability
criteria robustly, not marginally.**

**Remaining caveats, stated plainly**:
- Per-ticker trade counts are still thin (8-31 each; 136 combined over 4
  years) -- meaningfully more data than GLD/USO's 7-15, but still nowhere
  near the volume behind SPY/QQQ's catastrophic (but statistically solid)
  results in round 1.
- NVDA and META are famous, well-known outsized winners of the exact
  2022-2026 period tested (both had enormous AI-driven bull runs) -- there
  is a real risk that picking recognizable mega-cap names for a "wider
  universe" test reintroduces a hindsight-selection bias (of course a
  buy-the-dip strategy looks good on a stock that went up 5-10x). This is
  partially, but not fully, mitigated by MRK/COP/NKE/PYPL/GE also passing
  independently -- these are not "famous winner" picks (COP is oil & gas,
  MRK is pharma, GE had a rough 2022 before a partial recovery), which
  argues the effect isn't purely an NVDA/META artifact.
- This was tested on ONE specific 56-ticker universe choice (a hand-picked
  liquid large/mid-cap list) and ONE specific 4-year window. It has not been
  tested on a genuinely fresh, never-touched holdout period or an
  independently-drawn ticker universe -- the walk-forward OOS windows are
  real out-of-sample splits, but the universe-selection and signal-family
  choice were both informed by round 1's findings, which is a softer form
  of researcher degrees of freedom that a true blind holdout would remove.

**Robustness check: drop NVDA + META entirely.** Rebuilt the combined
portfolio from just the remaining 5 non-mega-cap passers (MRK, COP, NKE,
PYPL, GE). Result: **Sharpe=1.444, MaxDD=1.7%, MC_DD95=13.7%, PF=2.014, 97
trades -- still passes both Sharpe and drawdown viability comfortably.**
This substantially weakens the hindsight-selection-bias objection: the
effect isn't an artifact of including two famous 2023-2024 mega-cap winners,
it holds among five considerably more "boring" names.

## Direction 3: cross-sectional momentum (monthly rebalance, top-quintile)

- Price matrix: 56 tickers x 1021 days
- 36 (lookback,window) combos tested, 35 rejected by the 30% IS/OOS deviation rule. 0 lookback values survived as a plateau.
  No lookback value survived the anti-overfitting filter as a stable plateau.

**Verdict: no signal.** On a monthly rebalance cadence with only 12
walk-forward windows, each lookback choice effectively only gets a handful
of independent monthly observations per window, so IS/OOS Sharpe swings
wildly (same thin-data problem as GLD/USO in round 1, but worse here since
monthly rebalancing produces even fewer independent decision points than
daily bars do). Per the time-budget rule, this direction showed no sign of
life after a reasonable look and was not pursued further.

## Direction 4: portfolio-level SPY-SMA200 macro gate on mean reversion + trend filter

- **SPY / mean_reversion + trendfilter + SPY macro gate**: 36 combos, 18 rejected. 3 param sets survived.
  best params={'sma_period': 20, 'std_dev_mult': 2.0, 'trend_sma_period': 200}: Sharpe=-4.152 PF=0.240 MaxDD=99.9% MC_DD95=99.9% trades=476 cost0.15%=-100.0%
- **QQQ / mean_reversion + trendfilter + SPY macro gate**: 36 combos, 23 rejected. 3 param sets survived.
  best params={'sma_period': 20, 'std_dev_mult': 2.4, 'trend_sma_period': 200}: Sharpe=-2.876 PF=0.342 MaxDD=96.3% MC_DD95=97.9% trades=279 cost0.15%=-99.3%

**Verdict: no signal, and notably WORSE than the per-instrument trend filter
alone from round 1** (SPY alone was -4.29, this is -4.15 -- essentially
unchanged; QQQ alone was -2.88, this is -2.88 -- unchanged to the third
decimal). The portfolio-level SPY macro gate adds no measurable improvement
over each instrument gating on its own trend -- likely because SPY and QQQ
are highly correlated with each other already, so a SPY-derived gate carries
almost no additional information beyond what each instrument's own SMA200
already provides. This direction was not pursued further.

## All 4 directions complete: 17 candidates collected. See exploration_v2_candidates.json.


## Independent validation, check 1: objectively-drawn fresh ticker universe

- Sampling method: `bot/sp500_pool.py`'s fixed S&P 500 reference pool (278 tickers) minus the 56 tickers already tested in round 2 direction 2 (278 remaining), drawn via `np.random.default_rng(seed=20260703)` without replacement, n=56. Seed and full ticker list documented here for reproducibility.
- Drawn tickers: ['AMGN', 'AMP', 'AZO', 'BBWI', 'BDX', 'BLK', 'CFG', 'CHD', 'CNC', 'CTLT', 'CVS', 'DGX', 'DLTR', 'DOV', 'DPZ', 'ERIE', 'ETSY', 'EXPD', 'EXPE', 'FANG', 'GIS', 'GL', 'GRMN', 'HBAN', 'HII', 'HLT', 'HSY', 'HUM', 'INCY', 'IP', 'JBHT', 'JNPR', 'KHC', 'KMB', 'LHX', 'LOW', 'MAS', 'MU', 'NCLH', 'NDSN', 'NOC', 'NTRS', 'PNC', 'POOL', 'PWR', 'RCL', 'RF', 'ROL', 'SPGI', 'SWK', 'SYF', 'TFC', 'VLO', 'WHR', 'WM', 'WST']
- Fetched 56/56 tickers successfully. 28 produced a surviving anti-overfitting plateau. **2 of 56 passed all 3 viability criteria** (original universe: 7 of 56 = 12.5%; this universe: 2/56 = 3.6%).
    PASS: FANG Sharpe=1.182 PF=3.636 MaxDD=1.9% trades=12
    PASS: GRMN Sharpe=0.737 PF=1.961 MaxDD=4.6% trades=28
- **Combined equal-weight portfolio of 2 passers on the FRESH universe: Sharpe=1.140** (original universe combined Sharpe was 1.896) MaxDD=2.3% MC_DD95=12.1% PF=2.310 trades=40
- **Gate check**: combined Sharpe 1.140 vs required > 0.5 AND > 0.948 (50% of original 1.896). **PASSES the pre-declared gate.**

## Independent validation, check 2: genuinely unseen time window (pure holdout)

- Holdout window: 2020-07-27 to 2022-06-03 (~22.7 months) -- this is BEFORE the earliest train_start (2022-06-03) used anywhere in the original walk-forward, so it was never seen as train OR test data in any prior step.
- Using the 7 original winning tickers' EXACT already-selected parameters, no re-fitting: {'META': {'sma_period': 20, 'std_dev_mult': 2.0, 'trend_sma_period': 200}, 'NVDA': {'sma_period': 20, 'std_dev_mult': 1.5, 'trend_sma_period': 200}, 'MRK': {'sma_period': 20, 'std_dev_mult': 2.0, 'trend_sma_period': 200}, 'COP': {'sma_period': 20, 'std_dev_mult': 1.5, 'trend_sma_period': 200}, 'NKE': {'sma_period': 20, 'std_dev_mult': 1.5, 'trend_sma_period': 200}, 'PYPL': {'sma_period': 20, 'std_dev_mult': 2.0, 'trend_sma_period': 200}, 'GE': {'sma_period': 20, 'std_dev_mult': 1.5, 'trend_sma_period': 200}}
    META: Sharpe=-0.921 PF=0.000 MaxDD=4.1% trades=2
    NVDA: Sharpe=-1.003 PF=0.173 MaxDD=35.2% trades=16
    MRK: Sharpe=nan PF=nan MaxDD=0.0% trades=0
    COP: Sharpe=0.807 PF=1.737 MaxDD=2.5% trades=12
    NKE: Sharpe=-0.639 PF=0.599 MaxDD=8.7% trades=14
    PYPL: Sharpe=-0.559 PF=0.247 MaxDD=4.0% trades=5
    GE: Sharpe=0.622 PF=1.509 MaxDD=3.9% trades=13
- **Combined portfolio on the unseen 2020-2022 holdout: Sharpe=-0.969** (original 2022-2026 combined Sharpe was 1.896) MaxDD=7.3% PF=0.557 trades=62

## Step 2A: cautious universe expansion (unchanged criteria, no re-optimization)

- Universe: all 278 tickers in `bot/sp500_pool.py` minus the 112 already tested (56 original + 56 independent-validation draw) = 222 new tickers. Same exact grid ({sma_period:20, std_dev_mult in [1.5,2.0], trend_sma_period:200}), same 30% anti-overfitting rule, same viability bar (Sharpe>=0.5, MC_DD95<=20%, edge survives 0.15% slippage). No parameter changes of any kind from what was already validated.
- Fetched 222/222 tickers. 130 produced a surviving plateau. **29 of 222 new tickers passed all 3 criteria.**
    PASS: C Sharpe=1.513 PF=4.872 MaxDD=1.5% trades=14
    PASS: KLAC Sharpe=1.276 PF=3.639 MaxDD=3.6% trades=21
    PASS: JCI Sharpe=1.247 PF=6.316 MaxDD=3.6% trades=17
    PASS: HOLX Sharpe=1.175 PF=3.994 MaxDD=3.7% trades=22
    PASS: HWM Sharpe=1.012 PF=5.678 MaxDD=2.5% trades=11
    PASS: WMB Sharpe=0.992 PF=2.827 MaxDD=4.1% trades=19
    PASS: PGR Sharpe=0.910 PF=2.559 MaxDD=3.6% trades=21
    PASS: NOW Sharpe=0.907 PF=2.655 MaxDD=6.1% trades=28
    PASS: GD Sharpe=0.890 PF=2.585 MaxDD=4.0% trades=23
    PASS: WBA Sharpe=0.888 PF=2.216 MaxDD=5.7% trades=16
    PASS: RMD Sharpe=0.887 PF=3.898 MaxDD=2.4% trades=13
    PASS: LLY Sharpe=0.872 PF=2.229 MaxDD=4.5% trades=25
    PASS: DVA Sharpe=0.823 PF=2.323 MaxDD=5.3% trades=15
    PASS: TRV Sharpe=0.805 PF=2.221 MaxDD=3.8% trades=17
    PASS: MPWR Sharpe=0.801 PF=1.986 MaxDD=6.1% trades=27
    PASS: ROP Sharpe=0.793 PF=3.567 MaxDD=2.7% trades=8
    PASS: ORLY Sharpe=0.780 PF=2.058 MaxDD=7.9% trades=27
    PASS: STT Sharpe=0.719 PF=1.751 MaxDD=9.0% trades=28
    PASS: ON Sharpe=0.718 PF=2.437 MaxDD=5.2% trades=16
    PASS: INTU Sharpe=0.692 PF=3.340 MaxDD=3.8% trades=9
    PASS: PHM Sharpe=0.679 PF=2.890 MaxDD=2.4% trades=13
    PASS: K Sharpe=0.645 PF=2.152 MaxDD=3.4% trades=12
    PASS: AMAT Sharpe=0.619 PF=1.974 MaxDD=6.1% trades=14
    PASS: CB Sharpe=0.611 PF=2.255 MaxDD=4.7% trades=11
    PASS: GILD Sharpe=0.576 PF=1.832 MaxDD=3.5% trades=18
    PASS: PSX Sharpe=0.558 PF=2.072 MaxDD=4.3% trades=13
    PASS: LRCX Sharpe=0.553 PF=2.983 MaxDD=7.4% trades=24
    PASS: ODFL Sharpe=0.552 PF=1.583 MaxDD=7.3% trades=26
    PASS: AFL Sharpe=0.509 PF=1.977 MaxDD=3.7% trades=11

- **Cumulative multiple-testing accounting**: 334 total tickers tested across all rounds (56 original Direction 2 + 56 independent validation draw + 222 expansion), 38 total passed all 3 viability criteria at any point (11.4% overall hit rate). This is the honest denominator for judging whether ANY individual pass rate is more than chance would produce.
- **Combined portfolio of this batch's 29 new passers alone: Sharpe=3.184** MaxDD=0.4% MC_DD95=15.1% PF=2.532 trades=519
