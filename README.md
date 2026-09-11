# Trading Bot Research

A systematic trading strategy research project — four independent strategies investigated with the same rigorous validation discipline, all four rejected. This repo is the full paper trail: what was tried, what broke, what was found along the way, and why each strategy ultimately didn't clear the bar.

**Nothing here trades live money. `bot/main.py` is hard-disabled by design** (raises on startup) until a strategy actually passes validation — none has yet.

## Why publish a project that "didn't work"

A negative result, reached honestly and rigorously, is a real research outcome — not a failure to hide. The value here isn't a winning strategy; it's the methodology: catching your own bugs before they inflate a Sharpe ratio, testing across market regimes a strategy wasn't built on, and being willing to reject a result that *almost* passes rather than moving the goalposts.

## Strategies investigated

| Strategy | Status | Why it failed |
|---|---|---|
| **Mean-reversion / momentum / trend-following** (`bot/strategies/`) | Rejected | Out-of-sample Sharpe -3.46 / -2.51 across 12 walk-forward windows; max drawdown 97% / 43%; no transaction-cost scenario survives |
| **PEAD** (post-earnings-announcement drift, `bot/pead_backtest.py`) | Retired | Base signal had genuine edge but unstable parameters (in-sample parameter selection performed *worse than random* out-of-sample); a mid-hold regime-exit fix cleared all viability criteria on one recession (2021-2022) but failed decisively on a second, structurally different one (the fast Q4-2018 correction) — exactly the generalization test it needed to survive |
| **ORB-fade** (15-min opening-range fade, `bot_intraday_15min/`) | Rejected | OOS Sharpe < 0.5, Monte-Carlo drawdown > 20%, and a sign flip between in-sample and out-of-sample on every take-profit variant tested — on both a recent 2-year window and the Q4-2018 correction |
| **Monthly ATM put-write** (`bot_options_putwrite/`) | Rejected | Reconstructed edge matched a published academic benchmark almost exactly (Sharpe 0.67 vs. 0.65), but real bid-ask spreads measured from Alpaca's live options chain erode it below the 0.5 Sharpe bar even in the most favorable of three independent measurements |
| **Value + quality factor** (`bot/value_quality_signal.py`) | Does not pass | Sharpe 0.56 over the full 2016-2026 span, but negative or near-zero in *every one* of three regimes tested individually (Q4-2018, COVID crash, last 18 months) — the aggregate number was doing the hiding |

Full write-ups, numbers, and dead ends for each: `exploration_log_pead.md`, `exploration_log_intraday_orb.md`, `exploration_log_options.md`, `exploration_log_value_quality.md`, plus `comparison_report.md` and `exploration_log_v2.md` for the earliest round.

## The methodology (`lessons_learned.md`)

The most reused file in this repo. A running checklist of structural pitfalls hit more than once and the checks that catch them before they produce a misleading number:

- **No aggregate exposure cap = hidden tail risk.** Per-trade sizing that looks fine trade-by-trade can still let a book stack multiples of capital when signals cluster in time.
- **Entry-only regime filters usually aren't causal.** An aggregate metric can improve purely from shrinking the sample — verified here by reconstructing *which* trades a filter actually removed, not just trusting the summary stats.
- **A "pass" on one recession proves nothing about the next one.** A lagging trend filter that fixed PEAD's drawdown in a slow 2021-2022 bear market actively destroyed returns in the fast, V-shaped Q4-2018 correction. Multi-regime testing isn't optional rigor, it's the whole point.
- **In-sample parameter selection can be worse than random**, not just noisy — measured directly on PEAD's walk-forward: the in-sample-optimal parameter's out-of-sample rank averaged *worse* than picking at random.
- **Implausibly good numbers get investigated, not reported.** A Sharpe of 8+ turned out to be a linear-interpolation mark-to-market bug inflating it 10-50x. Found by refusing to believe the number, not by luck.
- **Monte Carlo and real historical drawdown can disagree sharply** when trades cluster in time — always compare both, never trust resampling alone.
- **Data source limits get verified live, never assumed from documentation** — this is how a stale IEX-feed assumption, a mispriced options-quote API, a wrong ticker→CIK mapping in SEC's own reference data, and a yfinance default that silently returns dividend-adjusted prices instead of real traded prices all got caught before they corrupted a result.

## Data sources

- **Alpaca** — price bars (equities, options quotes/bars, crypto). Free tier's IEX feed has much shallower history than SIP for this account (individual-equity minute bars: ~2021 vs. SIP's 2016); documented and worked around, not assumed.
- **SEC EDGAR** (`data.sec.gov`) — quarterly fundamentals via XBRL `companyfacts`, free, no key, no rate limit that matters at this scale. Replaced a paid/quota-limited path entirely; also gives the *real* filing date for each figure, which is a precision improvement over the date-proxy it replaced, not just a speed one.
- **Alpha Vantage** — earnings surprise history (PEAD) and a sector-classification fallback for the handful of tickers outside the current S&P 500 (free tier, 25 requests/day — the actual bottleneck behind most of the pacing decisions visible in the commit history).
- **Wikipedia's S&P 500 constituent table** — GICS sector/sub-industry for the whole universe in one free request instead of one quota-limited call per ticker.

## Structure

```
bot/                        Shared engine: backtesting, data clients, position/risk logic, strategy implementations
bot_intraday_15min/         ORB-fade intraday strategy (rejected)
bot_options_putwrite/       Monthly put-write strategy (rejected)
tests/                      Unit tests for the shared engine
exploration_log_*.md        Full research log per strategy track, in chronological order
lessons_learned.md          Cross-project methodology checklist
comparison_report.md        Strategy A vs. B head-to-head (the earliest round)
*_cache/, data_cache*/      Cached API responses — re-running any script never re-spends a quota on data already fetched
```

## Running it

```
pip install -r requirements.txt
cp .env.example .env   # fill in your own API keys — never commit .env
```

Every `run_*.py` / `collect_*.py` / `drip_feed_*.py` script at the repo root is a self-contained entry point for one piece of research and is safe to re-run — each one picks up wherever its cache left off rather than re-fetching or re-computing from scratch.

## License

Research code, published for its methodology. No warranty, no investment advice, not connected to any live trading account.
