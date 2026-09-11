"""
bot_options_putwrite/putwrite_backtest.py
=============================================
Systematic monthly ATM put-writing strategy (replicates CBOE PUT Index logic),
validated against published academic benchmark, with REAL transaction cost
calibration via Alpaca's live options chain.

METHODOLOGY (already validated in exploration, see notes below):
  1. Long-history backtest (1993-2026, SPY + VIX from yfinance) using
     Black-Scholes to reconstruct monthly ATM put premiums, since Alpaca
     only has real options data back to Feb 2024.
  2. VALIDATED: gross Sharpe of this reconstruction (0.76, 1993-2018) landed
     close to the published CBOE PUT Index gross Sharpe (0.65, Bondarenko
     2019) — confirms the reconstruction methodology is sound before trusting
     it further. Max drawdown also matched closely (31.9% vs 32.7% published).
  3. CRITICAL OPEN QUESTION: the strategy's edge is thin and sensitive to
     transaction costs. A guessed haircut of 0.5%/month erases essentially
     all of the edge (Sharpe drops to ~0.06); 1%/month makes it negative.
     This guessed haircut must be REPLACED with a real, measured bid-ask
     spread from Alpaca's actual options chain data (Feb 2024 onward is
     available) before any conclusion can be trusted.

WHAT THIS SCRIPT DOES:
  Step A. Re-run the long-history Black-Scholes reconstruction (gross, no
          costs) — sanity-check it still lands near Sharpe ~0.65-0.80 vs
          the published benchmark. If it doesn't, STOP — something changed
          or was miscoded; do not proceed to Step B.
  Step B. Pull real historical/live SPY ATM monthly put option quotes from
          Alpaca (OptionHistoricalDataClient) for the available window
          (Feb 2024 onward). Measure the actual bid-ask spread as a % of
          the mid price, across as many monthly cycles as available.
  Step C. Apply this MEASURED spread (not a guess) as the haircut to the
          full long-history backtest from Step A. This produces the final,
          realistic verdict.
  Step D. Also cross check: for the Feb-2024-onward window specifically,
          compare the Black-Scholes theoretical premium against the REAL
          traded premium, to sanity-check the reconstruction isn't
          systematically over- or under-pricing options relative to reality.

STOPPING RULES (same discipline as PEAD / value-quality / intraday ORB):
  - Do NOT wire into bot/main.py or paper/live trading regardless of result,
    without explicit user approval.
  - If final (cost-adjusted) Sharpe < 0.5, or MC_DD95 (block-bootstrap,
    monthly returns) > 20%, or Step A's sanity check fails to reproduce the
    published benchmark within a reasonable margin -> document as REJECTED
    in exploration_log_options.md and STOP. Do not keep adjusting the model
    (e.g. tweaking the risk-free rate, the vol proxy, the moneyness) to force
    a pass — see lessons_learned.md piège #4 (argmax selection bias).
  - Explicitly flag if Step D reveals the Black-Scholes reconstruction is
    biased vs real traded prices — this affects how much to trust Step A/C.
"""

import calendar
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

import config
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame

# VERIFIED LIVE against the real alpaca-py options API (v0.43.4) before
# writing Step B/D below -- do not re-assume, this took real investigation:
#   - get_option_bars / get_option_trades: real HISTORICAL data back to (at
#     least) Feb 2024 for a real expired contract (tested directly), but
#     price/trade data only -- NO bid/ask field anywhere in either schema.
#   - get_option_latest_quote: returns {} for an EXPIRED contract -- only
#     works for currently-listed contracts.
#   - get_option_chain: returns REAL, CURRENT bid/ask for every currently
#     listed contract (confirmed: 402 SPY put contracts, real bid/ask each).
#   - There is no "OptionQuotesRequest" (historical option quotes) at all in
#     alpaca-py -- confirmed by listing every class in alpaca.data.requests;
#     a `StockQuotesRequest` exists for equities, no options equivalent.
# CONCLUSION: Alpaca's options API has NO historical bid/ask quote data of
# any kind, for any date, at any tier -- not a subscription gate, an actual
# absence of the endpoint. The literal ask ("measure the real spread at
# entry, for each monthly cycle since Feb 2024") is therefore not
# achievable AS LITERALLY SPECIFIED -- there is no historical spread to
# pull, for Feb 2024 or any other past date. What IS real and measurable:
# TODAY's live bid/ask on real, currently-listed near-ATM SPY monthly puts.
# Step B below measures that -- a genuine, live, measured number -- and
# applies it uniformly across history, which is itself an explicit,
# flagged assumption (recent spreads are representative of past ones), not
# a fabricated haircut. See exploration_log_options.md for the full
# verification trail and this distinction, which matters for how much to
# trust the final verdict.

RISK_FREE_RATE = 0.03
DAYS_TO_EXPIRY = 30
N_MONTE_CARLO = 2000
ALPACA_OPTIONS_FEED_START = date(2024, 2, 1)  # earliest real Alpaca options data, per the module this file was given


def load_long_history():
    # auto_adjust=False is REQUIRED here: yfinance's default (True) returns
    # dividend-ADJUSTED close, not the real traded price -- verified directly
    # against Alpaca's real SPY close for 2024-02-01 (489.20 real vs 474.87
    # adjusted, a ~3% gap for a date this recent). Step A's normalized
    # (K=S0, pnl_pct=.../K) return series is largely scale-invariant to this,
    # but Step D's real-contract strike construction is NOT -- it silently
    # picked the wrong (non-ATM) real strike, which was the actual cause of
    # an apparent 150-224% Step D "bias" before this fix (see
    # exploration_log_options.md). Always real, traded prices for anything
    # that touches a real option contract.
    spy = yf.download("SPY", start="1993-01-01", progress=False, auto_adjust=False)
    vix = yf.download("^VIX", start="1993-01-01", progress=False, auto_adjust=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]
    return pd.DataFrame({"spy_close": spy["Close"], "vix_close": vix["Close"]}).dropna()


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def run_reconstruction(df, days_to_expiry=DAYS_TO_EXPIRY):
    df = df.copy()
    df["month"] = df.index.to_period("M")
    monthly_entries = df.groupby("month").head(1)
    trades = []
    for entry_date, row in monthly_entries.iterrows():
        S0, iv, K = row["spy_close"], row["vix_close"] / 100, row["spy_close"]
        T = days_to_expiry / 365
        premium = bs_put_price(S0, K, T, RISK_FREE_RATE, iv)
        target_expiry = entry_date + pd.Timedelta(days=days_to_expiry)
        future = df[df.index >= target_expiry]
        if future.empty:
            continue
        S_T = future.iloc[0]["spy_close"]
        payoff = -max(K - S_T, 0)
        pnl_pct = (premium + payoff) / K
        trades.append({"entry_date": entry_date, "S0": S0, "K": K, "S_T": S_T,
                        "iv": iv, "premium": premium, "pnl_pct": pnl_pct})
    return pd.DataFrame(trades)


def monte_carlo_drawdown(monthly_returns: pd.Series, n_sims=N_MONTE_CARLO, block_size=3):
    if len(monthly_returns) < 10:
        return np.nan
    rets = monthly_returns.values
    n = len(rets)
    max_dds = []
    for _ in range(n_sims):
        blocks = []
        while sum(len(b) for b in blocks) < n:
            start = np.random.randint(0, max(1, n - block_size))
            blocks.append(rets[start:start + block_size])
        sim = np.concatenate(blocks)[:n]
        equity = (1 + sim).cumprod()
        dd = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
        max_dds.append(dd.min())
    return -np.percentile(max_dds, 95)


def summarize(pnl_series, label, rf_annual=RISK_FREE_RATE, haircut=0.0):
    rets = pnl_series - haircut
    n = len(rets)
    rf_m = rf_annual / 12
    excess = rets.mean() - rf_m
    std = rets.std()
    sharpe = (excess / std) * np.sqrt(12) if std > 0 else np.nan
    equity = (1 + rets).cumprod()
    # Real historical MaxDD, not just the MC estimate -- lessons_learned.md
    # piège #6: MC (i.i.d.-ish block resampling) can understate real
    # clustered drawdown; always report both, never just one.
    running_peak = equity.cummax()
    real_max_dd = float(((running_peak - equity) / running_peak).max())
    mc_dd95 = monte_carlo_drawdown(rets)
    print(f"{label}: n={n} sharpe(excess)={sharpe:.2f} MC_DD95={mc_dd95:.1%} "
          f"real_MaxDD={real_max_dd:.1%} CAGR={(equity.iloc[-1])**(12/n)-1:.2%}")
    return {"label": label, "n": n, "sharpe": sharpe, "mc_dd95": mc_dd95, "real_max_dd": real_max_dd}


def step_a_sanity_check():
    print("=== STEP A: long-history reconstruction sanity check ===")
    df = load_long_history()
    trades = run_reconstruction(df)
    sub = trades[(trades["entry_date"] >= "1993-01-01") & (trades["entry_date"] <= "2018-12-31")]
    res = summarize(sub["pnl_pct"], "1993-2018 gross (no costs)")
    print("Published benchmark (Bondarenko 2019): Sharpe ~0.65, max_dd 32.7%")
    if not (0.4 <= res["sharpe"] <= 1.1):
        print("!!! SANITY CHECK FAILED: reconstruction Sharpe too far from "
              "published benchmark. STOP — do not proceed to Step B/C until "
              "this is understood and fixed.")
        return None, None
    print("Sanity check passed (within reasonable range of published benchmark).\n")
    return df, trades


def third_friday(year: int, month: int) -> date:
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])


def occ_put_symbol(underlying: str, expiry: date, strike: float) -> str:
    return f"{underlying}{expiry.strftime('%y%m%d')}P{int(round(strike * 1000)):08d}"


def nearest_monthly_expiry(entry_date: date, target_dte: int = DAYS_TO_EXPIRY) -> date:
    """Standard SPY monthly options expire the 3rd Friday. Picks whichever
    of (entry month, entry month + 1, entry month + 2)'s 3rd Friday lands
    closest to entry_date + target_dte."""
    candidates = []
    y, m = entry_date.year, entry_date.month
    for _ in range(3):
        candidates.append(third_friday(y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    target = entry_date + timedelta(days=target_dte)
    return min(candidates, key=lambda d: abs((d - target).days))


def step_b_measure_real_spread(client: OptionHistoricalDataClient) -> dict:
    """Measures the REAL, CURRENT bid-ask spread on real, currently-listed
    near-ATM SPY monthly puts (see module-level comment for why this is
    the honest substitute for "historical spread since Feb 2024" -- that
    literal ask is not achievable, verified, not assumed)."""
    print("=== STEP B: measure REAL bid-ask spread from Alpaca's live option chain ===")
    print("(Historical option quotes do not exist in Alpaca's API at any date -- verified; "
          "see module docstring. Measuring TODAY's real spread instead.)")
    now = datetime.now(timezone.utc)
    spreads = []
    for target_dte in (DAYS_TO_EXPIRY, DAYS_TO_EXPIRY + 30):  # front + next monthly-ish cycle
        expiry = nearest_monthly_expiry(now.date(), target_dte)
        req = OptionChainRequest(underlying_symbol="SPY", type="put",
                                 expiration_date=expiry)
        try:
            chain = client.get_option_chain(req)
        except Exception as e:
            print(f"  expiry {expiry}: chain fetch failed: {e}")
            continue
        rows = []
        for sym, snap in chain.items():
            q = snap.latest_quote
            bid, ask = getattr(q, "bid_price", None), getattr(q, "ask_price", None)
            if not bid or not ask or bid <= 0 or ask <= 0:
                continue
            strike = int(sym[-8:]) / 1000
            rows.append({"symbol": sym, "strike": strike, "bid": bid, "ask": ask,
                         "mid": (bid + ask) / 2, "spread_pct": (ask - bid) / ((ask + bid) / 2)})
        if not rows:
            continue
        chain_df = pd.DataFrame(rows)
        # underlying spot needed to find the ATM strike -- reuse the stock client pattern
        # already established elsewhere in this project (bot/data_client.py), quick direct call here.
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        stock_client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET)
        spot = stock_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=["SPY"]))["SPY"].price
        chain_df["dist"] = (chain_df["strike"] - spot).abs()
        near_atm = chain_df.nsmallest(3, "dist")
        print(f"  expiry {expiry} (spot={spot:.2f}): near-ATM spreads = "
              f"{near_atm['spread_pct'].round(4).tolist()}")
        spreads.extend(near_atm["spread_pct"].tolist())

    if not spreads:
        print("  Could not measure any real spread -- chain empty or all quotes invalid.")
        return {"measured_spread_pct": None, "n_contracts": 0}

    measured = float(np.mean(spreads))
    print(f"  MEASURED real spread (mean across {len(spreads)} near-ATM contracts, "
          f"2 expiries, today): {measured:.4%} of mid price")
    return {"measured_spread_pct": measured, "n_contracts": len(spreads), "raw_spreads": spreads}


def step_d_bs_vs_real(client: OptionHistoricalDataClient, trades: pd.DataFrame) -> pd.DataFrame:
    """For each monthly entry since Alpaca's real options data starts
    (Feb 2024), constructs the real OCC symbol for the nearest standard
    monthly expiry and compares its REAL historical premium (from
    get_option_bars, entry-day open) against the Black-Scholes theoretical
    premium already computed in Step A -- checks the reconstruction isn't
    systematically biased vs reality, per the module's stopping rules."""
    print("=== STEP D: Black-Scholes reconstruction vs REAL traded premiums (Feb 2024+) ===")
    recent = trades[trades["entry_date"].dt.date >= ALPACA_OPTIONS_FEED_START].copy()
    rows = []
    for _, row in recent.iterrows():
        entry_date = row["entry_date"].date()
        expiry = nearest_monthly_expiry(entry_date)
        real_dte = (expiry - entry_date).days
        strike = round(row["S0"])
        sym = occ_put_symbol("SPY", expiry, strike)
        req = OptionBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame.Day,
                                start=datetime.combine(entry_date, datetime.min.time(), tzinfo=timezone.utc),
                                end=datetime.combine(entry_date + timedelta(days=5), datetime.min.time(), tzinfo=timezone.utc))
        try:
            bars = client.get_option_bars(req).df
        except Exception:
            bars = pd.DataFrame()
        if bars.empty:
            continue
        real_premium = float(bars.iloc[0]["open"])
        # FAIR comparison: standard monthly (3rd-Friday) expiries closest to
        # entry+30d are often only 14-20 real days out, not 30 (checked
        # directly: e.g. 2024-03-01 entry -> nearest monthly expiry is only
        # 14 real days later). Comparing the STRATEGY's fixed-T=30 premium
        # against a real ~15-20 DTE contract's premium is comparing two
        # different maturities, not a fair test of the BS/VIX model itself
        # -- a shorter-dated option is mechanically cheaper regardless of
        # any pricing-model bias. Recomputing the BS premium at the REAL
        # contract's actual DTE isolates the model comparison from this
        # maturity-mismatch artifact.
        bs_premium_matched_T = bs_put_price(row["S0"], strike, real_dte / 365, RISK_FREE_RATE, row["iv"])
        rows.append({"entry_date": entry_date, "symbol": sym, "real_dte": real_dte,
                     "bs_premium_fixed_T30": row["premium"], "bs_premium_matched_T": bs_premium_matched_T,
                     "real_premium": real_premium,
                     "diff_pct_fixed_T30": (row["premium"] - real_premium) / real_premium,
                     "diff_pct": (bs_premium_matched_T - real_premium) / real_premium})

    result = pd.DataFrame(rows)
    if result.empty:
        print("  No real contracts resolved -- cannot compare (see log for why).")
        return result
    print(f"  Resolved {len(result)}/{len(recent)} monthly entries to real contracts.")
    print(f"  Mean (BS - real)/real: {result['diff_pct'].mean():.2%}  "
          f"median: {result['diff_pct'].median():.2%}  std: {result['diff_pct'].std():.2%}")
    if abs(result["diff_pct"].mean()) > 0.15:
        print("  !!! FLAG: Black-Scholes reconstruction appears systematically biased "
              "(>15% mean deviation) vs real traded premiums -- Step A/C's numbers should "
              "be read with this bias in mind, not trusted at face value.")
    return result


def step_c_final_verdict(trades: pd.DataFrame, step_b_result: dict) -> dict:
    print("=== STEP C: final cost-adjusted verdict (full history) ===")
    measured = step_b_result.get("measured_spread_pct")
    if measured is None:
        print("Cannot compute final verdict — Step B produced no measurement.")
        return {"rejected": None, "reason": "no measured spread"}

    # Mechanically accurate haircut: the strategy (run_reconstruction) sells
    # to open and holds to expiry -- ONE transaction, not a round trip -- so
    # the realistic cost is losing half the spread (mid -> bid on the sell),
    # not the full spread. Reporting the full-spread number too as an
    # explicit, separately labeled conservative sensitivity check, not as
    # the primary verdict -- applying the full round-trip cost to a
    # single-transaction strategy would overstate costs relative to how it
    # actually trades.
    half_spread = measured / 2
    res_primary = summarize(trades["pnl_pct"], f"Full history, real-spread/2-adjusted ({half_spread:.3%}/mo)",
                            haircut=half_spread)
    res_conservative = summarize(trades["pnl_pct"], f"Full history, full-real-spread-adjusted ({measured:.3%}/mo, "
                                 f"conservative sensitivity)", haircut=measured)

    rejected = (res_primary["sharpe"] != res_primary["sharpe"] or res_primary["sharpe"] < 0.5
               or res_primary["mc_dd95"] > 0.20)
    if rejected:
        print("VERDICT: REJECTED per stopping rules.")
    else:
        print("VERDICT: PASSES formal thresholds — still requires human review "
              "before any paper/live trading, per project rules.")
    return {"rejected": rejected, "primary": res_primary, "conservative": res_conservative,
           "measured_spread_pct": measured}


def log_md(text: str) -> None:
    with open("exploration_log_options.md", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    log_md(f"\n## Run -- {datetime.now(timezone.utc).isoformat()}\n")

    df, trades = step_a_sanity_check()
    if df is None:
        log_md("**STOPPED at Step A: sanity check failed to reproduce the published benchmark. "
              "Not proceeding to Step B/C.**")
        return None
    sub = trades[(trades["entry_date"] >= "1993-01-01") & (trades["entry_date"] <= "2018-12-31")]
    is_res = summarize(sub["pnl_pct"], "throwaway")  # recompute cleanly for logging (already printed above)
    log_md(f"- **Step A (sanity check)**: 1993-2018 gross reconstruction Sharpe={is_res['sharpe']:.3f} "
          f"(published Bondarenko 2019: ~0.65), real MaxDD={is_res['real_max_dd']:.1%} "
          f"(published: 32.7%) -- reproduces within tolerance, proceeding.")

    client = OptionHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET)

    log_md(f"\n- **Step B**: Alpaca's options API has NO historical bid/ask quote endpoint at any "
          f"date (verified live -- get_option_bars/get_option_trades give price/trade data only, "
          f"no bid/ask field; get_option_latest_quote returns empty for expired contracts; no "
          f"OptionQuotesRequest class exists at all, unlike StockQuotesRequest for equities). The "
          f"literal ask (spread since Feb 2024, cycle by cycle) is not achievable with this API. "
          f"Measuring instead: REAL, CURRENT bid/ask on real, live SPY near-ATM monthly puts today.")
    step_b_result = step_b_measure_real_spread(client)
    log_md(f"- Measured real spread: {step_b_result.get('measured_spread_pct')} "
          f"(mean over {step_b_result.get('n_contracts', 0)} near-ATM contracts, 2 expiries, "
          f"as of {datetime.now(timezone.utc).date()})")

    step_d_result = step_d_bs_vs_real(client, trades)
    if not step_d_result.empty:
        log_md(f"\n- **Step D**: resolved {len(step_d_result)}/{len(trades[trades['entry_date'].dt.date >= ALPACA_OPTIONS_FEED_START])} "
              f"monthly entries since Feb 2024 to real contracts. Mean (BS-real)/real = "
              f"{step_d_result['diff_pct'].mean():.2%}, median={step_d_result['diff_pct'].median():.2%}, "
              f"std={step_d_result['diff_pct'].std():.2%}.")
        if abs(step_d_result["diff_pct"].mean()) > 0.15:
            log_md(f"  **FLAG: >15% mean bias between BS reconstruction and real premiums -- "
                  f"Step A/C should be read with this in mind.**")
    else:
        log_md("- **Step D**: could not resolve any real contracts for comparison (see run output).")

    if step_b_result.get("measured_spread_pct") is None:
        log_md("\n**STOPPED: Step B produced no usable spread measurement. Cannot compute Step C.**")
        return None

    verdict = step_c_final_verdict(trades, step_b_result)
    log_md(f"\n### STEP C -- final verdict (primary: half-spread, mechanically matches "
          f"sell-to-open-and-hold-to-expiry; conservative: full spread as sensitivity)\n")
    p, c = verdict["primary"], verdict["conservative"]
    log_md(f"- Primary (half-spread {verdict['measured_spread_pct']/2:.3%}/mo): "
          f"Sharpe={p['sharpe']:.3f}, MC_DD95={p['mc_dd95']:.1%}, real_MaxDD={p['real_max_dd']:.1%}, n={p['n']}")
    log_md(f"- Conservative (full spread {verdict['measured_spread_pct']:.3%}/mo): "
          f"Sharpe={c['sharpe']:.3f}, MC_DD95={c['mc_dd95']:.1%}, real_MaxDD={c['real_max_dd']:.1%}, n={c['n']}")
    log_md(f"\n**VERDICT: {'REJECTED' if verdict['rejected'] else 'NOT REJECTED'}** per the stop rules "
          f"(Sharpe<0.5 or MC_DD95>20%), evaluated on the primary (half-spread) haircut.")

    return verdict


if __name__ == "__main__":
    main()
