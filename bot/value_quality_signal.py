"""
Value + quality composite factor signal. Two layers:

1. Per-ticker point-in-time metrics (`build_ticker_metrics`): raw ratios
   derived from fundamentals + price, dated by `available_date` (see
   bot/fundamentals_data.py's documented lag-proxy assumption) -- no
   cross-sectional comparison yet.
2. Cross-sectional composite score (`build_composite_cross_section`): at
   ONE rebalance date, takes each ticker's most recently AVAILABLE
   quarter as of that date, sector-ranks each raw metric
   (bot/factor_normalization.py), and combines into value/quality/
   composite scores.

Weighting: EQUAL weights between the value and quality pillars, and EQUAL
weights within each pillar's sub-metrics -- the plan's default, since no
documented reason to deviate was identified. If that changes, the reason
must be written here, not just in a commit message.

Metrics used (all oriented so a HIGHER final percentile is more
attractive -- see bot/factor_normalization.py):
  Value:   price_to_book (lower better), price_to_earnings (lower better,
           trailing-4Q EPS)
  Quality: net_margin_ttm (higher better), debt_to_equity (lower better),
           earnings_stability (higher better -- defined as the NEGATIVE of
           the trailing-8-quarter coefficient of variation of quarterly
           EPS, i.e. less volatile earnings score higher)

Deliberately excluded: ROE/ROA. For highly-levered sectors (financials),
ROE is mechanically inflated by leverage -- combining it with
debt_to_equity would double-count the same underlying leverage signal in
two metrics pointing the same direction, silently overweighting leverage
in the composite. Revisit only with a documented reason.
"""
import logging

import numpy as np
import pandas as pd

from bot.factor_normalization import sector_relative_percentile

logger = logging.getLogger(__name__)

VALUE_METRICS = {"price_to_book": False, "price_to_earnings": False}  # False = lower is better
QUALITY_METRICS = {"net_margin_ttm": True, "debt_to_equity": False, "earnings_stability": True}
EARNINGS_STABILITY_LOOKBACK_QUARTERS = 8


def build_ticker_metrics(symbol: str, pit_fundamentals: pd.DataFrame, price_bars: pd.DataFrame) -> pd.DataFrame:
    """`pit_fundamentals`: the DataFrame returned by
    `SECEdgarFundamentalsClient.get_quarterly_fundamentals` for this
    symbol (indexed by fiscal_date_ending, has available_date -- the REAL
    SEC filing date -- + the raw
    balance-sheet/income-statement columns). `price_bars`: daily OHLCV
    for this symbol (needs a 'close' column, tz-aware UTC index).
    Returns one row per fiscal quarter with available_date, the price
    used, and all derived ratios -- still just ONE ticker's own history,
    no cross-sectional ranking yet."""
    if pit_fundamentals.empty or price_bars.empty:
        return pd.DataFrame()

    df = pit_fundamentals.sort_index().copy()
    df["quarterly_eps"] = df["net_income"] / df["shares_outstanding"]
    # trailing-4Q EPS for P/E -- a single quarter's EPS is too noisy/seasonal
    # for a "how expensive is this stock" read; TTM is the market-standard unit.
    df["eps_ttm"] = df["quarterly_eps"].rolling(4, min_periods=4).sum()
    # coefficient of variation of trailing-8Q EPS, NEGATED so higher = more
    # stable = better (consistent with this module's "higher percentile is
    # always more attractive" convention). abs() in the denominator so a
    # negative mean EPS doesn't flip the sign of the ratio itself.
    roll = df["quarterly_eps"].rolling(EARNINGS_STABILITY_LOOKBACK_QUARTERS, min_periods=EARNINGS_STABILITY_LOOKBACK_QUARTERS)
    cv = roll.std() / roll.mean().abs()
    df["earnings_stability"] = -cv

    prices = price_bars.sort_index()
    rows = []
    for fiscal_date, row in df.iterrows():
        avail = row["available_date"]
        future = prices[prices.index >= avail]
        if future.empty:
            continue
        price = float(future["close"].iloc[0])
        price_date = future.index[0]

        bvps = row["total_shareholder_equity"] / row["shares_outstanding"] if row["shares_outstanding"] else np.nan
        p_to_b = price / bvps if bvps and bvps == bvps and bvps > 0 else np.nan
        p_to_e = price / row["eps_ttm"] if row["eps_ttm"] == row["eps_ttm"] and row["eps_ttm"] > 0 else np.nan
        # a negative/zero TTM EPS makes P/E undefined (not "very cheap") --
        # left as NaN rather than a negative or infinite ratio, exactly the
        # kind of degenerate value the sector-rank fallback (not a z-score)
        # was chosen partly to be robust against; see lessons_learned.md.
        net_margin = row["net_income"] / row["total_revenue"] if row["total_revenue"] else np.nan
        debt_to_equity = row["total_debt"] / row["total_shareholder_equity"] if row["total_shareholder_equity"] else np.nan

        rows.append({
            "symbol": symbol, "fiscal_date_ending": fiscal_date, "available_date": avail,
            "price_used_date": price_date, "price": price,
            "price_to_book": p_to_b, "price_to_earnings": p_to_e,
            "net_margin_ttm": net_margin, "debt_to_equity": debt_to_equity,
            "earnings_stability": row["earnings_stability"],
        })
    return pd.DataFrame(rows)


def monthly_rebalance_dates(start: pd.Timestamp, end: pd.Timestamp) -> list:
    # start/end are already tz-aware (UTC) here -- passing tz="UTC" AGAIN on
    # top of already-tz-aware endpoints trips a pandas assertion ("Inferred
    # time zone not equal to passed time zone") on some pandas versions.
    # Only pass tz= when the inputs are naive.
    if start.tzinfo is not None:
        return list(pd.date_range(start, end, freq="ME"))
    return list(pd.date_range(start, end, freq="ME", tz="UTC"))


def as_of_cross_section(ticker_metrics: dict, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """For each ticker, its most recent quarter with available_date <=
    rebalance_date (point-in-time -- no look-ahead). Tickers with no
    qualifying quarter yet are dropped for this date, not filled."""
    rows = []
    for symbol, panel in ticker_metrics.items():
        eligible = panel[panel["available_date"] <= rebalance_date]
        if eligible.empty:
            continue
        rows.append(eligible.iloc[-1])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["rebalance_date"] = rebalance_date
    return out


def build_composite_cross_section(cross_section: pd.DataFrame, sector_by_symbol: dict,
                                   min_group_size: int = 8) -> pd.DataFrame:
    """Attaches sector/industry, sector-ranks every metric in
    VALUE_METRICS/QUALITY_METRICS, and combines into value_score,
    quality_score, composite_score (all 0-1, equal-weighted, higher=better).
    `sector_by_symbol`: {symbol -> {'sector':..., 'industry':...}}."""
    df = cross_section.copy()
    df["sector"] = df["symbol"].map(lambda s: sector_by_symbol.get(s, {}).get("sector"))
    df["industry"] = df["symbol"].map(lambda s: sector_by_symbol.get(s, {}).get("industry"))
    missing_sector = df["sector"].isna() | df["industry"].isna()
    if missing_sector.any():
        logger.warning("Dropping %d/%d tickers with no sector/industry classification: %s",
                       missing_sector.sum(), len(df), df.loc[missing_sector, "symbol"].tolist())
        df = df[~missing_sector].reset_index(drop=True)
    if df.empty:
        return df

    percentile_cols = []
    for metric, higher_is_better in {**VALUE_METRICS, **QUALITY_METRICS}.items():
        ranked = sector_relative_percentile(df, metric, min_group_size=min_group_size,
                                            higher_is_better=higher_is_better)
        col = f"{metric}_pct"
        df[col] = ranked["percentile"]
        percentile_cols.append(col)

    df["value_score"] = df[[f"{m}_pct" for m in VALUE_METRICS]].mean(axis=1, skipna=True)
    df["quality_score"] = df[[f"{m}_pct" for m in QUALITY_METRICS]].mean(axis=1, skipna=True)
    df["composite_score"] = df[["value_score", "quality_score"]].mean(axis=1, skipna=True)
    return df


if __name__ == "__main__":
    # Mechanics self-test on SYNTHETIC data -- proves the pipeline glues
    # together correctly (per-ticker metrics -> as-of cross-section ->
    # sector ranking -> composite) before real sector data is available
    # (blocked on Alpha Vantage's quota as of this writing). NOT a
    # backtest result. See exploration_log_value_quality.md.
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-31", "2020-10-31", freq="QE", tz="UTC")
    tickers = [f"T{i}" for i in range(20)]
    sector_by_symbol = {t: {"sector": "Sector" + str(i % 2), "industry": "Industry" + str(i % 4)}
                        for i, t in enumerate(tickers)}

    panels = {}
    for t in tickers:
        n = len(dates)
        panels[t] = pd.DataFrame({
            "symbol": t, "fiscal_date_ending": dates, "available_date": dates + pd.Timedelta(days=35),
            "price_used_date": dates, "price": rng.uniform(20, 200, n),
            "price_to_book": rng.uniform(0.5, 8, n), "price_to_earnings": rng.uniform(5, 40, n),
            "net_margin_ttm": rng.uniform(0.02, 0.35, n), "debt_to_equity": rng.uniform(0.1, 3.5, n),
            "earnings_stability": rng.uniform(-1, 0, n),
        })

    cs = as_of_cross_section(panels, pd.Timestamp("2020-10-31", tz="UTC"))
    scored = build_composite_cross_section(cs, sector_by_symbol, min_group_size=8)
    print(scored[["symbol", "sector", "industry", "value_score", "quality_score", "composite_score"]]
         .sort_values("composite_score", ascending=False).to_string(index=False))
