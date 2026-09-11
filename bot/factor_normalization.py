"""
Sector-relative percentile-rank normalization for value/quality factor
scores -- NOT the composite score itself (that's a separate, later step).

Design agreed with the user before implementation:
  - Percentile rank within group, not z-score -- more robust to the fat
    tails / degenerate values (near-zero or negative book value or
    earnings) these ratios can produce. See lessons_learned.md.
  - Computed independently at EACH rebalance date, using only data with
    `available_date <= rebalance_date` (point-in-time) -- never a
    full-backtest-period group statistic, which would leak future
    information into the normalization step itself.
  - Falls back from the narrow `industry` grouping to the broader `sector`
    grouping when the industry group has fewer than `min_group_size`
    members with valid data at that rebalance date -- a small group gives
    an unstable/meaningless rank.

This module does NOT decide "value" vs "quality" or combine metrics --
`higher_is_better` is passed in per metric by the caller (e.g. False for
price-to-book/price-to-earnings, True for margin/ROE, False for
debt-to-equity) when that step is built.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MIN_GROUP_SIZE = 8


def sector_relative_percentile(cross_section: pd.DataFrame, value_col: str, industry_col: str = "industry",
                                sector_col: str = "sector", min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
                                higher_is_better: bool = True) -> pd.DataFrame:
    """One point-in-time cross-section (one row per ticker, already
    filtered by the caller to `available_date <= this rebalance date`,
    using each ticker's most recent qualifying quarter as of that date).
    Returns the input with 3 columns added: `group_used` ('industry' or
    'sector'), `group_label`, `group_size`, `percentile` (0-1, oriented so
    a HIGHER percentile is always "more attractive" per `higher_is_better`,
    regardless of the metric's own natural direction)."""
    df = cross_section.copy()
    valid = df[value_col].notna()

    industry_sizes = df.loc[valid].groupby(industry_col)[value_col].transform("size")
    use_sector_fallback = valid & (industry_sizes < min_group_size)

    df["group_used"] = pd.Series("industry", index=df.index)
    df.loc[use_sector_fallback, "group_used"] = "sector"
    df.loc[~valid, "group_used"] = None

    df["group_label"] = df[industry_col].where(df["group_used"] == "industry", df[sector_col])
    df.loc[~valid, "group_label"] = None

    df["group_size"] = df.groupby(["group_used", "group_label"])[value_col].transform("size")
    df.loc[~valid, "group_size"] = None

    pct = df.groupby(["group_used", "group_label"])[value_col].rank(pct=True, method="average")
    df["percentile"] = pct if higher_is_better else (1.0 - pct)
    df.loc[~valid, "percentile"] = None

    return df


def group_size_report(panel: pd.DataFrame, rebalance_date_col: str = "rebalance_date",
                       industry_col: str = "industry", sector_col: str = "sector",
                       value_col: str = "value_col_placeholder",
                       min_group_size: int = DEFAULT_MIN_GROUP_SIZE) -> dict:
    """Long-format panel: one row per (ticker, rebalance_date) already
    point-in-time filtered, with industry/sector labels and the metric
    that will be ranked. Documents, across the WHOLE panel, how often the
    industry-level group is too small and falls back to sector-level --
    the exact question the user asked to see answered on the final
    universe before any score is built on top of this normalization."""
    valid = panel[panel[value_col].notna()].copy()
    if valid.empty:
        return {"total_ticker_dates": 0, "industry_groups": 0, "industry_groups_below_threshold": 0,
                "pct_industry_groups_below_threshold": float("nan"),
                "ticker_dates_using_sector_fallback": 0, "pct_ticker_dates_using_sector_fallback": float("nan")}

    group_sizes = valid.groupby([rebalance_date_col, industry_col]).size()
    below = group_sizes[group_sizes < min_group_size]

    fallback_rows = valid.set_index([rebalance_date_col, industry_col]).index.isin(below.index)
    n_fallback = int(fallback_rows.sum())

    return {
        "total_ticker_dates": len(valid),
        "industry_groups": len(group_sizes),
        "industry_groups_below_threshold": len(below),
        "pct_industry_groups_below_threshold": round(100 * len(below) / len(group_sizes), 1) if len(group_sizes) else float("nan"),
        "ticker_dates_using_sector_fallback": n_fallback,
        "pct_ticker_dates_using_sector_fallback": round(100 * n_fallback / len(valid), 1) if len(valid) else float("nan"),
        "smallest_industry_groups": group_sizes.sort_values().head(10).to_dict(),
    }


if __name__ == "__main__":
    # Self-test with SYNTHETIC data only -- proves the mechanics (fallback
    # trigger, percentile orientation, reporting) before the real universe
    # is fully collected (blocked on Alpha Vantage's daily quota as of this
    # writing). NOT a substitute for running group_size_report on the real
    # final universe once collection completes -- see lessons_learned.md.
    demo = pd.DataFrame({
        "ticker": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",  # big industry group (10)
                   "B1", "B2", "B3",       # tiny "Trucking" industry group (3)...
                   "C1", "C2", "C3", "C4", "C5", "C6"],  # ...but the broader "Industrials" SECTOR
                                                          # they share with B1-B3 has 9 members total
        "industry": ["Banks"] * 10 + ["Trucking"] * 3 + ["Machinery"] * 6,
        "sector": ["Financials"] * 10 + ["Industrials"] * 9,
        "price_to_book": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,   # Banks
                          3.0, 3.5, 4.0,                                     # Trucking (B1-B3)
                          1.5, 2.0, 2.5, 3.0, 3.5, 4.0],                     # Machinery (C1-C6)
    })
    result = sector_relative_percentile(demo, "price_to_book", higher_is_better=False)
    print(result[["ticker", "industry", "group_used", "group_size", "percentile"]].to_string(index=False))
    print()
    print("Expected: A1-A10 ranked within 'Banks' (industry, n=10 >= threshold 8, self-contained). "
          "B1-B3 (Trucking, n=3 < 8) fall back to the broader 'Industrials' SECTOR, which pools them "
          "with C1-C6 (Machinery) for n=9 -- big enough, and B1-B3's percentile is now computed "
          "relative to all 9 Industrials names, not just the 3 Trucking ones.")

    panel_demo = pd.concat([demo.assign(rebalance_date="2024-01-31"),
                            demo.assign(rebalance_date="2024-02-29")], ignore_index=True)
    report = group_size_report(panel_demo, value_col="price_to_book")
    print()
    print("group_size_report on the synthetic 2-date panel:", report)
