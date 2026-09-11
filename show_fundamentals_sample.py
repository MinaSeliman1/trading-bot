"""
One-off verification script (not part of the signal itself): builds the
point-in-time fundamentals table for one ticker (JPM) and prints it, so the
availability-date logic and derived ratios can be eyeballed for sanity
before any value/quality signal is built on top of them.
"""
import sys

import pandas as pd

from bot.fundamentals_data import SECEdgarFundamentalsClient

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "JPM"


def main():
    fundamentals_client = SECEdgarFundamentalsClient()
    pit, missing_concepts = fundamentals_client.get_quarterly_fundamentals(SYMBOL)
    if missing_concepts:
        print(f"Concepts with NO data for {SYMBOL} under any known tag: {missing_concepts}\n")

    price_bars = pd.read_parquet(f"data_cache/{SYMBOL}_1Day.parquet")
    price_bars.index = pd.to_datetime(price_bars.index, utc=True)
    price_bars = price_bars.sort_index()
    print(f"Cached price data range for {SYMBOL}: {price_bars.index.min().date()} to {price_bars.index.max().date()} "
          f"({len(price_bars)} bars)\n")

    rows = []
    for fiscal_date, row in pit.iterrows():
        avail = row["available_date"]
        future_bars = price_bars[price_bars.index >= avail]
        if future_bars.empty:
            continue  # no price data on/after this quarter's availability date (outside our cached price range)
        price = float(future_bars["close"].iloc[0])
        price_date = future_bars.index[0]

        bvps = row["total_shareholder_equity"] / row["shares_outstanding"] if row["shares_outstanding"] else float("nan")
        p_to_b = price / bvps if bvps and bvps == bvps else float("nan")
        net_margin = row["net_income"] / row["total_revenue"] if row["total_revenue"] else float("nan")
        debt_to_equity = row["total_debt"] / row["total_shareholder_equity"] if row["total_shareholder_equity"] else float("nan")

        rows.append({
            "fiscal_date_ending": fiscal_date.date(),
            "available_date": avail.date(),
            "price_used_date": price_date.date(),
            "price": round(price, 2),
            "book_value_per_share": round(bvps, 2) if bvps == bvps else None,
            "price_to_book": round(p_to_b, 3) if p_to_b == p_to_b else None,
            "net_margin_pct": round(net_margin * 100, 1) if net_margin == net_margin else None,
            "debt_to_equity": round(debt_to_equity, 3) if debt_to_equity == debt_to_equity else None,
        })

    out = pd.DataFrame(rows)
    print(f"Point-in-time rows with cached price coverage: {len(out)} of {len(pit)} total fundamentals quarters\n")
    print(out.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
