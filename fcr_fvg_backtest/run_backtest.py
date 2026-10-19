#!/usr/bin/env python3
"""CLI: run the First-Candle-Rule + FVG study on real 1-minute OHLCV files.

Example
-------
  python run_backtest.py --data SPY=data/SPY_1m.parquet QQQ=data/QQQ_1m.parquet \
                         ES=data/ES_1m.csv NQ=data/NQ_1m.csv \
                         --input-tz UTC --out results/run1

Every file needs: a timestamp column, open, high, low, close (volume optional).
Futures files should also carry a 'contract' column (raw, UNADJUSTED per-contract bars).
"""
import argparse
import sys

from fcr_fvg.params import StrategyParams
from fcr_fvg.pipeline import run_study


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, metavar="SYMBOL=PATH",
                    help="one or more SYMBOL=file pairs (SPY, QQQ, ES, NQ)")
    ap.add_argument("--out", default="results/run", help="output directory")
    ap.add_argument("--input-tz", default=None,
                    help="timezone of NAIVE timestamps (e.g. UTC, America/New_York). Not needed if the file has offsets/epochs.")
    ap.add_argument("--ts-convention", choices=("start", "end"), default="start",
                    help="whether timestamps mark the bar open (default) or close")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--no-calendar", action="store_true", help="do not use pandas_market_calendars")
    ap.add_argument("--audit-n", type=int, default=10, help="dump bars of the first N trades per run for manual checking")
    a = ap.parse_args(argv)

    files = {}
    for item in a.data:
        if "=" not in item:
            ap.error(f"--data expects SYMBOL=PATH, got {item!r}")
        sym, path = item.split("=", 1)
        files[sym.upper()] = path

    res = run_study(files, a.out, input_tz=a.input_tz, ts_convention=a.ts_convention, base=StrategyParams(),
                    n_boot=a.n_boot, sensitivity=not a.no_sensitivity, audit_n=a.audit_n,
                    use_calendar=not a.no_calendar)
    print(res["report"])
    print(f"\nOutputs written to {a.out}/ (report_auto.md, summary.csv, trades/, charts/, breakdowns/, audit/, qc/)")


if __name__ == "__main__":
    sys.exit(main())
