# Data fetchers — NOT EXECUTED

The session that wrote this code had **no network access to any market-data provider**
(HTTP 403 from the sandbox proxy for Yahoo, Alpaca, Polygon, Databento, Stooq, ...).
These three scripts were therefore **never run**. They are short, use the providers'
documented APIs, and are meant to be reviewed before use. Whatever they write, the
backtest loader (`fcr_fvg.data.load_ohlcv`) + quality-control layer is what protects you:
run `python run_backtest.py ...` and read `qc/*.json` before trusting any number.

| Script | Instruments | History | Needs |
|---|---|---|---|
| `alpaca_fetch.py` | SPY, QQQ | SIP feed: 2016 → now (per the existing repo notes, entitlement-dependent) | `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `pip install alpaca-py` |
| `databento_fetch.py` | ES, NQ (per-contract) | 2010 → now (CME GLBX.MDP3), **paid, check the cost estimate first** | `DATABENTO_API_KEY`, `pip install databento` |
| `yfinance_fetch.py` | any (exploratory only) | **~7 days per request, ~30 days max** for 1-minute bars | `pip install yfinance` |

Always request RAW (unadjusted) prices. Dividend/split-adjusted intraday bars would distort levels.
