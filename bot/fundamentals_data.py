"""
Fundamentals data: quarterly balance sheet + income statement come from
SEC EDGAR's XBRL companyfacts API (free, no key, no daily quota -- see
lessons_learned.md and exploration_log_value_quality.md for why this
replaced Alpha Vantage's BALANCE_SHEET/INCOME_STATEMENT here). Sector/
industry classification: PRIMARY source is now the Wikipedia "List of
S&P 500 companies" GICS Sector/Sub-Industry table -- one HTML fetch
returns real GICS classification for the ENTIRE S&P 500 at once, free,
no per-ticker quota (verified live 2026-07-30: 91/97 of this project's
universe covered instantly). This replaced Alpha Vantage's OVERVIEW as
the primary path after the AV-quota-gated drip-feed stalled for days
(scheduled-task automation only runs while the app is open -- see
exploration_log_value_quality.md) with 82/97 tickers still outstanding.
Alpha Vantage OVERVIEW is kept ONLY as the fallback for tickers absent
from the current S&P 500 table (6 confirmed: BBWI, MHK, MKTX, MOH, NWL,
WHR -- almost certainly removed from the index at some point; a company's
CURRENT sector is applied retroactively either way, so this fallback
doesn't change the project's existing point-in-time assumption, just
which free source supplies it).

OVERVIEW's RATIOS are deliberately NOT used: it returns only a live
snapshot (today's values, no history), which would be a direct look-ahead
leak if used in any backtest -- see lessons_learned.md. OVERVIEW's
`Sector`/`Industry` fields ARE used -- a company's sector classification
is slow-moving enough that a current snapshot is an acceptable
simplification, unlike its financial ratios. Still a documented
assumption, not a verified fact: if a company was reclassified into a
different sector at some point in its history, this would (incorrectly)
apply its CURRENT sector retroactively. Not expected to matter much for
this project's large, stable, long-listed universe, but worth remembering.

POINT-IN-TIME IMPROVEMENT over the original Alpha-Vantage-only design:
SEC EDGAR's XBRL facts include a real `filed` date (the actual date the
10-Q/10-K was filed) for every reported figure -- this REPLACES the old
`estimate_public_availability_dates` proxy (earnings reportedDate + 5
business days), which was a documented ASSUMPTION, not a measured fact.
`available_date` in `get_quarterly_fundamentals`'s output is now the real
filing date. See lessons_learned.md for why this is a precision
improvement, not just a cost/speed one.
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from bot.earnings_data import AlphaVantageRateLimitError

logger = logging.getLogger(__name__)

CACHE_DIR = Path("fundamentals_cache")
AV_BASE_URL = "https://www.alphavantage.co/query"
AV_MIN_SECONDS_BETWEEN_CALLS = 15  # keeps us well under Alpha Vantage's 5/min free-tier limit

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_WIKI_CACHE = CACHE_DIR / "_sp500_gics_sectors.json"
SP500_WIKI_USER_AGENT = "trading-bot-research contact-not-yet-configured@example.com"

SEC_BASE_URL = "https://data.sec.gov"
# SEC's documented fair-use guidance: no more than ~10 requests/second, and
# a descriptive User-Agent identifying the requester (name/contact) is
# REQUIRED -- SEC will block generic/browser-looking User-Agents. The
# string below is a placeholder identifying this project, not a real
# monitored contact -- replace with a real name+email before any
# sustained/production use, per SEC's own published policy. Flagged
# explicitly rather than left silently generic.
SEC_USER_AGENT = "trading-bot-research contact-not-yet-configured@example.com"
SEC_MIN_SECONDS_BETWEEN_CALLS = 0.15  # ~6-7/sec, comfortably under the ~10/sec guidance
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_TICKER_MAP_CACHE = CACHE_DIR / "_sec_ticker_to_cik.json"

# Verified live, not guessed: SEC's own company_tickers.json currently maps
# "XOM" to CIK 2115436 ("ExxonMobil Holdings Corp") -- an internal
# subsidiary/holdco with entityType "other", tickers=[] and exchanges=[] on
# its OWN submissions endpoint (i.e. it does not actually claim to be
# listed anywhere). The real, NYSE-listed Exxon Mobil Corp is CIK 34088,
# confirmed via its submissions endpoint directly (`"tickers":["XOM"],
# "exchanges":["NYSE"]`, "category":"Large accelerated filer"). This is a
# staleness/error in SEC's ticker-mapping file itself, not a lookup bug
# here -- see exploration_log_value_quality.md for the full verification.
# Add further entries here ONLY after the same direct cross-check (fetch
# the candidate CIK's own /submissions/ endpoint and confirm it lists the
# expected ticker+exchange), never as a guess.
CIK_OVERRIDES = {"XOM": 34088}

# ---------------------------------------------------------------------------
# XBRL TAG PRIORITY LISTS -- documented, explicit, in priority order.
#
# Companies do not all use the same XBRL tag for the same concept (e.g. many
# switched from `Revenues` to `RevenueFromContractWithCustomerExcludingAssessedTax`
# around the 2018 ASC 606 revenue-recognition standard change). For each
# concept below, tags are tried IN ORDER, PER FISCAL QUARTER (not once per
# company) -- so a company that used tag A pre-2018 and tag B after is
# handled correctly, quarter by quarter, not forced onto a single tag for
# its whole history.
#
# If NONE of a concept's candidate tags have data for a given company, that
# concept is left as NaN for that company -- NEVER silently substituted or
# guessed from a different concept. See `get_quarterly_fundamentals`'s
# `missing_concepts` return value, which names exactly this failure mode
# per ticker so it's visible, not buried.
# ---------------------------------------------------------------------------
BALANCE_SHEET_TAGS = {
    "total_shareholder_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # `CommonStockSharesIssued` is deliberately NOT a fallback here -- it is a
    # DIFFERENT concept (includes treasury shares), not a same-concept
    # alternate tag. Verified live on JPM: Issued=4,104,933,895 (constant for
    # years -- authorized/issued share count) vs real Outstanding=~2.7-2.85B
    # (declining, tracks actual buybacks). Mixing them produced a fake
    # sawtooth in book-value-per-share (quarters silently using Issued
    # jumping to Outstanding at fiscal year end) before this was caught --
    # see exploration_log_value_quality.md. `dei:EntityCommonStockSharesOutstanding`
    # (cover-page shares outstanding, namespace "dei" not "us-gaap") IS the
    # correct same-concept fallback -- reported on every 10-Q/10-K cover page,
    # unlike some companies' (e.g. JPM's) us-gaap:CommonStockSharesOutstanding
    # which is only tagged in annual filings.
    "shares_outstanding": [("us-gaap", "CommonStockSharesOutstanding"), ("dei", "EntityCommonStockSharesOutstanding")],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
}
# total_debt = long-term + short-term component, each via its own priority
# list (mirrors the original Alpha Vantage fallback logic: prefer a single
# combined tag if one existed there -- SEC/GAAP has no single "total debt"
# concept, so this is always a sum of the two components here).
LONG_TERM_DEBT_TAGS = ["LongTermDebtNoncurrent", "LongTermDebt"]
SHORT_TERM_DEBT_TAGS = ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"]
INCOME_STATEMENT_TAGS = {
    "total_revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                      "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
}
QUARTERLY_DURATION_DAYS = (80, 100)  # filters duration facts to ~1 fiscal quarter, excludes annual (~365d) totals
# A concept can have SOME historical data (passes the "any_data" check) yet
# have gone silent in recent years -- verified live on JPM: `Revenues`
# (quarterly duration) has real entries 2011-2014, then NONE after
# 2014-09-30 (JPM's 10-Qs stopped tagging a standalone quarterly revenue
# figure at all -- banks typically split interest/noninterest income
# instead, with no aggregate "Revenues" duration fact). This is a real,
# structural SEC/XBRL data gap for financials specifically, not a bug --
# flagged as "stale" (not just "missing") so it's visible per ticker rather
# than silently producing all-NaN downstream ratios with no explanation.
RECENT_DATA_CUTOFF_YEARS = 3


class SECEdgarFundamentalsClient:
    def __init__(self):
        self._last_call_time = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._ticker_to_cik = None

    def _cache_path(self, symbol: str, kind: str) -> Path:
        return CACHE_DIR / f"{symbol.replace('/', '_')}_{kind}.json"

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < SEC_MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(SEC_MIN_SECONDS_BETWEEN_CALLS - elapsed)

    def _get(self, url: str) -> dict:
        self._throttle()
        resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
        self._last_call_time = time.time()
        resp.raise_for_status()
        return resp.json()

    def _load_ticker_map(self) -> dict:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik
        if SEC_TICKER_MAP_CACHE.exists():
            with open(SEC_TICKER_MAP_CACHE) as f:
                raw = json.load(f)
        else:
            raw = self._get(SEC_TICKER_MAP_URL)
            with open(SEC_TICKER_MAP_CACHE, "w") as f:
                json.dump(raw, f)
        # Real shape verified live: {"0": {"cik_str":1045810,"ticker":"NVDA",...}, "1": {...}, ...}
        # -- a dict keyed by string index, NOT a list and NOT {"data": [...]}.
        rows = raw.values() if isinstance(raw, dict) else raw
        self._ticker_to_cik = {row["ticker"]: row["cik_str"] for row in rows if "ticker" in row and "cik_str" in row}
        return self._ticker_to_cik

    def get_cik(self, symbol: str) -> int:
        if symbol.upper() in CIK_OVERRIDES:
            return CIK_OVERRIDES[symbol.upper()]
        mapping = self._load_ticker_map()
        cik = mapping.get(symbol.upper())
        if cik is None:
            raise ValueError(f"No CIK found for ticker {symbol} in SEC's company_tickers.json")
        return int(cik)

    def _fetch_companyfacts(self, symbol: str) -> dict:
        cache_path = self._cache_path(symbol, "secfacts")
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        cik = self.get_cik(symbol)
        raw = self._get(f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik:010d}.json")
        with open(cache_path, "w") as f:
            json.dump(raw, f)
        return raw

    @staticmethod
    def _tag_series(facts: dict, namespace: str, tag: str, duration: bool) -> dict:
        """{end_date -> (val, filed)} for one exact XBRL tag, across all
        units/forms. For duration facts, restricted to QUARTERLY_DURATION_DAYS
        spans only (excludes annual/YTD totals reported under the same tag).
        When the same end_date has multiple filings (restatements, comparative
        figures in later filings), the EARLIEST `filed` date wins -- for
        point-in-time correctness we want the value as it was FIRST
        disclosed, not a later restated figure with a later date, which
        would misrepresent when the information actually became public."""
        concept = facts.get(namespace, {}).get(tag)
        if not concept:
            return {}
        out = {}
        for unit_facts in concept.get("units", {}).values():
            for entry in unit_facts:
                end = entry.get("end")
                filed = entry.get("filed")
                if not end or not filed or entry.get("val") is None:
                    continue
                if duration:
                    start = entry.get("start")
                    if not start:
                        continue
                    span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
                    if not (QUARTERLY_DURATION_DAYS[0] <= span_days <= QUARTERLY_DURATION_DAYS[1]):
                        continue
                prior = out.get(end)
                if prior is None or filed < prior[1]:
                    out[end] = (entry["val"], filed)
        return out

    def _concept_series(self, facts: dict, tag_candidates: list, duration: bool) -> tuple:
        """Merges tag_candidates IN PRIORITY ORDER, per end-date (a company
        that switched tags mid-history is handled correctly: for each
        quarter, the highest-priority tag that HAS data for that quarter
        wins, independently per quarter). Each candidate is either a plain
        tag name (namespace defaults to "us-gaap") or an explicit
        (namespace, tag) tuple -- needed for e.g. dei:EntityCommonStockSharesOutstanding.
        Returns (series_dict, any_tag_had_data)."""
        merged = {}
        any_data = False
        for candidate in tag_candidates:
            namespace, tag = candidate if isinstance(candidate, tuple) else ("us-gaap", candidate)
            series = self._tag_series(facts, namespace, tag, duration)
            if series:
                any_data = True
            for end, (val, filed) in series.items():
                if end not in merged:  # earlier tags in the list take priority
                    merged[end] = (val, filed)
        return merged, any_data

    def get_quarterly_fundamentals(self, symbol: str) -> tuple:
        """Returns (df, missing_concepts). `df` is indexed by
        fiscal_date_ending, columns: total_shareholder_equity,
        shares_outstanding, total_debt, total_assets, total_liabilities,
        total_revenue, gross_profit, net_income, available_date (REAL SEC
        filing date, see module docstring) -- same shape as the previous
        Alpha-Vantage-backed version, so bot/value_quality_signal.py needs
        no changes. `missing_concepts`: list of our-name concepts for which
        NONE of the candidate tags had ANY data for this company -- surfaced
        explicitly, never silently left unnoticed (see BALANCE_SHEET_TAGS
        docstring above)."""
        raw = self._fetch_companyfacts(symbol)
        facts = raw.get("facts", {})
        if not facts:
            return pd.DataFrame(), list(BALANCE_SHEET_TAGS) + ["total_debt"] + list(INCOME_STATEMENT_TAGS)

        columns = {}
        missing_concepts = []

        for name, tags in BALANCE_SHEET_TAGS.items():
            series, any_data = self._concept_series(facts, tags, duration=False)
            if not any_data:
                missing_concepts.append(name)
            columns[name] = series

        # ALIGNMENT FIX, verified live on JPM: dei:EntityCommonStockSharesOutstanding's
        # `end` is the 10-Q/10-K COVER PAGE "as of" date, NOT the fiscal
        # period end -- typically a few weeks later (e.g. end=2024-01-31 for
        # the fiscal quarter actually ending 2023-12-31). Used raw, this
        # created phantom extra rows (a shares-outstanding-only row with no
        # matching equity/assets/etc, since those ARE keyed by the true
        # period end) instead of populating the real quarter's row. Snapped
        # here to the nearest TRUE fiscal-period-end already established by
        # `total_shareholder_equity` (which reliably uses the real period
        # end), within a 45-day tolerance; entries with no close anchor are
        # dropped rather than left as phantom rows.
        equity_ends = sorted(pd.Timestamp(e, tz="UTC") for e in columns["total_shareholder_equity"].keys())
        if columns.get("shares_outstanding") and equity_ends:
            snapped = {}
            for end, value in columns["shares_outstanding"].items():
                end_ts = pd.Timestamp(end, tz="UTC")
                nearest = min(equity_ends, key=lambda e: abs((e - end_ts).days))
                if abs((nearest - end_ts).days) <= 45:
                    nearest_str = nearest.strftime("%Y-%m-%d")
                    if nearest_str not in snapped:  # first (highest-priority-tag) value per anchor wins
                        snapped[nearest_str] = value
            columns["shares_outstanding"] = snapped

        long_series, long_any = self._concept_series(facts, LONG_TERM_DEBT_TAGS, duration=False)
        short_series, short_any = self._concept_series(facts, SHORT_TERM_DEBT_TAGS, duration=False)
        if not (long_any or short_any):
            missing_concepts.append("total_debt")
        debt_series = {}
        for end in set(long_series) | set(short_series):
            l = long_series.get(end)
            s = short_series.get(end)
            if l is None and s is None:
                continue
            val = (l[0] if l else 0) + (s[0] if s else 0)
            filed = max(f for f in (l[1] if l else None, s[1] if s else None) if f)
            debt_series[end] = (val, filed)
        columns["total_debt"] = debt_series

        for name, tags in INCOME_STATEMENT_TAGS.items():
            series, any_data = self._concept_series(facts, tags, duration=True)
            if not any_data:
                missing_concepts.append(name)
            columns[name] = series

        # Recency check -- see RECENT_DATA_CUTOFF_YEARS docstring: a concept
        # can pass "any_data" yet have no data at all in the window this
        # backtest actually uses.
        cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.DateOffset(years=RECENT_DATA_CUTOFF_YEARS)
        for name, series in columns.items():
            if name in missing_concepts or not series:
                continue
            latest_end = max(pd.Timestamp(e, tz="UTC") for e in series.keys())
            if latest_end < cutoff:
                missing_concepts.append(f"{name} (stale: no data since {latest_end.date()})")

        all_ends = set()
        for series in columns.values():
            all_ends |= set(series.keys())
        if not all_ends:
            return pd.DataFrame(), missing_concepts

        # available_date is deliberately gated ONLY by the balance-sheet
        # concepts (+ total_debt) -- NOT by the income-statement ones. Found
        # live on CAT: a discrete Q4 revenue figure sometimes exists ONLY via
        # a much-later 8-K (here, filed over a year after the quarter,
        # apparently the only place CAT ever disclosed that discrete
        # number) -- gating the WHOLE row's availability on that would make
        # otherwise-perfectly-timely P/B and debt/equity ratios (which don't
        # need revenue at all) wait 13+ months for no real reason. A late/
        # absent revenue figure still correctly produces a NaN net_margin
        # for that quarter (see INCOME_STATEMENT_TAGS) -- it just no longer
        # blocks the OTHER ratios that don't depend on it.
        AVAILABILITY_GATING_CONCEPTS = list(BALANCE_SHEET_TAGS) + ["total_debt"]

        rows = []
        for end in sorted(all_ends):
            row = {"fiscal_date_ending": end}
            filed_dates = []
            for name, series in columns.items():
                entry = series.get(end)
                row[name] = entry[0] if entry else float("nan")
                if entry and name in AVAILABILITY_GATING_CONCEPTS:
                    filed_dates.append(entry[1])
            # available_date = latest filing date among the concepts that
            # make up this row -- if equity was filed on the same day as
            # revenue (normal case, same 10-Q/10-K), this is just that date;
            # if they came from different filings for some reason, using the
            # LATEST keeps this point-in-time-safe (all pieces are public by
            # this date, never fewer).
            row["available_date"] = max(filed_dates) if filed_dates else None
            rows.append(row)

        df = pd.DataFrame(rows).dropna(subset=["available_date"])
        df["fiscal_date_ending"] = pd.to_datetime(df["fiscal_date_ending"], utc=True)
        df["available_date"] = pd.to_datetime(df["available_date"], utc=True)
        df = df.set_index("fiscal_date_ending").sort_index()
        # Essential concepts for any ratio at all -- drop rows missing ALL of
        # them (not individual NaNs elsewhere, which stay as explicit gaps).
        df = df.dropna(subset=["total_shareholder_equity", "shares_outstanding", "net_income", "total_revenue"],
                       how="all")
        return df, missing_concepts

    def _load_sp500_gics_mapping(self) -> dict:
        """One-shot fetch of the whole S&P 500 GICS Sector/Sub-Industry
        table from Wikipedia -- free, no key, no per-ticker quota. Cached
        to disk like everything else here. Returns {ticker: {"sector":...,
        "industry":...}} using AV's field names (`sector`/`industry`) for
        drop-in compatibility, even though the values are GICS Sector/
        Sub-Industry, not Alpha Vantage's own taxonomy -- both are
        GICS-based in practice, so this is not a meaningful format change
        for factor_normalization.py."""
        if SP500_WIKI_CACHE.exists():
            with open(SP500_WIKI_CACHE) as f:
                return json.load(f)
        resp = requests.get(SP500_WIKI_URL, headers={"User-Agent": SP500_WIKI_USER_AGENT}, timeout=30)
        resp.raise_for_status()
        import io
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        mapping = {row["Symbol"]: {"sector": row["GICS Sector"], "industry": row["GICS Sub-Industry"]}
                  for _, row in df.iterrows()}
        with open(SP500_WIKI_CACHE, "w") as f:
            json.dump(mapping, f)
        return mapping

    def get_sector_industry(self, symbol: str, api_key: str) -> dict:
        """PRIMARY source: the Wikipedia S&P 500 GICS table (see
        `_load_sp500_gics_mapping`) -- covers 91/97 of this project's
        universe with zero quota. FALLBACK: Alpha Vantage OVERVIEW, only
        for tickers absent from that table (confirmed: BBWI, MHK, MKTX,
        MOH, NWL, WHR -- see module docstring). Current snapshot only in
        both cases, see module docstring's point-in-time caveat."""
        gics = self._load_sp500_gics_mapping()
        if symbol.upper() in gics:
            result = gics[symbol.upper()]
            cache_path = self._cache_path(symbol, "overview")
            if not cache_path.exists():
                with open(cache_path, "w") as f:
                    json.dump({"Sector": result["sector"], "Industry": result["industry"],
                              "_source": "wikipedia_sp500_gics"}, f)
            return result

        cache_path = self._cache_path(symbol, "overview")
        if cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
        else:
            elapsed = time.time() - self._last_call_time
            if elapsed < AV_MIN_SECONDS_BETWEEN_CALLS:
                time.sleep(AV_MIN_SECONDS_BETWEEN_CALLS - elapsed)
            resp = requests.get(AV_BASE_URL, params={"function": "OVERVIEW", "symbol": symbol, "apikey": api_key},
                                 timeout=30)
            self._last_call_time = time.time()
            resp.raise_for_status()
            raw = resp.json()

            rate_limit_text = raw.get("Information") or raw.get("Note")
            if rate_limit_text and ("rate limit" in rate_limit_text.lower() or "per day" in rate_limit_text.lower()
                                    or "per minute" in rate_limit_text.lower()):
                logger.error("Alpha Vantage rate limit hit for %s/overview: %s", symbol, rate_limit_text)
                raise AlphaVantageRateLimitError(rate_limit_text)

            if not raw.get("Sector"):
                logger.error("Alpha Vantage returned no usable OVERVIEW data for %s: %s", symbol,
                             raw.get("Information") or raw.get("Note") or raw.get("Error Message") or
                             "response missing Sector")
                return {"sector": None, "industry": None}

            with open(cache_path, "w") as f:
                json.dump(raw, f)

        return {"sector": raw.get("Sector") or None, "industry": raw.get("Industry") or None}
