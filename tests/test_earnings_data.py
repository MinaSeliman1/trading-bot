import json

import pytest

from bot import earnings_data


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(earnings_data, "CACHE_DIR", tmp_path)


def _client(monkeypatch, payload):
    monkeypatch.setattr(earnings_data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    monkeypatch.setattr(earnings_data.time, "sleep", lambda s: None)  # skip real throttling in tests
    return earnings_data.AlphaVantageEarningsClient(api_key="test-key")


def test_successful_response_is_cached_and_parsed(monkeypatch):
    payload = {"symbol": "XYZ", "quarterlyEarnings": [
        {"fiscalDateEnding": "2024-03-31", "reportedDate": "2024-04-25", "reportedEPS": "1.10",
         "estimatedEPS": "1.00", "surprise": "0.10", "surprisePercentage": "10.0", "reportTime": "post-market"},
    ]}
    client = _client(monkeypatch, payload)
    df = client.get_quarterly_surprises("XYZ")
    assert len(df) == 1
    assert client._cache_path("XYZ").exists()


def test_rate_limit_response_raises_and_is_not_cached(monkeypatch):
    """Rate-limit responses must be distinguishable from 'this symbol has no
    data' so a batch caller can stop on quota exhaustion but keep going past
    an individual bad symbol (see test_malformed_empty_response... below)."""
    payload = {"Information": "standard API rate limit is 25 requests per day"}
    client = _client(monkeypatch, payload)
    with pytest.raises(earnings_data.AlphaVantageRateLimitError):
        client.get_quarterly_surprises("XYZ")
    assert not client._cache_path("XYZ").exists()


def test_malformed_empty_response_without_error_keys_is_not_cached(monkeypatch):
    """Regression test: a response shaped without 'Information'/'Note'/'Error
    Message' but with no (or empty) quarterlyEarnings must NOT be cached --
    caching it would permanently and silently treat the symbol as 'no data'
    on every future run, exactly what happened with BK in the live drip-feed."""
    client = _client(monkeypatch, {})
    df = client.get_quarterly_surprises("BK")
    assert df.empty
    assert not client._cache_path("BK").exists()

    client2 = _client(monkeypatch, {"symbol": "BK", "quarterlyEarnings": []})
    df2 = client2.get_quarterly_surprises("BK")
    assert df2.empty
    assert not client2._cache_path("BK").exists()


def test_non_rate_limit_error_message_does_not_raise_just_returns_empty(monkeypatch):
    """An 'Error Message' about an invalid/unsupported symbol (not a quota
    issue) must not be mistaken for rate-limiting -- a batch caller should
    skip this symbol and keep going, not stop the whole day's run."""
    payload = {"Error Message": "the symbol you requested is not supported"}
    client = _client(monkeypatch, payload)
    df = client.get_quarterly_surprises("BADSYM")
    assert df.empty
    assert not client._cache_path("BADSYM").exists()


def test_near_zero_estimate_surprise_pct_is_nulled_not_dropped():
    """surprise_pct = (actual-est)/est blows up to meaningless magnitudes when
    the consensus estimate is near zero (seen up to +-7500% in real cached
    data) -- a formula artifact, not a genuine outsized beat. The row must
    stay (reported/estimated EPS are still valid data) but surprise_pct
    should be nulled so it can't trigger a surprise_pct>=threshold signal."""
    raw = {"quarterlyEarnings": [
        {"fiscalDateEnding": "2009-01-31", "reportedDate": "2009-01-15", "reportedEPS": "-0.28",
         "estimatedEPS": "0.01", "surprise": "-0.29", "surprisePercentage": "-2900.0", "reportTime": "post-market"},
        {"fiscalDateEnding": "2009-04-30", "reportedDate": "2009-04-21", "reportedEPS": "0.39",
         "estimatedEPS": "0.30", "surprise": "0.09", "surprisePercentage": "30.0", "reportTime": "post-market"},
    ]}
    df = earnings_data.AlphaVantageEarningsClient._parse(raw, "TEST")
    assert len(df) == 2  # row is kept, not dropped
    near_zero_row = df.loc[df["estimated_eps"].abs() < earnings_data.MIN_ABS_ESTIMATED_EPS]
    assert near_zero_row["surprise_pct"].isna().all()
    normal_row = df.loc[df["estimated_eps"].abs() >= earnings_data.MIN_ABS_ESTIMATED_EPS]
    assert (normal_row["surprise_pct"] == 30.0).all()


def test_duplicate_report_date_is_deduplicated_preferring_usable_surprise_pct():
    """Regression test for a real case found in cached COST data: the same
    reportedDate appeared twice (a restated/delayed report), which would
    otherwise create a duplicated index and risk double-counting a single
    calendar-day event as two trades."""
    raw = {"quarterlyEarnings": [
        {"fiscalDateEnding": "2007-02-28", "reportedDate": "2007-05-31", "reportedEPS": "0.50",
         "estimatedEPS": "0.50", "surprise": "0.00", "surprisePercentage": "0.0", "reportTime": "post-market"},
        {"fiscalDateEnding": "2007-05-31", "reportedDate": "2007-05-31", "reportedEPS": None,
         "estimatedEPS": None, "surprise": None, "surprisePercentage": None, "reportTime": "post-market"},
    ]}
    df = earnings_data.AlphaVantageEarningsClient._parse(raw, "TEST")
    assert len(df) == 1
    assert not df.index.duplicated().any()
    assert df["surprise_pct"].iloc[0] == 0.0  # kept the row with a usable value, not the all-NaN one


def test_cached_file_is_used_without_a_new_request(monkeypatch, tmp_path):
    payload = {"quarterlyEarnings": [
        {"fiscalDateEnding": "2024-03-31", "reportedDate": "2024-04-25", "reportedEPS": "1.10",
         "estimatedEPS": "1.00", "surprise": "0.10", "surprisePercentage": "10.0", "reportTime": "post-market"},
    ]}
    cache_path = tmp_path / "XYZ_earnings.json"
    with open(cache_path, "w") as f:
        json.dump(payload, f)

    def _fail_if_called(*a, **k):
        raise AssertionError("should not call the API when a valid cache file exists")
    monkeypatch.setattr(earnings_data.requests, "get", _fail_if_called)

    client = earnings_data.AlphaVantageEarningsClient(api_key="test-key")
    df = client.get_quarterly_surprises("XYZ")
    assert len(df) == 1
