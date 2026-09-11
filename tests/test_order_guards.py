from bot.order_guards import OrderGuards, ShortNotAllowedError


class FakeAccount:
    def __init__(self, shorting_enabled=True):
        self.shorting_enabled = shorting_enabled


class FakeAsset:
    def __init__(self, shortable=True, marginable=True):
        self.shortable = shortable
        self.marginable = marginable


class FakeClient:
    def __init__(self, shorting_enabled=True, shortable=True, marginable=True):
        self._shorting_enabled = shorting_enabled
        self._shortable = shortable
        self._marginable = marginable

    def get_account(self):
        return FakeAccount(self._shorting_enabled)

    def get_asset(self, symbol):
        return FakeAsset(self._shortable, self._marginable)


class ErrorClient:
    def get_account(self):
        raise RuntimeError("network down")


def test_btc_short_always_blocked_unconditionally():
    guards = OrderGuards(FakeClient())
    allowed, _ = guards.check_short_allowed("BTC/USD", "crypto")
    assert allowed is False


def test_btc_bearish_signal_resolves_to_exit_when_long():
    guards = OrderGuards(FakeClient())
    assert guards.resolve_bearish_signal("BTC/USD", "crypto", has_open_long=True) == "exit"


def test_btc_bearish_signal_resolves_to_none_when_flat():
    guards = OrderGuards(FakeClient())
    assert guards.resolve_bearish_signal("BTC/USD", "crypto", has_open_long=False) == "none"


def test_btc_short_order_raises_at_submit_boundary():
    guards = OrderGuards(FakeClient())
    called = []
    try:
        guards.submit_short_order(lambda **kw: called.append(kw), "BTC/USD", "crypto", qty=1)
        assert False, "expected ShortNotAllowedError"
    except ShortNotAllowedError:
        pass
    assert called == []


def test_equity_short_allowed_when_all_checks_pass():
    guards = OrderGuards(FakeClient(shorting_enabled=True, shortable=True, marginable=True))
    allowed, reason = guards.check_short_allowed("SPY", "us_equity")
    assert allowed is True


def test_equity_short_blocked_when_not_shortable():
    guards = OrderGuards(FakeClient(shorting_enabled=True, shortable=False, marginable=True))
    allowed, _ = guards.check_short_allowed("XYZ", "us_equity")
    assert allowed is False


def test_equity_short_blocked_when_account_shorting_disabled():
    guards = OrderGuards(FakeClient(shorting_enabled=False, shortable=True, marginable=True))
    allowed, _ = guards.check_short_allowed("SPY", "us_equity")
    assert allowed is False


def test_equity_short_fails_safe_on_api_error():
    guards = OrderGuards(ErrorClient())
    allowed, _ = guards.check_short_allowed("SPY", "us_equity")
    assert allowed is False
