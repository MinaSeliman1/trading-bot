import numpy as np
import pandas as pd

from bot.regime_detector import RegimeDetector, RANGING, TRENDING


def _trend_bars(n=100, seed=1):
    rng = np.random.default_rng(seed)
    close = np.linspace(100, 200, n) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"high": close + 0.5, "low": close - 0.5, "close": close},
                         index=pd.date_range("2026-01-01", periods=n, freq="D"))


def _choppy_bars(n=100, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close},
                         index=pd.date_range("2026-01-01", periods=n, freq="D"))


def test_adx_classifies_trending_market():
    rd = RegimeDetector()
    regime, adx_val = rd.current_regime(_trend_bars())
    assert regime == TRENDING
    assert adx_val > 25


def test_adx_classifies_ranging_market():
    rd = RegimeDetector()
    regime, adx_val = rd.current_regime(_choppy_bars())
    assert regime == RANGING
    assert adx_val < 20


def test_covariance_shrinkage_is_symmetric_psd():
    rng = np.random.default_rng(7)
    n = 90
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    spy = rng.normal(0.0005, 0.01, n)
    qqq = spy * 0.9 + rng.normal(0, 0.004, n)
    gld = rng.normal(0.0002, 0.008, n)
    returns = pd.DataFrame({"SPY": spy, "QQQ": qqq, "GLD": gld}, index=dates)

    rd = RegimeDetector(corr_lookback_days=60)
    result = rd.covariance_matrices(returns, spy_col="SPY")
    cov = result["normal_covariance"]

    assert np.allclose(cov.values, cov.values.T)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert eigvals.min() >= -1e-10
    # correlated pair should show materially higher covariance than the independent one
    assert abs(cov.loc["SPY", "QQQ"]) > abs(cov.loc["SPY", "GLD"])


def test_stress_covariance_falls_back_to_normal_with_too_few_observations():
    rng = np.random.default_rng(2)
    n = 8
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    returns = pd.DataFrame({
        "SPY": rng.normal(0, 0.01, n), "QQQ": rng.normal(0, 0.01, n), "GLD": rng.normal(0, 0.01, n),
    }, index=dates)
    rd = RegimeDetector(corr_lookback_days=60)
    result = rd.covariance_matrices(returns, spy_col="SPY")
    assert result["stress_covariance"].equals(result["normal_covariance"])
