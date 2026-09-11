"""
Regime detection: rolling ADX per instrument, plus normal and stress
cross-instrument covariance matrices (EWMA volatility x correlation, then
Ledoit-Wolf shrunk toward a stable target rather than used raw).
"""
import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

RANGING = "RANGING"
TRENDING = "TRENDING"
NEUTRAL = "NEUTRAL"

ADX_RANGING_MAX = 20.0
ADX_TRENDING_MIN = 25.0

MIN_STRESS_OBSERVATIONS = 5  # below this, stress covariance falls back to normal


def calculate_adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ADX. `bars` must be oldest-first with high/low/close columns."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=bars.index)
    minus_dm = pd.Series(minus_dm, index=bars.index)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def classify_regime(adx_value: float) -> str:
    if adx_value != adx_value:  # NaN
        return NEUTRAL
    if adx_value < ADX_RANGING_MAX:
        return RANGING
    if adx_value > ADX_TRENDING_MIN:
        return TRENDING
    return NEUTRAL


def ewma_volatility(returns: pd.Series, span: int = 20) -> pd.Series:
    """Annualized EWMA volatility of a daily-return series."""
    return returns.ewm(span=span, adjust=False).std() * np.sqrt(252)


class RegimeDetector:
    def __init__(self, adx_period: int = 14, corr_lookback_days: int = 60, stress_percentile: float = 0.10):
        self.adx_period = adx_period
        self.corr_lookback_days = corr_lookback_days
        self.stress_percentile = stress_percentile

    def adx_series(self, bars: pd.DataFrame) -> pd.Series:
        return calculate_adx(bars, self.adx_period)

    def current_regime(self, bars: pd.DataFrame) -> tuple:
        adx = self.adx_series(bars)
        if adx.empty or adx.iloc[-1] != adx.iloc[-1]:
            return NEUTRAL, float("nan")
        value = float(adx.iloc[-1])
        return classify_regime(value), value

    # ------------------------------------------------------------------ #
    # covariance
    # ------------------------------------------------------------------ #
    def _shrunk_covariance(self, returns_window: pd.DataFrame) -> tuple:
        """Fit Ledoit-Wolf shrinkage on a window of daily returns (columns =
        instruments). Returns (shrunk_covariance_df, shrinkage_intensity)."""
        clean = returns_window.dropna(axis=0, how="any")
        if len(clean) < 2 or clean.shape[1] < 2:
            return None, None
        lw = LedoitWolf().fit(clean.values)
        cov_df = pd.DataFrame(lw.covariance_, index=clean.columns, columns=clean.columns)
        return cov_df, float(lw.shrinkage_)

    def covariance_matrices(self, daily_returns: pd.DataFrame, spy_col: str = "SPY", ewma_span: int = 20) -> dict:
        """`daily_returns`: DataFrame of daily returns indexed by date, one
        column per instrument, already trimmed to the lookback window.
        Returns normal + stress shrunk covariance matrices plus per-instrument
        EWMA annualized vol (used downstream for vol-targeted sizing)."""
        window = daily_returns.tail(self.corr_lookback_days)
        ewma_vol = {col: float(ewma_volatility(daily_returns[col].dropna(), span=ewma_span).iloc[-1])
                    if daily_returns[col].dropna().shape[0] else float("nan")
                    for col in daily_returns.columns}

        normal_cov, normal_shrinkage = self._shrunk_covariance(window)

        stress_cov, stress_shrinkage = None, None
        if spy_col in window.columns:
            spy_returns = window[spy_col].dropna()
            if len(spy_returns) >= 10:
                threshold = spy_returns.quantile(self.stress_percentile)
                stress_dates = spy_returns[spy_returns <= threshold].index
                stress_window = window.loc[window.index.isin(stress_dates)]
                if len(stress_window) >= MIN_STRESS_OBSERVATIONS:
                    stress_cov, stress_shrinkage = self._shrunk_covariance(stress_window)

        if stress_cov is None:
            logger.debug("Insufficient stress-day observations; falling back to normal covariance for stress estimate")
            stress_cov, stress_shrinkage = normal_cov, normal_shrinkage

        return {
            "normal_covariance": normal_cov,
            "normal_shrinkage": normal_shrinkage,
            "stress_covariance": stress_cov,
            "stress_shrinkage": stress_shrinkage,
            "ewma_vol": ewma_vol,
        }
