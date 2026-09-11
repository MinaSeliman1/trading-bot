"""
Covariance-aware position sizing: before accepting a new position, project
portfolio volatility with the trade added using BOTH the normal and stress
shrunk covariance matrices, take the worse (higher) of the two, and shrink
the new position until projected volatility stays within the vol target --
not a simple pairwise correlation cutoff.

Every sizing decision is logged to risk_decisions.jsonl with its inputs and
the resulting adjustment.
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BISECTION_ITERS = 40


class CorrelationManager:
    def __init__(self, target_vol: float = 0.10):
        self.target_vol = target_vol

    @staticmethod
    def _portfolio_annualized_vol(weights: np.ndarray, daily_cov: np.ndarray) -> float:
        daily_var = float(weights @ daily_cov @ weights)
        return float(np.sqrt(max(daily_var, 0.0) * 252))

    def size_new_position(self, instrument: str, candidate_weight: float, current_weights: dict,
                           normal_cov: pd.DataFrame, stress_cov: pd.DataFrame) -> dict:
        """`candidate_weight` is the proposed new position's notional / equity
        (signed: positive for long, negative for short). `current_weights` maps
        instrument -> existing notional/equity (signed) for already-open
        positions. Returns the approved scale (0..1) to apply to the candidate
        position, plus the projected vol and which covariance regime drove it."""
        instruments = list(normal_cov.columns)
        if instrument not in instruments:
            return {"approved_scale": 1.0, "projected_vol": None, "worst_case_source": None,
                    "reason": f"{instrument} not in covariance matrix; sizing unconstrained"}

        w_current = np.array([current_weights.get(s, 0.0) for s in instruments])
        idx = instruments.index(instrument)

        def worst_case_vol(scale: float) -> tuple:
            w = w_current.copy()
            w[idx] += candidate_weight * scale
            v_normal = self._portfolio_annualized_vol(w, normal_cov.values)
            v_stress = self._portfolio_annualized_vol(w, stress_cov.values)
            if v_stress >= v_normal:
                return v_stress, "stress"
            return v_normal, "normal"

        base_vol, base_source = worst_case_vol(0.0)
        if base_vol > self.target_vol:
            result = {"approved_scale": 0.0, "projected_vol": base_vol, "worst_case_source": base_source,
                      "reason": "existing portfolio already exceeds target vol"}
            return result

        full_vol, full_source = worst_case_vol(1.0)
        if full_vol <= self.target_vol:
            return {"approved_scale": 1.0, "projected_vol": full_vol, "worst_case_source": full_source,
                    "reason": "within target at full candidate size"}

        lo, hi = 0.0, 1.0
        for _ in range(BISECTION_ITERS):
            mid = (lo + hi) / 2
            v, _ = worst_case_vol(mid)
            if v <= self.target_vol:
                lo = mid
            else:
                hi = mid
        final_vol, final_source = worst_case_vol(lo)
        return {"approved_scale": lo, "projected_vol": final_vol, "worst_case_source": final_source,
                "reason": "shrunk to fit target vol"}


def log_sizing_decision(path: str, instrument: str, candidate_weight: float, current_weights: dict,
                         result: dict) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instrument": instrument,
        "candidate_weight": candidate_weight,
        "current_weights": current_weights,
        "approved_scale": result["approved_scale"],
        "projected_vol": result["projected_vol"],
        "worst_case_source": result["worst_case_source"],
        "reason": result["reason"],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
