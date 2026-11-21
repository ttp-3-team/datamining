"""Economic decision rule and per-dispute revenue.

Fight if the expected benefit exceeds the action cost: p × 0.2 × A > 6.
Without a reliable USD amount no decision is made (`review_amount`) and revenue is unknown (NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from disputes.config import ACTION_COST, BENEFIT_SHARE


def check_aligned(data: pd.DataFrame, audit: pd.DataFrame) -> None:
    if not data.dispute_id.equals(audit.dispute_id):
        raise ValueError("Data and audit rows do not match")


def check_probabilities(p, n: int) -> None:
    p = np.asarray(p, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")


def payoff(data: pd.DataFrame) -> np.ndarray:
    """Realised contribution if fought: 0.2 × amount × won − 6."""
    return BENEFIT_SHARE * data.amount_usd.to_numpy(dtype=float) * data.won.to_numpy(dtype=int) - ACTION_COST


def expected_profit(amount, p) -> np.ndarray:
    return BENEFIT_SHARE * np.asarray(amount, dtype=float) * np.asarray(p, dtype=float) - ACTION_COST


def break_even_probability(amount) -> np.ndarray:
    return ACTION_COST / (BENEFIT_SHARE * np.asarray(amount, dtype=float))


def policy(data: pd.DataFrame, audit: pd.DataFrame, p) -> np.ndarray:
    """Action per dispute: fight / skip / review_amount. The target `won` is not used."""
    check_aligned(data, audit)
    check_probabilities(p, len(data))
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    amount = data.amount_usd.to_numpy(dtype=float)
    if not (np.isfinite(amount[eligible]) & (amount[eligible] > 0)).all():
        raise ValueError("Score eligibility flag contradicts amounts")
    fight = eligible & (BENEFIT_SHARE * amount * np.asarray(p) > ACTION_COST)
    return np.where(~eligible, "review_amount", np.where(fight, "fight", "skip"))


def profit_rows(data: pd.DataFrame, audit: pd.DataFrame, p) -> np.ndarray:
    """Contribution of each dispute to Score; NaN where the amount is unknown."""
    actions = policy(data, audit, p)
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    return np.where(eligible, np.where(actions == "fight", payoff(data), 0.), np.nan)
