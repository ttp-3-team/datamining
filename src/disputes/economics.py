"""Грошове правило: коли боротися і скільки це приносить.

Боремося, якщо очікуваний виграш більший за ціну дії: p × 0,2 × A > 6.
Без надійної USD-суми рішення не приймається (`review_amount`), а дохід невідомий (NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from disputes.config import ACTION_COST, BENEFIT_SHARE


def check_aligned(data: pd.DataFrame, audit: pd.DataFrame) -> None:
    if not data.dispute_id.equals(audit.dispute_id):
        raise ValueError("Рядки даних і аудиту не відповідають одне одному")


def check_probabilities(p, n: int) -> None:
    p = np.asarray(p, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Некоректні ймовірності")


def payoff(data: pd.DataFrame) -> np.ndarray:
    """Фактичний внесок, якщо боротися: 0,2 × сума × won − 6."""
    return BENEFIT_SHARE * data.amount_usd.to_numpy(dtype=float) * data.won.to_numpy(dtype=int) - ACTION_COST


def expected_profit(amount, p) -> np.ndarray:
    return BENEFIT_SHARE * np.asarray(amount, dtype=float) * np.asarray(p, dtype=float) - ACTION_COST


def break_even_probability(amount) -> np.ndarray:
    return ACTION_COST / (BENEFIT_SHARE * np.asarray(amount, dtype=float))


def policy(data: pd.DataFrame, audit: pd.DataFrame, p) -> np.ndarray:
    """Дія для кожного спору: fight / skip / review_amount. Ціль `won` не використовується."""
    check_aligned(data, audit)
    check_probabilities(p, len(data))
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    amount = data.amount_usd.to_numpy(dtype=float)
    if not (np.isfinite(amount[eligible]) & (amount[eligible] > 0)).all():
        raise ValueError("Прапорець Score суперечить сумам")
    fight = eligible & (BENEFIT_SHARE * amount * np.asarray(p) > ACTION_COST)
    return np.where(~eligible, "review_amount", np.where(fight, "fight", "skip"))


def profit_rows(data: pd.DataFrame, audit: pd.DataFrame, p) -> np.ndarray:
    """Внесок кожного спору в Score; NaN там, де сума невідома."""
    actions = policy(data, audit, p)
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    return np.where(eligible, np.where(actions == "fight", payoff(data), 0.), np.nan)
