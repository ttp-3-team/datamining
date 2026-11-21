"""Experiment settings in one place. Nothing is computed here."""

from __future__ import annotations

# Business rule from data_dictionary.md: benefit V = 0.2 × amount_usd, action cost C = $6.
BENEFIT_SHARE = 0.2
ACTION_COST = 6.0
DECISION_RULE = "p * 0.2 * amount_usd > 6; unavailable amount -> review_amount"

# Temporal split: train weeks 0–17, validation 18–21, test 22+.
TRAIN_LAST_WEEK = 17
VALIDATION_LAST_WEEK = 21

# Nested temporal folds on train: (last training week, last evaluation week).
OUTER_FOLDS = [(5, 9), (9, 13), (13, 17)]
INNER_FOLDS = {5: [(1, 3), (3, 5)], 9: [(3, 6), (6, 9)], 13: [(5, 9), (9, 13)]}

C_VALUES = [.01, .1, 1., 10., 100.]
PRACTICAL_GAP_USD_PER_1000 = 50.
SEED = 42
THREADS = 4

# Model candidates (see disputes.models); these lists are written into the study protocols.
LOGISTIC_SPECS = [{"name": f"lr_C_{c:g}", "kind": "logistic", "C": c} for c in C_VALUES]
HGB_SPECS = [
    {"name": "hgb_depth4", "kind": "boosting", "depth": 4},
    {"name": "hgb_depth6", "kind": "boosting", "depth": 6},
    {"name": "hgb_add_amount", "kind": "boosting", "depth": 4, "add_amount": True},
    {"name": "hgb_add_quality", "kind": "boosting", "depth": 4, "add_quality": True},
    {"name": "hgb_without_merchant", "kind": "boosting", "depth": 4, "drop_merchant": True},
    {"name": "hgb_add_week", "kind": "boosting", "depth": 4, "add_week": True},
]
CATBOOST_SPECS = [{"name": f"cat_d{depth}_l2_{l2}", "kind": "catboost", "depth": depth, "l2": l2}
                  for depth in [4, 6] for l2 in [3., 10.]]
