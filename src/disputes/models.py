"""Ознаки й моделі. Модель задає `spec` — звичайний dict, який записується в протоколи.

    {"name": "lr_C_0.1", "kind": "logistic", "C": 0.1}                    логістична регресія
    {"kind": "logistic", "C": 1., "variant": "psp_currency",               ... з взаємодіями
     "weight": "amount_p99"}                                              ... і вагами за сумою
    {"name": "hgb_depth4", "kind": "boosting", "depth": 4, "add_week": True}
    {"name": "cat_d4_l2_3.0", "kind": "catboost", "depth": 4, "l2": 3.}
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from disputes.config import SEED, THREADS
from disputes.economics import check_aligned, check_probabilities

BASE_CATEGORIES = ["merchant_id", "psp", "payment_method", "card_brand",
                   "reason_category", "reason_code", "currency", "dispute_weekday"]
# Прапорці аудиту сум, які лише один діагностичний HGB-кандидат отримує як ознаки.
QUALITY = ["currency_dollar_assumed_usd", "amount_invalid_removed", "amount_usd_invalid_removed",
           "amount_restored", "amount_usd_restored", "amount_usd_scale_corrected"]
INTERACTION_PAIRS = {"psp_currency": ("psp", "currency"), "psp_reason": ("psp", "reason_category")}
INTERACTION_VARIANTS = {"baseline": [], "psp_currency": ["psp_currency"], "psp_reason": ["psp_reason"],
                        "both_interactions": ["psp_currency", "psp_reason"]}


def feature_columns(spec: dict) -> tuple[list[str], list[str]]:
    """(категоріальні, числові) колонки для специфікації."""
    cats = [c for c in BASE_CATEGORIES if not (spec.get("drop_merchant") and c == "merchant_id")]
    cats += ["interaction_" + pair for pair in INTERACTION_VARIANTS[spec.get("variant", "baseline")]]
    nums = ["amount_usd"]
    nums += ["amount"] if spec.get("add_amount") else []
    nums += ["dispute_week"] if spec.get("add_week") else []
    nums += QUALITY if spec.get("add_quality") else []
    return cats, nums


def features(data: pd.DataFrame, audit: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[str]]:
    """Лише ознаки, доступні до результату. Пропуск категорії — окреме значення MISSING."""
    check_aligned(data, audit)
    cats, nums = feature_columns(spec)
    base = [c for c in cats if not c.startswith("interaction_")]
    x = data[base].astype("string").fillna("MISSING").astype(str)
    for pair in INTERACTION_VARIANTS[spec.get("variant", "baseline")]:
        left, right = INTERACTION_PAIRS[pair]
        # JSON-пара зберігає межі значень навіть якщо в них є розділювачі.
        x["interaction_" + pair] = [json.dumps([a, b], ensure_ascii=False) for a, b in zip(x[left], x[right], strict=True)]
    for col in nums:
        x[col] = audit[col].to_numpy(dtype=np.int8) if col in QUALITY else data[col].astype(float).to_numpy()
    if {"won", "dispute_id"} & set(x) or any(c.startswith("raw__") for c in x):
        raise AssertionError("Витік цілі або службових полів")
    return x, cats


def new_model(spec: dict):
    cats, nums = feature_columns(spec)
    if spec["kind"] == "logistic":
        amount = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scale", StandardScaler())])
        preprocess = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), cats),
                                        ("amount", amount, nums)])
        classifier = LogisticRegression(C=spec.get("C", 1.), max_iter=2000, random_state=SEED)
        return Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    if spec["kind"] == "boosting":
        preprocess = ColumnTransformer([
            ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats),
            ("numeric", "passthrough", nums)], sparse_threshold=0)
        classifier = HistGradientBoostingClassifier(
            max_iter=400, max_depth=spec["depth"], max_leaf_nodes=31, learning_rate=.05,
            l2_regularization=5., min_samples_leaf=30, categorical_features=list(range(len(cats))),
            early_stopping=False, random_state=SEED)
        return Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    if spec["kind"] == "catboost":
        return CatBoostClassifier(iterations=500, learning_rate=.05, depth=spec["depth"], l2_leaf_reg=spec["l2"],
                                  loss_function="Logloss", boosting_type="Ordered", has_time=True,
                                  thread_count=THREADS, random_seed=SEED, allow_writing_files=False, verbose=False)
    raise ValueError(f"Невідомий тип моделі: {spec['kind']}")


def fit(data: pd.DataFrame, audit: pd.DataFrame, spec: dict, sample_weight=None):
    """Навчити модель на рядках у часовому порядку; ConvergenceWarning — помилка."""
    x, cats = features(data, audit, spec)
    order = np.argsort(data.dispute_week.to_numpy(), kind="stable")
    x, y = x.iloc[order], data.won.iloc[order].astype(int)
    model = new_model(spec)
    kwargs = {}
    if spec["kind"] == "catboost":
        kwargs["cat_features"] = cats
    if sample_weight is not None:
        kwargs["classifier__sample_weight"] = np.asarray(sample_weight)[order]
    with warnings.catch_warnings(), threadpool_limits(limits=THREADS):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, **kwargs)
    return model


def predict(model, data: pd.DataFrame, audit: pd.DataFrame, spec: dict) -> np.ndarray:
    x, _ = features(data, audit, spec)
    with threadpool_limits(limits=THREADS):
        p = model.predict_proba(x)[:, list(model.classes_).index(1)]
    check_probabilities(p, len(data))
    return p


def logits(p) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)


def fit_platt(p, won, C: float = 1.) -> LogisticRegression:
    """Sigmoid-калібрування (Platt) на out-of-time прогнозах."""
    return LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(logits(p), np.asarray(won).astype(int))


def calibrate(calibrator, p) -> np.ndarray:
    """Застосувати калібратор; None означає «залишити сирі ймовірності»."""
    return p if calibrator is None else calibrator.predict_proba(logits(p))[:, 1]
