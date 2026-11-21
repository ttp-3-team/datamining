"""Часові зрізи, вкладений підбір кандидатів і правило калібрування.

Кожен зріз: модель бачить тижні ≤ last_train, оцінюється на тижнях last_train+1…last_eval.
Опора для очищення сум щоразу навчається лише на ранній частині зрізу.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd

from disputes.evaluation import prediction_rows, summarize_rows
from disputes.models import fit, fit_platt, predict
from disputes.preparation import fit_scale_reference, prepare_amounts

CALIBRATION_RULE = {"method": "sigmoid on logit out-of-time train predictions", "C": 1.,
                    "minimum_rows": 1000, "minimum_each_class": 50, "minimum_periods": 2,
                    "if_insufficient": "raw probabilities, explicitly flagged; exclude strategy from final selection"}
SPEC_COLUMNS = ("variant", "weight", "C")


class Window(NamedTuple):
    train: pd.DataFrame
    train_audit: pd.DataFrame
    eval: pd.DataFrame
    eval_audit: pd.DataFrame
    last_train: int


def time_window(data: pd.DataFrame, source_audit: pd.DataFrame, last_train: int, last_eval: int) -> Window:
    a = data.loc[data.dispute_week.le(last_train)].copy()
    b = data.loc[data.dispute_week.gt(last_train) & data.dispute_week.le(last_eval)].copy()
    if a.empty or b.empty or a.dispute_week.max() >= b.dispute_week.min():
        raise ValueError("Порожній або неправильно впорядкований часовий зріз")
    reference = fit_scale_reference(a)
    return Window(*prepare_amounts(a, source_audit.loc[a.index], reference),
                  *prepare_amounts(b, source_audit.loc[b.index], reference), last_train)


def fit_predict(window: Window, spec: dict) -> np.ndarray:
    model = fit(window.train, window.train_audit, spec)
    return predict(model, window.eval, window.eval_audit, spec)


def spec_columns(spec: dict) -> dict:
    return {key: spec[key] for key in SPEC_COLUMNS if key in spec}


def inner_search(training: pd.DataFrame, source_audit: pd.DataFrame, folds, specs,
                 predictor: Callable[[Window, dict], np.ndarray] = fit_predict):
    """Обрати кандидата за сумарним Score на внутрішніх зрізах; нічия — log loss, менший C, назва.

    Повертає (обрана специфікація, рядки за зрізами, out-of-time прогнози кожного кандидата, рейтинг).
    """
    records, oof = [], {spec["name"]: [] for spec in specs}
    for cutoff, end in folds:
        window = time_window(training, source_audit, cutoff, end)
        constant = float(window.train.won.mean())
        for spec in specs:
            frame = prediction_rows(window.eval, window.eval_audit, predictor(window, spec), constant)
            frame["inner_period"], frame["base_train_max_week"] = f"{cutoff + 1}-{end}", cutoff
            oof[spec["name"]].append(frame)
            records.append({"candidate": spec["name"], **spec_columns(spec), "inner_train_last": cutoff,
                            "inner_eval_last": end, **summarize_rows(frame)})
    oof = {name: pd.concat(frames, ignore_index=True) for name, frames in oof.items()}
    ranking = pd.DataFrame([{"candidate": spec["name"], **spec_columns(spec), **summarize_rows(oof[spec["name"]])}
                            for spec in specs])
    if "C" not in ranking:
        ranking["C"] = np.nan
    ranking = ranking.sort_values(["score_usd", "log_loss", "C", "candidate"],
                                  ascending=[False, True, True, True], na_position="last")
    selected = next(spec for spec in specs if spec["name"] == ranking.candidate.iloc[0])
    return selected, pd.DataFrame(records), oof, ranking


def train_calibrator(oof: pd.DataFrame, outer_train_last: int):
    """Platt-калібратор на out-of-time прогнозах train або None, якщо вибірка замала за правилом."""
    if not oof.dispute_id.is_unique:
        raise ValueError("Повторні калібрувальні ID")
    if not (oof.base_train_max_week < oof.dispute_week).all():
        raise ValueError("Прогноз для калібратора не є out-of-time")
    if oof.dispute_week.max() > outer_train_last:
        raise ValueError("Калібратор отримав мітки зовнішнього періоду")
    lost, won = int(oof.won.eq(0).sum()), int(oof.won.eq(1).sum())
    periods = int(oof.inner_period.nunique())
    rule = CALIBRATION_RULE
    enough = (len(oof) >= rule["minimum_rows"] and min(lost, won) >= rule["minimum_each_class"]
              and periods >= rule["minimum_periods"])
    status = {"calibration_rows": len(oof), "calibration_lost": lost, "calibration_won": won,
              "calibration_periods": periods, "calibration_max_week": int(oof.dispute_week.max()),
              "outer_train_last": outer_train_last, "adequate_by_protocol": bool(enough),
              "status": "fitted_heuristic_sample_rule" if enough else "insufficient_sample_raw_fallback"}
    if not enough:
        return None, status
    calibrator = fit_platt(oof.p, oof.won, rule["C"])
    status.update({"slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])})
    return calibrator, status


def inadequate_calibrations(calibration: pd.DataFrame, column: str = "model") -> set[str]:
    """Назви «<family>_calibrated», які не пройшли правило калібрування хоча б в одному періоді."""
    passed = calibration.groupby(column).adequate_by_protocol.all()
    return {f"{family}_calibrated" for family, ok in passed.items() if not ok}
