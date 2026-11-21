"""Вкладене часове порівняння стратегій лише на train (ноутбук 05).

Для кожного зовнішнього періоду параметри логістичної регресії й HGB та калібратори
обираються на внутрішніх зрізах його train-частини; потім стратегії оцінюються на
зовнішньому періоді. Фінальну стратегію обирає наперед задане правило практичного допуску.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes.backtest import CALIBRATION_RULE, inadequate_calibrations, inner_search, time_window, train_calibrator
from disputes.config import (
    C_VALUES,
    DECISION_RULE,
    HGB_SPECS,
    INNER_FOLDS,
    LOGISTIC_SPECS,
    OUTER_FOLDS,
    PRACTICAL_GAP_USD_PER_1000,
    SEED,
    THREADS,
)
from disputes.evaluation import assert_same_rows, prediction_rows, select_from_summary, summarize_rows, summarize_strategies
from disputes.models import calibrate, fit, predict
from disputes.preparation import SOURCE_COLUMNS, load_table, normalize_source

LR_SPECS = LOGISTIC_SPECS
GB_SPECS = HGB_SPECS
COMPLEXITY = {"constant_train_rate": 0, "logistic": 1, "logistic_calibrated": 2,
              "boosting": 3, "boosting_calibrated": 4, "blend_50_50": 5}
SELECTION = {"primary": "pooled eligible Score", "practical_gap_usd_per_1000": PRACTICAL_GAP_USD_PER_1000,
             "near_best": "within 0.05 USD * pooled eligible rows of highest Score",
             "tie_order": ["lower complexity", "higher worst-period uplift per eligible dispute vs constant",
                           "lower SD of period uplift per dispute vs constant", "lower pooled log loss", "strategy name"],
             "complexity": COMPLEXITY, "include_constant_baseline": True,
             "observed_only": "reported sensitivity, not an additional tuning objective"}
LIMITATIONS = ["earlier EDA and model choices already used these train periods",
               "validation was used in previous experiments and remains reused data",
               "no outcome availability date; no verified label maturity",
               "3 external periods do not establish future stability",
               "calibration sample criteria are heuristics, not a reliability guarantee"]
PROTOCOL = {
    "version": "1.0", "input_allowlist": ["train_audit.csv"],
    "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
    "logistic_C": C_VALUES, "boosting_candidates": GB_SPECS,
    "hyperparameter_choice": "highest summed inner eligible Score; tie: pooled log loss; then smaller C / candidate name",
    "calibration": CALIBRATION_RULE,
    "ensemble": "0.5 * uncalibrated tuned logistic + 0.5 * uncalibrated tuned boosting",
    "decision": DECISION_RULE, "selection": SELECTION,
    "random_seed": SEED, "threads": THREADS, "validation_opened": False, "test_opened": False,
    "limitations": LIMITATIONS,
}
OUTPUT = "train_only_outputs"


class OuterResult(NamedTuple):
    predictions: pd.DataFrame
    grid: pd.DataFrame
    calibration: pd.DataFrame
    choice: dict
    inner: pd.DataFrame
    oof: pd.DataFrame


def freeze_protocol(root):
    """Зафіксувати протокол до читання даних."""
    return proto.freeze(Path(root) / OUTPUT, PROTOCOL, "Протокол train-only")


def load_train_only(root):
    """Відновити train із початкових значень аудиту — єдина таблиця, яку дозволено читати."""
    filename = "train_audit.csv"
    audit = load_table(Path(root) / "outputs", filename)
    raw = audit[["raw__" + c for c in SOURCE_COLUMNS]].rename(columns=lambda c: c.removeprefix("raw__"))
    data, source_audit = normalize_source(raw)
    if data.dispute_id.tolist() != audit.dispute_id.tolist():
        raise ValueError("Відновлені train-ID відрізняються від аудиту")
    if not data.dispute_week.between(0, 17).all():
        raise ValueError("У train є тижні поза 0–17")
    return data, source_audit, [filename]


def labelled_rows(window, probabilities: dict, period: str, column: str = "strategy") -> pd.DataFrame:
    """Рядкові прогнози кількох стратегій на одному зовнішньому періоді."""
    constant = float(window.train.won.mean())
    frames = [prediction_rows(window.eval, window.eval_audit, p, constant).assign(**{column: name, "outer_period": period})
              for name, p in probabilities.items()]
    return pd.concat(frames, ignore_index=True)


def run_outer(data, source_audit, cutoff, end, progress=print) -> OuterResult:
    """Спочатку вибір і калібрування тільки в training; потім зовнішнє оцінювання."""
    training = data.loc[data.dispute_week.le(cutoff)]
    audit = source_audit.loc[training.index]
    lr_spec, lr_records, lr_oof, _ = inner_search(training, audit, INNER_FOLDS[cutoff], LR_SPECS)
    gb_spec, gb_records, gb_oof, _ = inner_search(training, audit, INNER_FOLDS[cutoff], GB_SPECS)
    lr_cal, lr_status = train_calibrator(lr_oof[lr_spec["name"]], cutoff)
    gb_cal, gb_status = train_calibrator(gb_oof[gb_spec["name"]], cutoff)
    # Зовнішні мітки до цього моменту не використовувалися.
    window = time_window(data, source_audit, cutoff, end)
    period = f"{cutoff + 1}-{end}"
    models = {spec["name"]: fit(window.train, window.train_audit, spec) for spec in [*LR_SPECS, gb_spec]}
    raw = {spec["name"]: predict(models[spec["name"]], window.eval, window.eval_audit, spec)
           for spec in [*LR_SPECS, gb_spec]}
    lr_p, gb_p = raw[lr_spec["name"]], raw[gb_spec["name"]]
    probabilities = {"constant_train_rate": np.full(len(window.eval), float(window.train.won.mean())),
                     "logistic": lr_p, "logistic_calibrated": calibrate(lr_cal, lr_p),
                     "boosting": gb_p, "boosting_calibrated": calibrate(gb_cal, gb_p),
                     "blend_50_50": .5 * lr_p + .5 * gb_p}
    # Фіксована C-сітка на тих самих зрізах — діагностика, не заміна вкладеного вибору.
    grid = labelled_rows(window, {spec["C"]: raw[spec["name"]] for spec in LR_SPECS}, period, column="C")
    status = pd.DataFrame([{"model": "logistic", **lr_status}, {"model": "boosting", **gb_status}]).assign(outer_period=period)
    choice = {"outer_period": period, "outer_train_last": cutoff, "outer_eval_last": end,
              "train_rows": len(window.train), "eval_rows": len(window.eval), "selected_C": lr_spec["C"],
              "selected_boosting": gb_spec["name"], "logistic_spec": lr_spec, "boosting_spec": gb_spec}
    inner = pd.concat([lr_records.assign(family="logistic"), gb_records.assign(family="boosting")]).assign(outer_period=period)
    oof = pd.concat([lr_oof[lr_spec["name"]].assign(family="logistic"),
                     gb_oof[gb_spec["name"]].assign(family="boosting")], ignore_index=True).assign(outer_period=period)
    progress(f"Завершено зовнішній період {period}: C={lr_spec['C']}, {gb_spec['name']}", flush=True)
    return OuterResult(labelled_rows(window, probabilities, period), grid, status, choice, inner, oof)


def summarize_grid(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, periods = [], []
    for c, frame in grid.groupby("C"):
        observed = summarize_rows(frame, "observed_only")
        row = {"C": c, **summarize_rows(frame), "observed_score_usd": observed["score_usd"],
               "observed_rows": observed["evaluated_rows"]}
        for period, group in frame.groupby("outer_period", sort=False):
            result = summarize_rows(group)
            row[f"score_weeks_{period}"] = result["score_usd"]
            periods.append({"C": c, "outer_period": period, **result})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score_usd", ascending=False), pd.DataFrame(periods)


def run(root):
    root = Path(root)
    output, protocol_hash = freeze_protocol(root)
    with proto.train_only_access(root) as opened:
        data, source_audit, access_log = load_train_only(root)
        runs = [run_outer(data, source_audit, cutoff, end) for cutoff, end in OUTER_FOLDS]
        predictions, grid, calibration, inner, calibration_oof = (
            pd.concat([getattr(r, field) for r in runs], ignore_index=True)
            for field in ["predictions", "grid", "calibration", "inner", "oof"])
        choices = [r.choice for r in runs]
        assert_same_rows(predictions, columns=["dispute_id", "outer_period", "amount_usd", "eligible", "observed_only"])
        summary, periods = summarize_strategies(predictions, complexity=COMPLEXITY,
                                                not_selectable=inadequate_calibrations(calibration))
        grid_summary, grid_periods = summarize_grid(grid)
        selection = {**select_from_summary(summary), "protocol_sha256": protocol_hash, "outer_choices": choices,
                     "opened_data_tables": access_log, "validation_opened": False, "test_opened": False,
                     "claim": "train-only model selection; no independently confirmed winner"}
        tables = {"strategy_summary": summary, "strategy_periods": periods, "C_grid_summary": grid_summary,
                  "C_grid_periods": grid_periods, "calibration_status": calibration,
                  "inner_candidate_periods": inner, "outer_predictions": predictions,
                  "C_grid_predictions": grid, "calibration_oof": calibration_oof,
                  "outer_choices": pd.DataFrame([{k: v for k, v in c.items() if not k.endswith("_spec")} for c in choices])}
        hashes = proto.write_tables(output, tables)
        proto.write_json(output / "selection.json", selection)
        proto.write_json(output / "run_evidence.json", {
            "protocol_sha256": protocol_hash, "opened_data_tables": access_log, "csv_open_events": opened,
            "train_audit_sha256": proto.sha256(root / "outputs" / "train_audit.csv"),
            "code_sha256": proto.code_hashes("train_only_study.py", "backtest.py", "models.py", "evaluation.py",
                                             "economics.py", "preparation.py", "config.py"),
            "versions": proto.runtime_versions(), "result_sha256": hashes})
    print("Обрана стратегія:", selection["selected_strategy"], flush=True)
    return summary, periods, grid_summary, calibration, selection


if __name__ == "__main__":
    proto.main(freeze_protocol, run)
