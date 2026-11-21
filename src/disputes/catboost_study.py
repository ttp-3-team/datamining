"""CatBoost у тій самій вкладеній train-only схемі (ноутбук 06).

Стратегії ноутбука 05 не перераховуються: їхні зовнішні прогнози читаються з перевіркою
контрольних сум, а CatBoost додається на тих самих спорах і періодах.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from disputes import protocol as proto
from disputes import train_only_study as previous
from disputes.backtest import inadequate_calibrations, inner_search, time_window, train_calibrator
from disputes.config import CATBOOST_SPECS, INNER_FOLDS, OUTER_FOLDS
from disputes.evaluation import assert_same_rows, select_from_summary, summarize_strategies
from disputes.models import calibrate, fit, predict

SPECS = CATBOOST_SPECS
COMPLEXITY = {**previous.COMPLEXITY, "catboost": 3, "catboost_calibrated": 4, "lr_cat_50_50": 5}
PROTOCOL = {
    "version": "1.0", "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
    "candidates": SPECS, "base_features": "same categorical features and amount_usd as logistic regression",
    "parameters": {"iterations": 500, "learning_rate": .05, "boosting_type": "Ordered",
                   "has_time": True, "thread_count": 4, "random_seed": 42, "early_stopping": False},
    "tuning": "highest pooled inner eligible Score; tie: pooled log loss then candidate name",
    "calibration": previous.PROTOCOL["calibration"], "blend": "0.5 raw tuned logistic + 0.5 raw tuned CatBoost",
    "selection": {**previous.SELECTION, "complexity": COMPLEXITY},
    "baselines": "verified frozen train-only out-of-time predictions from notebook 05",
    "input_tables": ["outputs/train_audit.csv", "train_only_outputs/outer_predictions.csv"],
    "validation_opened": False, "test_opened": False,
    "limitations": previous.LIMITATIONS + ["within-week event order is unknown",
                                           "ordered boosting does not establish outcome availability at prediction time"],
}
OUTPUT = "catboost_outputs"


def freeze(root):
    """Зафіксувати протокол до читання даних."""
    return proto.freeze(Path(root) / OUTPUT, PROTOCOL, "Протокол CatBoost")


def load_previous(root):
    """Перевірені результати ноутбука 05: зовнішні прогнози, статус калібрування, докази запуску."""
    folder = Path(root) / previous.OUTPUT
    evidence = proto.read_json(folder / "run_evidence.json")
    if proto.sha256(Path(root) / "outputs" / "train_audit.csv") != evidence["train_audit_sha256"]:
        raise ValueError("Train-аудит відрізняється від попереднього експерименту")
    predictions = proto.verified_csv(folder / "outer_predictions.csv", evidence["result_sha256"]["outer_predictions"],
                                     dtype={"dispute_id": "string", "outer_period": "string"})
    calibration = proto.verified_csv(folder / "calibration_status.csv", evidence["result_sha256"]["calibration_status"])
    return predictions, calibration, evidence


def run(root):
    root = Path(root)
    output, protocol_hash = freeze(root)
    with proto.train_only_access(root) as opened:
        data, audit, _ = previous.load_train_only(root)
        baselines, old_calibration, old_evidence = load_previous(root)
        frames, calibrations, records, choices, oof_frames = [], [], [], [], []
        for cutoff, end in OUTER_FOLDS:
            training = data.loc[data.dispute_week.le(cutoff)]
            spec, candidates, oof, _ = inner_search(training, audit.loc[training.index], INNER_FOLDS[cutoff], SPECS)
            oof = oof[spec["name"]]
            calibrator, status = train_calibrator(oof, cutoff)
            window = time_window(data, audit, cutoff, end)
            model = fit(window.train, window.train_audit, spec)
            p = predict(model, window.eval, window.eval_audit, spec)
            period = f"{cutoff + 1}-{end}"
            lr = baselines.loc[baselines.strategy.eq("logistic") & baselines.outer_period.eq(period)]
            if lr.dispute_id.tolist() != window.eval.dispute_id.tolist():
                raise ValueError("ID CatBoost і попередніх стратегій не збігаються")
            probabilities = {"catboost": p, "catboost_calibrated": calibrate(calibrator, p),
                             "lr_cat_50_50": .5 * p + .5 * lr.p.to_numpy()}
            frames.append(previous.labelled_rows(window, probabilities, period))
            calibrations.append({"model": "catboost", "outer_period": period, **status})
            records.append(candidates.assign(outer_period=period))
            oof_frames.append(oof.assign(outer_period=period))
            choices.append({"outer_period": period, **spec})
            # Збережена модель має відтворювати ті самі прогнози без повторного навчання.
            path = output / f"catboost_weeks_{period}.cbm"
            model.save_model(str(path))
            restored = CatBoostClassifier().load_model(str(path))
            np.testing.assert_allclose(predict(restored, window.eval, window.eval_audit, spec), p, atol=1e-12)
            print(f"Completed train-only outer weeks {period}: {spec['name']}", flush=True)
        predictions = pd.concat([baselines, *frames], ignore_index=True)
        calibration = pd.concat([old_calibration, pd.DataFrame(calibrations)], ignore_index=True)
        assert_same_rows(predictions)
        summary, periods = summarize_strategies(predictions, complexity=COMPLEXITY,
                                                not_selectable=inadequate_calibrations(calibration))
        selection = {**select_from_summary(summary), "protocol_sha256": protocol_hash, "outer_choices": choices,
                     "validation_opened": False, "test_opened": False,
                     "prior_protocol_sha256": old_evidence["protocol_sha256"]}
        tables = {"strategy_summary": summary, "strategy_periods": periods, "outer_predictions": predictions,
                  "calibration_status": calibration, "inner_candidate_periods": pd.concat(records),
                  "catboost_calibration_oof": pd.concat(oof_frames), "outer_choices": pd.DataFrame(choices)}
        hashes = proto.write_tables(output, tables)
        proto.write_json(output / "selection.json", selection)
        proto.write_json(output / "run_evidence.json", {
            "protocol_sha256": protocol_hash, "csv_open_events": opened, "prior_evidence": old_evidence,
            "code_sha256": proto.code_hashes("catboost_study.py", "train_only_study.py", "backtest.py", "models.py",
                                             "evaluation.py", "economics.py", "preparation.py", "config.py"),
            "versions": proto.runtime_versions(), "result_sha256": hashes})
    print("Selected:", selection["selected_strategy"], flush=True)
    return summary, periods, calibration, selection


if __name__ == "__main__":
    proto.main(freeze, run)
