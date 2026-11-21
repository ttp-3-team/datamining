"""One-off first evaluation on the temporal test set of the strategy selected on train only.

    uv run python -m disputes.final_test_evaluation --freeze-only   # freeze the selection; test is not opened
    uv run python -m disputes.final_test_evaluation                 # single evaluation; reruns are refused

Only aggregate metrics are stored, no per-dispute predictions or IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes import train_only_study as study
from disputes.backtest import inner_search, train_calibrator
from disputes.config import DECISION_RULE, HGB_SPECS, LOGISTIC_SPECS, OUTER_FOLDS, TRAIN_LAST_WEEK
from disputes.economics import expected_profit
from disputes.evaluation import calibration_bins, evaluate, merchant_status, segment_scores
from disputes.models import calibrate, fit, predict
from disputes.paths import PROJECT
from disputes.preparation import SOURCE_COLUMNS, fit_scale_reference, load_table, normalize_source, prepare_amounts

FAMILIES = {"logistic": LOGISTIC_SPECS, "boosting": HGB_SPECS}
STRATEGY_FAMILIES = {"logistic": ["logistic"], "logistic_calibrated": ["logistic"], "boosting": ["boosting"],
                     "boosting_calibrated": ["boosting"], "blend_50_50": ["logistic", "boosting"]}
OUTPUT = "final_test_outputs"


def freeze_protocol(root: Path) -> dict:
    """Freeze the train-only selection before any access to test."""
    root = Path(root).resolve()
    selection_path = root / study.OUTPUT / "selection.json"
    evidence_path = root / study.OUTPUT / "run_evidence.json"
    train_audit_path = root / "outputs" / "train_audit.csv"
    for path in (selection_path, evidence_path, train_audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact of the earlier train-only stage: {path}")
    selection = proto.read_json(selection_path)
    if proto.read_json(evidence_path).get("test_opened") or selection.get("test_opened"):
        raise ValueError("Train-only report states that test was already opened")
    strategy = selection.get("selected_strategy")
    if strategy not in STRATEGY_FAMILIES:
        raise ValueError(f"Unsupported or missing train-only strategy: {strategy!r}")
    protocol = {
        "version": 1, "selected_strategy": strategy,
        "selection_source": f"{study.OUTPUT}/selection.json", "selection_sha256": proto.sha256(selection_path),
        "train_only_evidence_sha256": proto.sha256(evidence_path), "train_audit_sha256": proto.sha256(train_audit_path),
        "code_sha256": proto.code_hashes("final_test_evaluation.py", "train_only_study.py", "backtest.py", "models.py",
                                         "evaluation.py", "economics.py", "preparation.py", "config.py"),
        "lockfile_sha256": proto.sha256(root / "uv.lock"),
        "inner_folds": OUTER_FOLDS, "decision_rule": DECISION_RULE,
        "test_opened": False,
        # Recorded fact: one failed technical run read the test CSV and crashed before predicting.
        "test_accessed_in_failed_attempt_without_metrics": True,
        "test_reused_for_selection": False,
    }
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    lock_path = output / "final_protocol.json"
    if lock_path.exists():
        if proto.read_json(lock_path) != proto.canonical(protocol):
            raise ValueError("Frozen final protocol differs; do not re-evaluate test")
    else:
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, ensure_ascii=False, indent=2)
    if (output / "test_report.json").exists():
        raise FileExistsError("Final test report already exists; re-evaluation is not allowed")
    return protocol


def raw_from_audit(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = audit[["raw__" + c for c in SOURCE_COLUMNS]].rename(columns=lambda c: c.removeprefix("raw__"))
    data, source_audit = normalize_source(raw)
    if data.dispute_id.tolist() != audit.dispute_id.tolist():
        raise ValueError("Restored raw data IDs do not match the audit")
    return data, source_audit


def fit_strategy(strategy: str, train_raw, source_audit, train, train_audit) -> dict:
    """Tune parameters on temporal train windows and fit final models on all of train."""
    fitted = {}
    for family in STRATEGY_FAMILIES[strategy]:
        spec, _, oof, _ = inner_search(train_raw, source_audit, OUTER_FOLDS, FAMILIES[family])
        calibrator = None
        if strategy == f"{family}_calibrated":
            calibrator, status = train_calibrator(oof[spec["name"]], TRAIN_LAST_WEEK)
            if calibrator is None:
                raise ValueError(f"Calibration failed the train-only criterion: {status}")
        fitted[family] = (spec, fit(train, train_audit, spec), calibrator)
    return fitted


def strategy_probabilities(strategy: str, fitted: dict, data, audit) -> np.ndarray:
    p = {family: calibrate(calibrator, predict(model, data, audit, spec))
         for family, (spec, model, calibrator) in fitted.items()}
    return (p["logistic"] + p["boosting"]) / 2 if strategy == "blend_50_50" else next(iter(p.values()))


def records(frame: pd.DataFrame) -> list[dict]:
    """Table rows for JSON: NaN → null."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def economic_segments(data, audit, p, segment) -> list[dict]:
    table = segment_scores(data, audit, p, segment).rename(columns={"always_usd": "always_score_usd"})
    return records(table[["segment", "rows", "eligible_rows", "win_rate", "mean_probability", "fight_rows",
                          "fight_rate", "score_usd", "always_score_usd", "delta_vs_always_usd"]])


def evaluate_once(root: Path) -> dict:
    """Single test evaluation after checking that selection, data, code and dependencies are unchanged."""
    root = Path(root).resolve()
    protocol = freeze_protocol(root)
    unchanged = {"train_audit": proto.sha256(root / "outputs" / "train_audit.csv") == protocol["train_audit_sha256"],
                 "selection": proto.sha256(root / study.OUTPUT / "selection.json") == protocol["selection_sha256"],
                 "evidence": proto.sha256(root / study.OUTPUT / "run_evidence.json") == protocol["train_only_evidence_sha256"],
                 "code": proto.code_hashes_match(protocol["code_sha256"]),
                 "lockfile": proto.sha256(root / "uv.lock") == protocol["lockfile_sha256"]}
    if not all(unchanged.values()):
        raise ValueError(f"Changed after the final protocol was frozen: {[k for k, ok in unchanged.items() if not ok]}")

    train_raw, train_source_audit, _ = study.load_train_only(root)
    reference = fit_scale_reference(train_raw)
    train, train_audit = prepare_amounts(train_raw, train_source_audit, reference)
    strategy = protocol["selected_strategy"]
    fitted = fit_strategy(strategy, train_raw, train_source_audit, train, train_audit)

    # Test is opened only after the strategy and its final fit are complete.
    test_clean = load_table(root / "outputs", "test_clean.csv")
    test, test_audit = prepare_amounts(*raw_from_audit(load_table(root / "outputs", "test_audit.csv")), reference)
    if test.dispute_id.tolist() != test_clean.dispute_id.tolist():
        raise ValueError("Test audit and clean test table are not aligned by ID")
    for column in ("amount", "amount_usd"):
        if not np.allclose(test[column], test_clean[column], equal_nan=True):
            raise ValueError(f"Reprocessed test does not match the export: {column}")
    if train.dispute_week.max() >= test.dispute_week.min() or set(train.dispute_id) & set(test.dispute_id):
        raise ValueError("Test is not later than train or shares disputes with it")

    p = strategy_probabilities(strategy, fitted, test, test_audit)
    classification, economics = evaluate(test, test_audit, p)
    eligible = test_audit.score_eligible.to_numpy(dtype=bool)
    margin = np.abs(expected_profit(test.amount_usd, p))
    report = {
        "protocol": protocol, "runtime_versions": proto.runtime_versions(),
        "test_clean_sha256": proto.sha256(root / "outputs" / "test_clean.csv"),
        "test_audit_sha256": proto.sha256(root / "outputs" / "test_audit.csv"),
        "strategy": strategy,
        "tuned_logistic_spec": fitted.get("logistic", [None])[0],
        "tuned_boosting_spec": fitted.get("boosting", [None])[0],
        "classification": {k: float(v) if isinstance(v, np.floating) else v for k, v in classification.items()},
        "economics": economics,
        "eligible_rows": int(eligible.sum()), "review_amount_rows": int((~eligible).sum()),
        "decision_margin": {"within_1_usd": int((eligible & (margin <= 1)).sum()),
                            "within_2_usd": int((eligible & (margin <= 2)).sum()),
                            "calibration_bins": records(calibration_bins(test.won, p))},
        "stability_diagnostics": {
            "by_week": economic_segments(test, test_audit, p, test.dispute_week),
            "merchant_seen_status": economic_segments(test, test_audit, p, merchant_status(test, train))},
        "baseline_comparison": "same eligible rows; score and delta against always-fight reported",
        "row_level_predictions_saved": False, "test_used_for_model_selection": False,
        "population_limit": "This dataset includes only historically fought disputes with final outcomes; "
                            "results do not identify effects for disputes historically not fought.",
    }
    with (root / OUTPUT / "test_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze-only", action="store_true", help="freeze the train-only selection without opening test")
    result = freeze_protocol(PROJECT) if parser.parse_args().freeze_only else evaluate_once(PROJECT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
