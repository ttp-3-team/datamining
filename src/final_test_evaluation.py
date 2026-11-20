"""One-shot, train-only-selected final evaluation on the temporal test split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from preparation import (
    SOURCE_COLUMNS, fit_scale_reference, load_table, normalize_source,
    prepare_amounts,
)
import train_only_study as study
from model_comparison import evaluate, predict

FINAL_FOLDS = [[5, 9], [9, 13], [13, 17]]
SUPPORTED_STRATEGIES = {
    "logistic", "logistic_calibrated", "boosting", "boosting_calibrated", "blend_50_50"
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_protocol(root: Path) -> dict:
    """Lock the strategy selected using train-only evidence before touching test."""
    root = Path(root).resolve()
    selection_path = root / "train_only_outputs" / "selection.json"
    evidence_path = root / "train_only_outputs" / "run_evidence.json"
    train_audit_path = root / "outputs" / "train_audit.csv"
    for path in (selection_path, evidence_path, train_audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Потрібен артефакт попереднього train-only етапу: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("test_opened") is True or selection.get("test_opened") is True:
        raise ValueError("Train-only звіт указує, що test вже відкривали")
    strategy = selection.get("selected_strategy")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Непідтримувана або відсутня train-only стратегія: {strategy!r}")
    output = root / "final_test_outputs"
    output.mkdir(exist_ok=True)
    lock_path = output / "final_protocol.json"
    code_files = ["src/final_test_evaluation.py", "src/train_only_study.py",
                  "src/model_comparison.py", "src/preparation.py"]
    protocol = {
        "version": 1,
        "selected_strategy": strategy,
        "selection_source": "train_only_outputs/selection.json",
        "selection_sha256": sha256(selection_path),
        "train_only_evidence_sha256": sha256(evidence_path),
        "train_audit_sha256": sha256(train_audit_path),
        "code_sha256": {name: sha256(root / name) for name in code_files},
        "requirements_sha256": sha256(root / "requirements_comparison.txt"),
        "inner_folds": FINAL_FOLDS,
        "decision_rule": "p * 0.2 * amount_usd > 6; unavailable amount -> review_amount",
        "test_opened": False,
        "test_accessed_in_failed_attempt_without_metrics": True,
        "test_reused_for_selection": False,
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("Зафіксований фінальний протокол відрізняється; test повторно не оцінювати")
    else:
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, ensure_ascii=False, indent=2)
    if (output / "test_report.json").exists():
        raise FileExistsError("Фінальний звіт test уже існує; повторне оцінювання заборонено")
    return protocol


def _raw_from_audit(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = audit[["raw__" + c for c in SOURCE_COLUMNS]].rename(columns=lambda c: c[5:])
    data, source_audit = normalize_source(raw)
    if data.dispute_id.tolist() != audit.dispute_id.tolist():
        raise ValueError("ID відновлених сирих даних не збігаються з аудитом")
    return data, source_audit


def _fit_strategy(strategy, train_raw, source_audit, train_clean, train_clean_audit):
    lr_spec = gb_spec = lr_oof = gb_oof = None
    if strategy in {"logistic", "logistic_calibrated", "blend_50_50"}:
        lr_spec, _, lr_oof, _ = study.inner_search(train_raw, source_audit, FINAL_FOLDS, study.LR_SPECS)
    if strategy in {"boosting", "boosting_calibrated", "blend_50_50"}:
        gb_spec, _, gb_oof, _ = study.inner_search(train_raw, source_audit, FINAL_FOLDS, study.GB_SPECS)
    lr_model = study.fit(train_clean, train_clean_audit, lr_spec) if lr_spec else None
    gb_model = study.fit(train_clean, train_clean_audit, gb_spec) if gb_spec else None
    lr_cal = gb_cal = None
    if strategy == "logistic_calibrated":
        lr_cal, status = study.train_calibrator(lr_oof[lr_spec["name"]], 17)
        if lr_cal is None:
            raise ValueError(f"Калібрування логістичної моделі не пройшло train-only критерій: {status}")
    elif strategy == "boosting_calibrated":
        gb_cal, status = study.train_calibrator(gb_oof[gb_spec["name"]], 17)
        if gb_cal is None:
            raise ValueError(f"Калібрування бустингу не пройшло train-only критерій: {status}")
    return lr_spec, gb_spec, lr_model, gb_model, lr_cal, gb_cal


def _probabilities(strategy, fitted, data, audit):
    lr_spec, gb_spec, lr_model, gb_model, lr_cal, gb_cal = fitted
    ps = {}
    if lr_model is not None:
        ps["logistic"] = predict(lr_model, data, audit, lr_spec)
    if gb_model is not None:
        ps["boosting"] = predict(gb_model, data, audit, gb_spec)
    if strategy == "logistic_calibrated":
        ps[strategy] = lr_cal.predict_proba(study.logits(ps["logistic"]))[:, 1]
    elif strategy == "boosting_calibrated":
        ps[strategy] = gb_cal.predict_proba(study.logits(ps["boosting"]))[:, 1]
    elif strategy == "blend_50_50":
        ps[strategy] = (ps["logistic"] + ps["boosting"]) / 2
    else:
        ps[strategy] = ps[strategy]
    return ps[strategy]


def _calibration_bins(y, p, bins=10):
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.cut(frame.p, np.linspace(0, 1, bins + 1), include_lowest=True, duplicates="drop")
    result = []
    for interval, group in frame.groupby("bin", observed=True):
        result.append({"bin": str(interval), "rows": len(group),
                       "mean_probability": float(group.p.mean()), "win_rate": float(group.y.mean())})
    return result


def _economic_segments(data, audit, p, segment: pd.Series) -> list[dict]:
    from model_comparison import policy, profit_rows

    actions = policy(data, audit, p)
    profit = profit_rows(data, audit, p)
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    always = .2 * data.amount_usd.to_numpy(dtype=float) * data.won.to_numpy(dtype=int) - 6
    frame = pd.DataFrame({"segment": segment.astype("string").fillna("MISSING").to_numpy(),
                          "won": data.won.to_numpy(dtype=int), "p": p,
                          "eligible": eligible, "fight": actions == "fight",
                          "profit": profit, "always_profit": always})
    result = []
    for name, group in frame.groupby("segment", sort=True, dropna=False):
        scored = group.loc[group.eligible]
        result.append({"segment": str(name), "rows": len(group),
                       "eligible_rows": len(scored), "win_rate": float(group.won.mean()),
                       "mean_probability": float(group.p.mean()),
                       "fight_rows": int(scored.fight.sum()),
                       "fight_rate": float(scored.fight.mean()) if len(scored) else None,
                       "score_usd": float(scored.profit.sum()) if len(scored) else None,
                       "always_score_usd": float(scored.always_profit.sum()) if len(scored) else None,
                       "delta_vs_always_usd": float((scored.profit - scored.always_profit).sum()) if len(scored) else None})
    return result


def evaluate_once(root: Path) -> dict:
    """Evaluate once; save aggregate outputs only, never row-level IDs or predictions."""
    root = Path(root).resolve()
    protocol = freeze_protocol(root)
    if sha256(root / "outputs" / "train_audit.csv") != protocol["train_audit_sha256"]:
        raise ValueError("Train audit змінився після фіксації протоколу")
    selection_path = root / "train_only_outputs" / "selection.json"
    evidence_path = root / "train_only_outputs" / "run_evidence.json"
    if sha256(selection_path) != protocol["selection_sha256"] or sha256(evidence_path) != protocol["train_only_evidence_sha256"]:
        raise ValueError("Train-only вибір або докази змінилися після фіксації протоколу")
    if any(sha256(root / name) != digest for name, digest in protocol["code_sha256"].items()):
        raise ValueError("Код моделі змінився після фіксації фінального протоколу")
    if sha256(root / "requirements_comparison.txt") != protocol["requirements_sha256"]:
        raise ValueError("Залежності змінилися після фіксації фінального протоколу")

    train_raw, train_source_audit, _ = study.load_train_only(root)
    reference = fit_scale_reference(train_raw)
    train, train_audit = prepare_amounts(train_raw, train_source_audit, reference)
    strategy = protocol["selected_strategy"]
    fitted = _fit_strategy(strategy, train_raw, train_source_audit, train, train_audit)

    # Open test only after the strategy and its final train fit are ready.
    test_clean = load_table(root / "outputs", "test_clean.csv")
    test_audit = load_table(root / "outputs", "test_audit.csv")
    test_raw, test_source_audit = _raw_from_audit(test_audit)
    test, test_audit = prepare_amounts(test_raw, test_source_audit, reference)
    if test.dispute_id.tolist() != test_clean.dispute_id.tolist():
        raise ValueError("Test-аудит і очищена test-таблиця не вирівняні за ID")
    for column in ("amount", "amount_usd"):
        if not np.allclose(test[column], test_clean[column], equal_nan=True):
            raise ValueError(f"Повторна обробка test не збігається з експортом: {column}")
    if train.dispute_week.max() >= test.dispute_week.min():
        raise ValueError("Test не є пізнішим за train")
    if set(train.dispute_id) & set(test.dispute_id):
        raise ValueError("Train і test мають спільні dispute_id")

    p = _probabilities(strategy, fitted, test, test_audit)
    classification, economics = evaluate(test, test_audit, p)
    y = test.won.to_numpy(dtype=int)
    margin = np.full(len(test), np.nan)
    eligible = test_audit.score_eligible.to_numpy(dtype=bool)
    amount = test.amount_usd.to_numpy(dtype=float)
    margin[eligible] = p[eligible] * .2 * amount[eligible] - 6
    report = {
        "protocol": protocol,
        "runtime_versions": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "test_clean_sha256": sha256(root / "outputs" / "test_clean.csv"),
        "test_audit_sha256": sha256(root / "outputs" / "test_audit.csv"),
        "strategy": strategy,
        "tuned_logistic_spec": fitted[0], "tuned_boosting_spec": fitted[1],
        "classification": classification,
        "economics": economics,
        "eligible_rows": int(eligible.sum()),
        "review_amount_rows": int((~eligible).sum()),
        "decision_margin": {
            "within_1_usd": int((eligible & (np.abs(margin) <= 1)).sum()),
            "within_2_usd": int((eligible & (np.abs(margin) <= 2)).sum()),
            "calibration_bins": _calibration_bins(y, p),
        },
        "stability_diagnostics": {
            "by_week": _economic_segments(test, test_audit, p, test.dispute_week),
            "merchant_seen_status": _economic_segments(
                test, test_audit, p,
                pd.Series(np.where(test.merchant_id.isna(), "missing",
                                   np.where(test.merchant_id.isin(train.merchant_id.dropna()), "seen", "unseen")),
                          index=test.index)),
        },
        "baseline_comparison": "same eligible rows; score and delta against always-fight reported",
        "row_level_predictions_saved": False,
        "test_used_for_model_selection": False,
        "population_limit": "This dataset includes only historically fought disputes with final outcomes; results do not identify effects for disputes historically not fought.",
    }
    output = root / "final_test_outputs"
    with (output / "test_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report
