"""Взаємодії PSP × валюта / причина в логістичній регресії, лише train (ноутбук 08).

Кожен варіант ознак отримує власний C на внутрішніх зрізах; фіксований C=0,1 —
лише діагностика (його вже обирали на цих самих періодах).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes import train_only_study as previous
from disputes.backtest import inner_search, time_window
from disputes.config import C_VALUES, DECISION_RULE, INNER_FOLDS, OUTER_FOLDS, SEED, THREADS
from disputes.evaluation import assert_same_rows, select_from_summary, summarize_strategies
from disputes.models import INTERACTION_PAIRS, INTERACTION_VARIANTS, features, fit, predict

VARIANTS = INTERACTION_VARIANTS
FIXED = "fixed_C_0_1"
PROTOCOL = {
    "version": "1.0", "input_allowlist": ["train_audit.csv"],
    "variants": VARIANTS, "pairs": INTERACTION_PAIRS, "C_values": C_VALUES,
    "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
    "encoding": "retain base features; add JSON-encoded categorical pairs; train-only one-hot; unknown=ignore",
    "tuning": "highest pooled inner Score for each variant; pooled log loss then smaller C break ties",
    "fixed_C_0_1": "diagnostic only, previously selected on these train periods; excluded from selection",
    "calibration": "none; only interaction hypothesis is changed",
    "selection": {**previous.SELECTION, "complexity": {k: len(v) for k, v in VARIANTS.items()}},
    "decision": DECISION_RULE, "random_seed": SEED, "threads": THREADS,
    "validation_opened": False, "test_opened": False, "limitations": previous.LIMITATIONS,
}
OUTPUT = "interaction_outputs"


class OuterResult(NamedTuple):
    predictions: pd.DataFrame
    choices: pd.DataFrame
    inner: pd.DataFrame
    ranks: pd.DataFrame
    coverage: pd.DataFrame
    coefficients: pd.DataFrame


def freeze(root):
    """Зафіксувати протокол до читання даних."""
    return proto.freeze(Path(root) / OUTPUT, PROTOCOL, "Протокол взаємодій")


def spec(variant: str, c: float) -> dict:
    return {"name": f"{variant}_C{c:g}", "kind": "logistic", "variant": variant, "C": c}


def run_outer(data, audit, cutoff, end) -> OuterResult:
    training = data.loc[data.dispute_week.le(cutoff)]
    searches = {variant: inner_search(training, audit.loc[training.index], INNER_FOLDS[cutoff],
                                      [spec(variant, c) for c in C_VALUES]) for variant in VARIANTS}
    # Усі параметри визначено до створення зовнішньої оцінюваної частини.
    window = time_window(data, audit, cutoff, end)
    period = f"{cutoff + 1}-{end}"
    chosen = {variant: result[0] for variant, result in searches.items()} | {FIXED: spec("baseline", .1)}
    predictions, choices, coverage, coefficients = [], [], [], []
    for variant, model_spec in chosen.items():
        model = fit(window.train, window.train_audit, model_spec)
        p = predict(model, window.eval, window.eval_audit, model_spec)
        predictions.append(previous.labelled_rows(window, {variant: p}, period, column="variant"))
        classifier = model.named_steps["classifier"]
        choices.append({"variant": variant, "outer_period": period, "C": model_spec["C"], "train_rows": len(window.train),
                        "eval_rows": len(window.eval), "encoded_features": classifier.coef_.shape[1]})
        x_train, _ = features(window.train, window.train_audit, model_spec)
        x_eval, _ = features(window.eval, window.eval_audit, model_spec)
        for pair in VARIANTS[model_spec["variant"]]:
            col = "interaction_" + pair
            coverage.append({"variant": variant, "outer_period": period, "pair": pair,
                             "train_combinations": x_train[col].nunique(), "eval_rows": len(window.eval),
                             "unseen_combination_rows": int((~x_eval[col].isin(x_train[col])).sum())})
        names = model.named_steps["preprocess"].get_feature_names_out()
        coefficients += [{"variant": variant, "outer_period": period, "feature": name, "coefficient": float(value)}
                         for name, value in zip(names, classifier.coef_[0], strict=True)]
    inner = pd.concat([result[1] for result in searches.values()], ignore_index=True).assign(outer_period=period)
    ranks = pd.concat([result[3] for result in searches.values()], ignore_index=True).assign(outer_period=period)
    return OuterResult(pd.concat(predictions, ignore_index=True), pd.DataFrame(choices), inner, ranks,
                       pd.DataFrame(coverage), pd.DataFrame(coefficients))


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complexity = {variant: len(pairs) for variant, pairs in VARIANTS.items()} | {FIXED: 0}
    summary, periods = summarize_strategies(predictions, by="variant", complexity=complexity, not_selectable={FIXED})
    for reference, prefix in [("baseline", "vs_baseline"), (FIXED, "vs_fixed_C_0_1")]:
        row = summary.loc[summary.strategy.eq(reference)].iloc[0]
        summary[f"delta_{prefix}_usd"] = summary.score_usd - row.score_usd
    baseline = summary.loc[summary.strategy.eq("baseline")].iloc[0]
    summary["observed_delta_vs_baseline_usd"] = summary.observed_score_usd - baseline.observed_score_usd
    return summary, periods


def run(root):
    root = Path(root)
    output, protocol_hash = freeze(root)
    with proto.train_only_access(root) as opened:
        data, audit, _ = previous.load_train_only(root)
        outcomes = []
        for cutoff, end in OUTER_FOLDS:
            outcomes.append(run_outer(data, audit, cutoff, end))
            print(f"Completed interaction outer weeks {cutoff + 1}-{end}", flush=True)
        predictions, choices, inner, ranks, coverage, coefficients = (
            pd.concat([getattr(r, field) for r in outcomes], ignore_index=True) for field in OuterResult._fields)
        assert_same_rows(predictions, by="variant")
        summary, periods = summarize(predictions)
        # Базовий варіант має відтворювати логістичну регресію ноутбука 05.
        evidence = proto.read_json(root / previous.OUTPUT / "run_evidence.json")
        old_path = root / previous.OUTPUT / "outer_predictions.csv"
        old = proto.verified_csv(old_path, evidence["result_sha256"]["outer_predictions"], dtype={"dispute_id": "string"})
        old = old.loc[old.strategy.eq("logistic")].reset_index(drop=True)
        baseline = predictions.loc[predictions.variant.eq("baseline")].reset_index(drop=True)
        assert old.dispute_id.tolist() == baseline.dispute_id.tolist()
        np.testing.assert_allclose(old.p, baseline.p, atol=1e-10)
        selection = {**select_from_summary(summary), "protocol_sha256": protocol_hash, "validation_opened": False,
                     "test_opened": False, "diagnostic_fixed_C_0_1_excluded": True}
        tables = {"strategy_summary": summary, "strategy_periods": periods, "outer_predictions": predictions,
                  "outer_choices": choices, "inner_candidate_periods": inner, "inner_candidate_ranking": ranks,
                  "interaction_coverage": coverage, "coefficients": coefficients}
        hashes = proto.write_tables(output, tables)
        proto.write_json(output / "selection.json", selection)
        proto.write_json(output / "run_evidence.json", {
            "protocol_sha256": protocol_hash, "csv_open_events": opened, "baseline_predictions_reproduced": True,
            "train_audit_sha256": proto.sha256(root / "outputs" / "train_audit.csv"),
            "prior_predictions_sha256": proto.sha256(old_path), "versions": proto.runtime_versions(),
            "code_sha256": proto.code_hashes("interaction_study.py", "train_only_study.py", "backtest.py", "models.py",
                                             "evaluation.py", "economics.py", "preparation.py", "config.py"),
            "result_sha256": hashes})
    print("Selected:", selection["selected_strategy"], flush=True)
    return summary, periods, choices, coverage, selection


if __name__ == "__main__":
    proto.main(freeze, run)
