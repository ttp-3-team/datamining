"""Логістична регресія з вагами за сумою та щотижневий вкладений backtest, лише train (ноутбук 09).

Два режими: «primary» — ті самі три зовнішні періоди, що й у ноутбуці 05; «weekly» —
кожен тиждень 10–17 оцінюється моделями, навченими на попередніх тижнях.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes import train_only_study as base
from disputes.backtest import Window, inadequate_calibrations, inner_search, time_window, train_calibrator
from disputes.config import C_VALUES, CATBOOST_SPECS, HGB_SPECS, INNER_FOLDS, OUTER_FOLDS, SEED, THREADS
from disputes.evaluation import assert_same_rows, select_from_summary, summarize_strategies
from disputes.models import INTERACTION_VARIANTS, calibrate, fit, predict

WEIGHT_MODES = ["none", "amount_p99", "sqrt_amount_p99"]
WEEKLY_WEEKS = list(range(10, 18))
WEEKLY_INNER = {t: [(t - 6, t - 3), (t - 3, t)] for t in range(9, 17)}
FIXED = "lr_fixed_C_0_1"
COMPLEXITY = {"constant_train_rate": 0, "logistic": 1, "logistic_calibrated": 2,
              "lr_weight_amount": 2, "lr_weight_amount_calibrated": 3,
              "lr_weight_sqrt": 2, "lr_weight_sqrt_calibrated": 3,
              "boosting": 3, "boosting_calibrated": 4, "catboost": 3, "catboost_calibrated": 4,
              "lr_psp_currency": 2, "lr_psp_reason": 2, "lr_both_interactions": 3,
              "blend_50_50": 5, "lr_cat_50_50": 5, FIXED: 0}
PROTOCOL = {
    "version": "1.0", "source_allowlist": ["train_audit.csv"],
    "weights": {"modes": WEIGHT_MODES, "cap_quantile": .99,
                "missing": "use median positive training amount for importance weight only",
                "normalization": "divide weights by their training mean",
                "preprocessing": "unweighted; weights go only to classifier loss",
                "score": "original unweighted monetary formula; original cleaned USD amounts"},
    "C_values": C_VALUES, "primary_outer_folds": OUTER_FOLDS,
    "primary_inner_folds": INNER_FOLDS, "weekly_eval_weeks": WEEKLY_WEEKS,
    "weekly_inner_folds": WEEKLY_INNER, "weekly_methods": list(COMPLEXITY),
    "hgb_candidates": HGB_SPECS, "catboost_candidates": CATBOOST_SPECS,
    "interactions": INTERACTION_VARIANTS,
    "parameter_selection": "highest summed inner raw Score; pooled log loss; C or candidate name",
    "calibration": base.PROTOCOL["calibration"],
    "selection": {**base.SELECTION, "complexity": COMPLEXITY},
    "fixed_C_0_1": "previously selected diagnostic only; not selectable",
    "blends": "fixed 50/50 uncalibrated tuned logistic with HGB / CatBoost",
    "weekly_stability": "per-week uplift vs always, constant, and same-week logistic; counts and USD/1000",
    "no_class_balance": True, "seed": SEED, "threads": THREADS,
    "validation_opened": False, "test_opened": False,
    "limitations": base.LIMITATIONS + [
        "weekly cases overlap previously evaluated weeks 10-17; not new independent data",
        "training windows overlap and outcomes may not have matured by next week",
        "clipped amount weighting is a hypothesis, not direct optimization of Score"],
}
OUTPUT = "weighted_weekly_outputs"


def freeze(root):
    """Зафіксувати протокол до читання даних."""
    return proto.freeze(Path(root) / OUTPUT, PROTOCOL, "Протокол зваженого навчання")


def sample_weights(data: pd.DataFrame, mode: str) -> tuple[np.ndarray, dict]:
    """Ваги за сумою, обрізані на p99 train і нормовані до середнього 1; без суми — медіана."""
    amount = data.amount_usd.to_numpy(dtype=float)
    valid = np.isfinite(amount) & (amount > 0)
    if mode == "none":
        return np.ones(len(data)), {"mode": mode, "rows": len(data), "mean": 1., "max": 1.,
                                    "weight_effective_n": float(len(data))}
    if mode not in WEIGHT_MODES or not valid.any():
        raise ValueError("Немає опори для ваг або невідомий режим")
    median, cap = float(np.median(amount[valid])), float(np.quantile(amount[valid], .99))
    values = np.minimum(np.where(valid, amount, median), cap)
    if mode == "sqrt_amount_p99":
        values = np.sqrt(values)
    weights = values / values.mean()
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Некоректні ваги")
    return weights, {"mode": mode, "rows": len(data), "median_amount": median, "cap_amount": cap,
                     "missing_weight_rows": int((~valid).sum()), "mean": float(weights.mean()),
                     "max": float(weights.max()), "weight_effective_n": float(weights.sum() ** 2 / (weights ** 2).sum())}


def lr_specs(mode: str = "none", variant: str = "baseline") -> list[dict]:
    return [{"name": f"lr_{variant}_{mode}_C{c:g}", "kind": "logistic", "variant": variant, "weight": mode, "C": c}
            for c in C_VALUES]


class ModelCache:
    """Одна модель на (останній тиждень навчання, специфікацію): сімейства й режими її не перенавчають."""

    def __init__(self):
        self.models, self.fit_log = {}, []

    def model(self, window: Window, spec: dict):
        key = (window.last_train, json.dumps(spec, sort_keys=True))
        if key not in self.models:
            weights, stats = (None, {})
            if spec["kind"] == "logistic":
                weights, stats = sample_weights(window.train, spec["weight"])
                weights = None if spec["weight"] == "none" else weights
            self.models[key] = fit(window.train, window.train_audit, spec, sample_weight=weights)
            self.fit_log.append({"cutoff": window.last_train, "max_fit_week": int(window.train.dispute_week.max()),
                                 "candidate": spec["name"], **stats})
        return self.models[key]

    def predict(self, window: Window, spec: dict) -> np.ndarray:
        return predict(self.model(window, spec), window.eval, window.eval_audit, spec)


def evaluate_cutoff(cache: ModelCache, data, audit, cutoff, end, inner_folds, all_methods):
    families = {"logistic": lr_specs(), "lr_weight_amount": lr_specs("amount_p99"),
                "lr_weight_sqrt": lr_specs("sqrt_amount_p99")}
    if all_methods:
        families |= {"boosting": HGB_SPECS, "catboost": CATBOOST_SPECS}
        families |= {f"lr_{v}": lr_specs(variant=v) for v in ["psp_currency", "psp_reason", "both_interactions"]}
    training = data.loc[data.dispute_week.le(cutoff)]
    window = time_window(data, audit, cutoff, end)
    probabilities, choices, calibration, inner = {}, [], [], []
    for family, specs in families.items():
        selected, records, oof, _ = inner_search(training, audit.loc[training.index], inner_folds, specs, cache.predict)
        raw = probabilities[family] = cache.predict(window, selected)
        choices.append({"family": family, "cutoff": cutoff, "eval_end": end, "spec": json.dumps(selected, sort_keys=True)})
        inner.append(records.assign(family=family, outer_cutoff=cutoff, outer_eval_end=end))
        if family not in {"lr_psp_currency", "lr_psp_reason", "lr_both_interactions"}:
            calibrator, status = train_calibrator(oof[selected["name"]], cutoff)
            probabilities[f"{family}_calibrated"] = calibrate(calibrator, raw)
            calibration.append({"family": family, "cutoff": cutoff, "eval_end": end, **status})
    probabilities["constant_train_rate"] = np.full(len(window.eval), window.train.won.mean())
    if all_methods:
        probabilities["blend_50_50"] = .5 * probabilities["logistic"] + .5 * probabilities["boosting"]
        probabilities["lr_cat_50_50"] = .5 * probabilities["logistic"] + .5 * probabilities["catboost"]
        probabilities[FIXED] = cache.predict(window, lr_specs()[1])
    predictions = base.labelled_rows(window, probabilities, f"{cutoff + 1}-{end}")
    return predictions, pd.DataFrame(choices), pd.DataFrame(calibration), pd.concat(inner, ignore_index=True)


def summarize(predictions: pd.DataFrame, calibration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, periods = summarize_strategies(
        predictions, complexity=COMPLEXITY,
        not_selectable={FIXED} | inadequate_calibrations(calibration, column="family"))
    reference = periods.loc[periods.strategy.eq("logistic"), ["outer_period", "score_usd"]]
    periods = periods.merge(reference.rename(columns={"score_usd": "logistic_usd"}), on="outer_period",
                            validate="many_to_one")
    periods["delta_vs_logistic_usd"] = periods.score_usd - periods.logistic_usd
    periods["delta_vs_logistic_per1000"] = 1000 * periods.delta_vs_logistic_usd / periods.evaluated_rows
    stability = periods.groupby("strategy").agg(
        periods=("outer_period", "size"), periods_above_always=("delta_always_usd", lambda s: int(s.gt(0).sum())),
        periods_above_logistic=("delta_vs_logistic_usd", lambda s: int(s.gt(1e-8).sum())),
        worst_delta_vs_logistic_per1000=("delta_vs_logistic_per1000", "min"),
        sd_delta_vs_logistic_per1000=("delta_vs_logistic_per1000", lambda s: s.std(ddof=0)))
    summary = summary.merge(stability, left_on="strategy", right_index=True, how="left")
    logistic = summary.loc[summary.strategy.eq("logistic"), "score_usd"].iloc[0]
    summary["delta_vs_logistic_usd"] = summary.score_usd - logistic
    summary["uplift_vs_always_percent"] = 100 * summary.delta_always_usd / summary.always_usd
    summary["uplift_vs_constant_percent"] = 100 * summary.delta_constant_usd / summary.constant_usd
    summary["uplift_vs_logistic_percent"] = 100 * summary.delta_vs_logistic_usd / logistic
    summary["score_per1000_usd"] = 1000 * summary.score_usd / summary.evaluated_rows
    return summary.sort_values("score_usd", ascending=False), periods


def run(root):
    root = Path(root)
    output, protocol_hash = freeze(root)
    with proto.train_only_access(root) as opened:
        data, audit, _ = base.load_train_only(root)
        cache, tables = ModelCache(), {}
        regimes = {"primary": [(cutoff, end, INNER_FOLDS[cutoff]) for cutoff, end in OUTER_FOLDS],
                   "weekly": [(w - 1, w, WEEKLY_INNER[w - 1]) for w in WEEKLY_WEEKS]}
        for regime, windows in regimes.items():
            results = []
            for cutoff, end, folds in windows:
                results.append(evaluate_cutoff(cache, data, audit, cutoff, end, folds, regime == "weekly"))
                print(f"Completed {regime}: train through {cutoff}, evaluate through {end}", flush=True)
            predictions, choices, calibration, inner = (pd.concat([r[i] for r in results], ignore_index=True)
                                                        for i in range(4))
            assert_same_rows(predictions)
            summary, periods = summarize(predictions, calibration)
            tables |= {f"{regime}_{name}": frame for name, frame in
                       {"predictions": predictions, "choices": choices, "calibration": calibration,
                        "inner_candidates": inner, "summary": summary, "periods": periods}.items()}
            if regime == "primary":
                # Логістична регресія та її калібрування мають відтворювати ноутбук 05.
                evidence = proto.read_json(root / base.OUTPUT / "run_evidence.json")
                old = proto.verified_csv(root / base.OUTPUT / "outer_predictions.csv",
                                         evidence["result_sha256"]["outer_predictions"], dtype={"dispute_id": "string"})
                for method in ["logistic", "logistic_calibrated"]:
                    current = predictions.loc[predictions.strategy.eq(method)].reset_index(drop=True)
                    historical = old.loc[old.strategy.eq(method)].reset_index(drop=True)
                    assert current.dispute_id.tolist() == historical.dispute_id.tolist()
                    np.testing.assert_allclose(current.p, historical.p, atol=1e-10)
            proto.write_json(output / f"{regime}_selection.json", select_from_summary(summary))
        tables["fit_log"] = pd.DataFrame(cache.fit_log)
        hashes = proto.write_tables(output, tables)
        proto.write_json(output / "run_evidence.json", {
            "protocol_sha256": protocol_hash, "csv_open_events": opened, "baseline_predictions_reproduced": True,
            "validation_opened": False, "test_opened": False, "versions": proto.runtime_versions(),
            "code_sha256": proto.code_hashes("weighted_weekly_study.py", "train_only_study.py", "backtest.py",
                                             "models.py", "evaluation.py", "economics.py", "preparation.py", "config.py"),
            "result_sha256": hashes})
    return tables["primary_summary"], tables["weekly_summary"], tables["weekly_periods"]


if __name__ == "__main__":
    proto.main(freeze, run)
