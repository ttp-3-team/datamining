"""Pre-registered second evaluation (notebook 10): improvements chosen on weeks 0–21, one test measurement.

The first evaluation (train weeks 0–17 only) stays unchanged in final_test_outputs/.
Steps:
1. develop: nested temporal validation on train+validation (weeks 0–21) of five recipes:
   logistic regression; + recency weights; + merchant history; + both; CatBoost.
   Selection uses the same practical-tolerance rule; test stays closed.
2. freeze: record the selected recipe, code and dependencies before test is opened.
3. test: evaluate EVERY recipe (each refit on weeks 0–21) and the baseline rules on test once;
   the headline result is the pre-selected recipe, whether it is better or worse.

    uv run python -m disputes.second_evaluation develop|freeze|test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes import train_only_study as v1_study
from disputes.backtest import Window, inner_search, time_window
from disputes.config import (
    BENEFIT_SHARE,
    C_VALUES,
    CATBOOST_SPECS,
    DECISION_RULE,
    INNER_FOLDS,
    SEED,
    VALIDATION_LAST_WEEK,
)
from disputes.economics import payoff
from disputes.evaluation import (
    assert_same_rows,
    bootstrap_interval,
    classification_metrics,
    evaluate,
    merchant_status,
    segment_scores,
    select_from_summary,
    summarize_strategies,
)
from disputes.final_test_evaluation import fit_strategy, strategy_probabilities
from disputes.models import fit, merchant_history, predict
from disputes.paths import PROJECT
from disputes.preparation import SOURCE_COLUMNS, fit_scale_reference, load_table, normalize_source, prepare_amounts

OUTPUT, TEST_OUTPUT = "second_evaluation_outputs", "final_test_outputs_v2"
DEV_OUTER = [(5, 9), (9, 13), (13, 17), (17, 21)]
DEV_INNER = {**INNER_FOLDS, 17: [(9, 13), (13, 17)]}
FINAL_FOLDS = [(9, 13), (13, 17), (17, 21)]
HALF_LIVES = [4., 8.]
COSTS = [4., 6., 8., 10.]


def lr_spec(c: float, half_life: float | None = None, history: bool = False) -> dict:
    name = f"lr_C{c:g}" + (f"_h{half_life:g}" if half_life else "") + ("_merchant" if history else "")
    return {"name": name, "kind": "logistic", "C": c, **({"half_life": half_life} if half_life else {}),
            **({"merchant_history": True} if history else {})}


RECIPES = {
    "logistic": [lr_spec(c) for c in C_VALUES],
    "lr_recency": [lr_spec(c, h) for c in C_VALUES for h in HALF_LIVES],
    "lr_merchant_history": [lr_spec(c, history=True) for c in C_VALUES],
    "lr_merchant_history_recency": [lr_spec(c, h, True) for c in C_VALUES for h in HALF_LIVES],
    "catboost": CATBOOST_SPECS,
}
COMPLEXITY = {"constant_train_rate": 0, "logistic": 1, "lr_recency": 2, "lr_merchant_history": 2,
              "lr_merchant_history_recency": 3, "catboost": 3}
PROTOCOL = {
    "version": "2.0", "development_weeks": [0, VALIDATION_LAST_WEEK],
    "input_allowlist": ["train_audit.csv", "validation_audit.csv"],
    "outer_folds": DEV_OUTER, "inner_folds": DEV_INNER, "final_tuning_folds": FINAL_FOLDS,
    "recipes": RECIPES, "complexity": COMPLEXITY,
    "merchant_history": "smoothed past win rate (prior 20 disputes) and log count; only weeks before each row",
    "recency": "sample weight 0.5 ** ((last train week - week) / half_life), normalised to mean 1",
    "selection": {**v1_study.SELECTION, "complexity": COMPLEXITY},
    "final_model": "selected recipe tuned on final_tuning_folds, refit on all of weeks 0-21",
    "decision": DECISION_RULE, "random_seed": SEED,
    "commitment": ["test scored once for every recipe and the amount-only rule",
                   "headline = recipe selected here, whatever its test result",
                   "first frozen evaluation (+1740 USD, weeks 0-17 model) stays reported unchanged"],
    "known_contamination": "test aggregates of the first evaluation were seen before this protocol",
}


def freeze_development(root):
    """Freeze the protocol before reading data."""
    return proto.freeze(Path(root) / OUTPUT, PROTOCOL, "Second evaluation protocol")


def load_development(root) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and validation (weeks 0–21) restored from the original audit values."""
    audit = pd.concat([load_table(Path(root) / "outputs", f"{part}_audit.csv") for part in ["train", "validation"]],
                      ignore_index=True)
    data, source_audit = normalize_source(audit[["raw__" + c for c in SOURCE_COLUMNS]].rename(
        columns=lambda c: c.removeprefix("raw__")))
    if not data.dispute_week.between(0, VALIDATION_LAST_WEEK).all():
        raise ValueError("Development data contain weeks outside 0–21")
    return data, source_audit


def recency_weights(train: pd.DataFrame, half_life: float) -> np.ndarray:
    weights = 0.5 ** ((train.dispute_week.max() - train.dispute_week.to_numpy(dtype=float)) / half_life)
    return weights / weights.mean()


def with_features(window: Window, spec: dict) -> Window:
    if not spec.get("merchant_history"):
        return window
    return window._replace(train=merchant_history(window.train, window.train),
                           eval=merchant_history(window.train, window.eval))


def recipe_predict(window: Window, spec: dict) -> np.ndarray:
    """Fit a recipe on window.train (with weights and history if set) and predict window.eval."""
    window = with_features(window, spec)
    weights = recency_weights(window.train, spec["half_life"]) if spec.get("half_life") else None
    model = fit(window.train, window.train_audit, spec, sample_weight=weights)
    return predict(model, window.eval, window.eval_audit, spec)


def cost_sensitivity(rows: pd.DataFrame, p_column: str = "p") -> pd.DataFrame:
    """Model and always-fight Score under a different action cost (same benefit share)."""
    rows = rows.loc[rows.eligible]
    result = []
    for cost in COSTS:
        payoff_rows = BENEFIT_SHARE * rows.amount_usd * rows.won - cost
        fight = BENEFIT_SHARE * rows.amount_usd * rows[p_column] > cost
        result.append({"action_cost_usd": cost, "score_usd": float(payoff_rows[fight].sum()),
                       "always_usd": float(payoff_rows.sum()), "fight_share": float(fight.mean())})
    frame = pd.DataFrame(result)
    return frame.assign(delta_vs_always_usd=frame.score_usd - frame.always_usd)


def run_development(root):
    root = Path(root)
    output, protocol_hash = freeze_development(root)
    guard = proto.data_access({root / "outputs": {"train_audit.csv", "validation_audit.csv"}},
                              forbidden_names={"dataset.csv"}, forbidden_prefixes=("test_", "unknown_time_"))
    with guard as opened:
        data, audit = load_development(root)
        frames, choices = [], []
        for cutoff, end in DEV_OUTER:
            training = data.loc[data.dispute_week.le(cutoff)]
            window = time_window(data, audit, cutoff, end)
            probabilities = {"constant_train_rate": np.full(len(window.eval), float(window.train.won.mean()))}
            for recipe, specs in RECIPES.items():
                selected, *_ = inner_search(training, audit.loc[training.index], DEV_INNER[cutoff], specs, recipe_predict)
                probabilities[recipe] = recipe_predict(window, selected)
                choices.append({"outer_period": f"{cutoff + 1}-{end}", "recipe": recipe, **selected})
            frames.append(v1_study.labelled_rows(window, probabilities, f"{cutoff + 1}-{end}"))
            print(f"Development period {cutoff + 1}-{end} done", flush=True)
        predictions = pd.concat(frames, ignore_index=True)
        assert_same_rows(predictions)
        summary, periods = summarize_strategies(predictions, complexity=COMPLEXITY)
        selection = {**select_from_summary(summary), "protocol_sha256": protocol_hash, "test_opened": False}
        sensitivity = pd.concat([cost_sensitivity(g).assign(strategy=name)
                                 for name, g in predictions.groupby("strategy", sort=False)], ignore_index=True)
        tables = {"strategy_summary": summary, "strategy_periods": periods, "outer_choices": pd.DataFrame(choices),
                  "outer_predictions": predictions, "cost_sensitivity": sensitivity}
        hashes = proto.write_tables(output, tables)
        proto.write_json(output / "selection.json", selection)
        proto.write_json(output / "run_evidence.json", {
            "protocol_sha256": protocol_hash, "csv_open_events": opened, "versions": proto.runtime_versions(),
            "code_sha256": proto.code_hashes("second_evaluation.py", "backtest.py", "models.py", "evaluation.py",
                                             "economics.py", "preparation.py", "config.py"),
            "result_sha256": hashes})
    print("Selected recipe:", selection["selected_strategy"], flush=True)
    return summary, selection


def freeze_test(root) -> dict:
    """Freeze the selected recipe, evidence and code before test is opened."""
    root = Path(root)
    selection_path, evidence_path = root / OUTPUT / "selection.json", root / OUTPUT / "run_evidence.json"
    selection = proto.read_json(selection_path)
    protocol = {"version": 2, "selected_recipe": selection["selected_strategy"],
                "selection_sha256": proto.sha256(selection_path), "development_evidence_sha256": proto.sha256(evidence_path),
                "inputs_sha256": {name: proto.sha256(root / "outputs" / name)
                                  for name in ["train_audit.csv", "validation_audit.csv", "test_clean.csv", "test_audit.csv"]},
                "code_sha256": proto.code_hashes("second_evaluation.py", "backtest.py", "models.py", "evaluation.py",
                                                 "economics.py", "preparation.py", "config.py"),
                "lockfile_sha256": proto.sha256(root / "uv.lock"), "final_tuning_folds": FINAL_FOLDS,
                "decision_rule": DECISION_RULE, "commitment": PROTOCOL["commitment"]}
    output = root / TEST_OUTPUT
    output.mkdir(exist_ok=True)
    lock = output / "final_protocol.json"
    if lock.exists():
        if proto.read_json(lock) != proto.canonical(protocol):
            raise ValueError("Frozen second evaluation protocol differs; do not re-evaluate test")
    else:
        proto.write_json(lock, protocol)
    if (output / "test_report.json").exists():
        raise FileExistsError("Second test evaluation already done; re-evaluation is not allowed")
    return protocol


def data_quality(root) -> dict[str, pd.DataFrame]:
    """Aggregate data checks (no rows): volume and win rate by week and reason category."""
    data = pd.concat([load_table(Path(root) / "outputs", f"{part}_clean.csv").assign(split=part)
                      for part in ["train", "validation", "test", "unknown_time"]], ignore_index=True)
    grouped = lambda keys: data.groupby(keys, dropna=False).won.agg(rows="size", win_rate="mean").reset_index()
    return {"weekly_volume": grouped(["split", "dispute_week"]), "reason_categories": grouped(["split", "reason_category"])}


def evaluate_test_once(root) -> dict:
    """Refit every recipe on weeks 0–21 and evaluate once on test together with the baseline rules."""
    root = Path(root)
    protocol = freeze_test(root)
    if not proto.code_hashes_match(protocol["code_sha256"]) or proto.sha256(root / "uv.lock") != protocol["lockfile_sha256"]:
        raise ValueError("Code or dependencies changed after freezing")
    data, audit = load_development(root)
    tuned = {recipe: inner_search(data, audit, FINAL_FOLDS, specs, recipe_predict)[0] for recipe, specs in RECIPES.items()}
    development, development_audit = prepare_amounts(data, audit, fit_scale_reference(data))

    # Test is opened only here, after all recipes are selected and tuned. Same tables as in the first evaluation.
    test, test_audit = load_table(root / "outputs", "test_clean.csv"), load_table(root / "outputs", "test_audit.csv")
    window = Window(development, development_audit, test, test_audit, VALIDATION_LAST_WEEK)
    probabilities = {"constant_train_rate": np.full(len(test), float(development.won.mean()))}
    probabilities |= {recipe: recipe_predict(window, spec) for recipe, spec in tuned.items()}
    # First-evaluation model (train 0–17), for a paired comparison on the same disputes.
    train_raw, train_source_audit, _ = v1_study.load_train_only(root)
    train, train_audit = prepare_amounts(train_raw, train_source_audit, fit_scale_reference(train_raw))
    first = fit_strategy("logistic", train_raw, train_source_audit, train, train_audit)
    probabilities["first_evaluation_logistic"] = strategy_probabilities("logistic", first, test, test_audit)
    frozen = proto.read_json(root / "final_test_outputs" / "test_report.json")["economics"][0]["score_usd"]
    assert np.isclose(evaluate(test, test_audit, probabilities["first_evaluation_logistic"])[1][0]["score_usd"], frozen)

    eligible = test_audit.score_eligible.to_numpy(dtype=bool)
    merchants = test.merchant_id.astype("string").fillna("MISSING").to_numpy()[eligible]
    profit = {name: v1_study.labelled_rows(window, {name: p}, "22-25").profit.to_numpy()[eligible]
              for name, p in probabilities.items()}
    always = payoff(test)[eligible]
    headline = protocol["selected_recipe"]
    rows = []
    for name, p in probabilities.items():
        classification, economics = evaluate(test, test_audit, p)
        vs_always, vs_first = profit[name] - always, profit[name] - profit["first_evaluation_logistic"]
        rows.append({"strategy": name, "headline": name == headline,
                     **{k: classification[k] for k in ["roc_auc", "log_loss", "brier", "accuracy", "precision_won",
                                                       "recall_won", "precision_lost", "recall_lost"]},
                     "score_usd": economics[0]["score_usd"], "always_usd": economics[0]["always_usd"],
                     "delta_vs_always_usd": economics[0]["delta_vs_always_usd"],
                     "delta_vs_always_ci_disputes": bootstrap_interval(vs_always),
                     "delta_vs_always_ci_merchants": bootstrap_interval(vs_always, merchants),
                     "delta_vs_first_evaluation_usd": float(vs_first.sum()),
                     "delta_vs_first_ci_disputes": bootstrap_interval(vs_first),
                     "observed_only_delta_vs_always_usd": economics[1]["delta_vs_always_usd"],
                     "fight_rows": economics[0]["fight_rows"]})
    comparison = pd.DataFrame(rows)
    p = probabilities[headline]
    segments = pd.concat([segment_scores(test, test_audit, p, values).assign(field=field) for field, values in {
        "payment_method": test.payment_method, "reason_category": test.reason_category,
        "amount_band": pd.cut(test.amount_usd, [0, 30, 60, 100, np.inf], include_lowest=True).astype("string"),
        "week": test.dispute_week, "merchant_seen": merchant_status(test, development)}.items()], ignore_index=True)
    report = {"protocol": protocol, "headline_recipe": headline, "tuned_specs": tuned,
              "headline_classification": classification_metrics(test.won, p),
              "comparison": comparison.to_dict(orient="records"),
              "cost_sensitivity": cost_sensitivity(v1_study.labelled_rows(window, {headline: p}, "22-25"))
              .to_dict(orient="records"),
              "eligible_rows": int(eligible.sum()), "review_amount_rows": int((~eligible).sum()),
              "row_level_predictions_saved": False, "test_used_for_selection": False,
              "known_contamination": PROTOCOL["known_contamination"]}
    output = root / TEST_OUTPUT
    interval_columns = [c for c in comparison if "_ci_" in c]
    proto.write_tables(output, {"strategy_comparison": comparison.drop(columns=interval_columns),
                                "headline_segments": segments, **data_quality(root)})
    with (output / "test_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["develop", "freeze", "test"])
    step = parser.parse_args().step
    if step == "develop":
        run_development(PROJECT)
    elif step == "freeze":
        print(freeze_test(PROJECT))
    else:
        report = evaluate_test_once(PROJECT)
        print(pd.DataFrame(report["comparison"])[["strategy", "headline", "score_usd", "delta_vs_always_usd",
                                                   "delta_vs_first_evaluation_usd", "roc_auc", "log_loss"]].to_string())
