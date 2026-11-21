"""Довідкове порівняння на validation (ноутбук 07): параметри лише з train, діагностика розбіжностей.

Validation уже використовували раніше, тому ці числа — не незалежний фінальний тест.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes import train_only_study as train_study
from disputes.backtest import inner_search, train_calibrator
from disputes.config import CATBOOST_SPECS, DECISION_RULE, OUTER_FOLDS, TRAIN_LAST_WEEK
from disputes.economics import break_even_probability, expected_profit, policy, profit_rows
from disputes.evaluation import evaluate, segment_scores
from disputes.models import calibrate, fit, predict
from disputes.preparation import load_table

PROTOCOL = {
    "version": "1.0", "input_splits": ["train", "validation"], "test_opened": False,
    "catboost_search": {"candidates": CATBOOST_SPECS, "folds": OUTER_FOLDS,
                        "choice": "highest summed train-fold Score; pooled log loss; candidate name"},
    "logistic_choice": "highest summed train-fold Score from verified C_grid_summary; log loss; smaller C",
    "calibration": "sigmoid on out-of-time train predictions of selected candidate only; original sample rule",
    "new_models": ["logistic_train_tuned", "logistic_train_tuned_calibrated", "catboost", "catboost_calibrated",
                   "lr_tuned_cat_50_50"],
    "blend": "0.5 uncalibrated train-tuned logistic + 0.5 uncalibrated train-tuned CatBoost",
    "old_validation_rows": "preserve frozen predictions from notebook 04; verify hashes and reproduce their Scores",
    "decision": DECISION_RULE,
    "diagnostics": {"probability_threshold_for_class": .5, "near_economic_boundary_usd": 1.,
                    "pairs": "all model pairs; detailed rows for logistic_C1 vs CatBoost and tuned logistic vs CatBoost",
                    "no_per_row_model_choice_using_outcome": True, "no_weight_or_threshold_search_on_validation": True},
    "practical_margin": "50 USD per 1000 eligible disputes, unchanged; descriptive validation comparison only",
    "validation_status": "previously used for selection, not an independent final test",
}
OUTPUT = "validation_extension_outputs"
DETAILED_PAIRS = {("logistic", "catboost"), ("logistic_train_tuned", "catboost"),
                  ("catboost", "catboost_calibrated"), ("logistic_train_tuned", "lr_tuned_cat_50_50")}
SCENARIOS = ["score_eligible", "score_observed_only"]


def freeze(root):
    """Зафіксувати протокол до читання даних."""
    return proto.freeze(Path(root) / OUTPUT, PROTOCOL, "Протокол доповнення validation")


def access_guard(root):
    """Train і validation дозволені; test і весь початковий датасет — ні."""
    return proto.data_access(
        {Path(root) / "outputs": {"train_clean.csv", "train_audit.csv", "validation_clean.csv", "validation_audit.csv"}},
        forbidden_names={"dataset.csv"}, forbidden_prefixes=("test_", "unknown_time_"))


def fit_train_parameters(root):
    """Повністю завершити вибір параметрів і навчання до відкриття validation."""
    normalized, source_audit, _ = train_study.load_train_only(root)
    prior = proto.read_json(root / train_study.OUTPUT / "run_evidence.json")
    if proto.sha256(root / "outputs" / "train_audit.csv") != prior["train_audit_sha256"]:
        raise ValueError("Train відрізняється від попереднього підбору C")
    verified = lambda name, **kw: proto.verified_csv(root / train_study.OUTPUT / f"{name}.csv",
                                                     prior["result_sha256"][name], **kw)
    ranking = verified("C_grid_summary").sort_values(["score_usd", "log_loss", "C"], ascending=[False, True, True])
    selected_c = float(ranking.C.iloc[0])
    lr_spec = {"name": f"lr_C_{selected_c:g}", "kind": "logistic", "C": selected_c}
    lr_oof = verified("C_grid_predictions", dtype={"dispute_id": "string"}).query("C == @selected_c").copy()
    lr_oof["inner_period"] = lr_oof.outer_period
    lr_oof["base_train_max_week"] = lr_oof.outer_period.str.split("-").str[0].astype(int) - 1
    lr_calibrator, lr_status = train_calibrator(lr_oof, TRAIN_LAST_WEEK)
    cat_spec, cat_candidates, cat_oof, _ = inner_search(normalized, source_audit, OUTER_FOLDS, CATBOOST_SPECS)
    cat_oof = cat_oof[cat_spec["name"]]
    cat_calibrator, cat_status = train_calibrator(cat_oof, TRAIN_LAST_WEEK)
    train = load_table(root / "outputs", "train_clean.csv")
    audit = load_table(root / "outputs", "train_audit.csv")
    if train.dispute_id.tolist() != normalized.dispute_id.tolist():
        raise ValueError("Train ID відрізняються")
    models = {"lr": fit(train, audit, lr_spec), "catboost": fit(train, audit, cat_spec)}
    calibrators = {"lr": lr_calibrator, "catboost": cat_calibrator}
    locked = {"logistic_spec": lr_spec, "catboost_spec": cat_spec,
              "logistic_calibration": lr_status, "catboost_calibration": cat_status,
              "blend_weights": [.5, .5], "threshold_tuned": False, "parameters_selected_without_validation": True}
    return train, models, calibrators, locked, cat_candidates, cat_oof


def pair_details(data, audit, p_a, p_b, name_a, name_b) -> pd.DataFrame:
    """Рядкове порівняння двох моделей: дії, класи при 0,5, дохід і близькість до порогу."""
    a, b = policy(data, audit, p_a), policy(data, audit, p_b)
    amount = data.amount_usd.to_numpy(dtype=float)
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    y = data.won.to_numpy(dtype=int)
    correct_a, correct_b = (p_a >= .5) == y, (p_b >= .5) == y
    frame = data[["dispute_id", "dispute_week", "won", "amount_usd", "psp", "reason_category", "payment_method"]].copy()
    frame["model_a"], frame["model_b"] = name_a, name_b
    frame["p_a"], frame["p_b"] = p_a, p_b
    frame["class_a"], frame["class_b"] = p_a >= .5, p_b >= .5
    frame["action_a"], frame["action_b"] = a, b
    frame["classification_pattern"] = np.select(
        [correct_a & correct_b, correct_a & ~correct_b, ~correct_a & correct_b],
        ["both_correct", "only_a_correct", "only_b_correct"], default="both_wrong")
    frame["decision_group"] = np.select(
        [~eligible, (a == "fight") & (b == "fight"), (a == "fight") & (b == "skip"), (a == "skip") & (b == "fight")],
        ["review_amount", "both_fight", "a_fight_b_skip", "a_skip_b_fight"], default="both_skip")
    frame["score_eligible"] = eligible
    frame["score_observed_only"] = audit.score_observed_only.to_numpy(dtype=bool)
    frame["break_even_p"] = np.where(eligible, break_even_probability(amount), np.nan)
    frame["expected_profit_a"] = np.where(eligible, expected_profit(amount, p_a), np.nan)
    frame["expected_profit_b"] = np.where(eligible, expected_profit(amount, p_b), np.nan)
    frame["profit_a"] = profit_rows(data, audit, p_a)
    frame["profit_b"] = profit_rows(data, audit, p_b)
    frame["delta_b_minus_a_usd"] = frame.profit_b - frame.profit_a
    frame["action_disagreement"] = eligible & (a != b)
    frame["near_boundary"] = eligible & (np.minimum(frame.expected_profit_a.abs(), frame.expected_profit_b.abs()) <= 1.)
    return frame


def pair_summary(details: pd.DataFrame, flag: str) -> dict:
    group = details.loc[details[flag].to_numpy(dtype=bool)]
    different = group.action_disagreement
    a_correct = details.class_a.to_numpy() == details.won.to_numpy()
    b_correct = details.class_b.to_numpy() == details.won.to_numpy()
    correlation = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan
    delta = float(group.delta_b_minus_a_usd.sum())
    assert np.isclose(delta, group.loc[different, "delta_b_minus_a_usd"].sum())
    return {"model_a": details.model_a.iloc[0], "model_b": details.model_b.iloc[0],
            "scenario": flag, "evaluated_rows": len(group), "action_disagreements": int(different.sum()),
            "action_disagreement_rate": float(different.mean()), "delta_b_minus_a_usd": delta,
            "a_fight_b_skip": int(group.decision_group.eq("a_fight_b_skip").sum()),
            "a_skip_b_fight": int(group.decision_group.eq("a_skip_b_fight").sum()),
            "near_boundary_disagreements": int((different & group.near_boundary).sum()),
            "classification_rows": len(details), "class_disagreements_at_0_5": int((details.class_a != details.class_b).sum()),
            "both_correct": int((a_correct & b_correct).sum()), "only_a_correct": int((a_correct & ~b_correct).sum()),
            "only_b_correct": int((~a_correct & b_correct).sum()), "both_wrong": int((~a_correct & ~b_correct).sum()),
            "mean_abs_probability_gap": float((details.p_a - details.p_b).abs().mean()),
            "probability_correlation": correlation(details.p_a, details.p_b),
            "residual_correlation": correlation(details.won - details.p_a, details.won - details.p_b)}


def disagreements(data, audit, probabilities):
    rows, groups, detailed, segments = [], [], [], []
    for a, b in itertools.combinations(probabilities, 2):
        frame = pair_details(data, audit, probabilities[a], probabilities[b], a, b)
        rows += [pair_summary(frame, flag) for flag in SCENARIOS]
        if (a, b) not in DETAILED_PAIRS:
            continue
        detailed.append(frame)
        for flag in SCENARIOS:
            grouped = frame.loc[frame[flag]].groupby("decision_group").agg(
                rows=("won", "size"), win_rate=("won", "mean"), profit_a=("profit_a", "sum"),
                profit_b=("profit_b", "sum"), delta_b_minus_a_usd=("delta_b_minus_a_usd", "sum"))
            groups.append(grouped.reset_index(names="group").assign(model_a=a, model_b=b, scenario=flag))
        for field in ["psp", "reason_category", "payment_method"]:
            eligible = frame.loc[frame.score_eligible]
            grouped = eligible.groupby(eligible[field].astype("string").fillna("MISSING"), sort=False).agg(
                rows=("won", "size"), disagreements=("action_disagreement", "sum"),
                delta_b_minus_a_usd=("delta_b_minus_a_usd", "sum"))
            segments.append(grouped.reset_index(names="group").assign(model_a=a, model_b=b, field=field))
    order = lambda frames, first: pd.concat(frames, ignore_index=True).pipe(
        lambda f: f[first + [c for c in f if c not in first]])
    return (pd.DataFrame(rows), order(groups, ["model_a", "model_b", "scenario", "group"]),
            pd.concat(detailed, ignore_index=True), order(segments, ["model_a", "model_b", "field", "group"]))


def run(root):
    root = Path(root)
    output, protocol_hash = freeze(root)
    with access_guard(root) as opened:
        train, models, calibrators, locked, candidates, oof = fit_train_parameters(root)
        # Параметри й калібратори записані ДО першого читання validation.
        lock = output / "train_parameter_lock.json"
        if lock.exists() and proto.read_json(lock) != proto.canonical(locked):
            raise ValueError("Параметри з train відрізняються від зафіксованого запуску")
        proto.write_json(lock, locked)
        assert not any(name.startswith("validation") for name in opened)
        validation = load_table(root / "outputs", "validation_clean.csv")
        audit = load_table(root / "outputs", "validation_audit.csv")
        assert train.dispute_week.max() < validation.dispute_week.min()
        assert set(train.dispute_id).isdisjoint(validation.dispute_id)
        # Прогнози ноутбука 04 зберігаються як є; їхні контрольні суми зафіксовані першим запуском.
        snapshot = proto.read_json(root / "comparison_outputs" / "first_run_sha256.json")
        old = proto.verified_csv(root / "comparison_outputs" / "validation_predictions.csv",
                                 snapshot["validation_predictions.csv"], dtype={"dispute_id": "string"})
        if old.dispute_id.tolist() != validation.dispute_id.tolist():
            raise ValueError("Історичні validation ID не відповідають поточним")
        np.testing.assert_allclose(old.amount_usd, validation.amount_usd, equal_nan=True)
        np.testing.assert_array_equal(old.won, validation.won)
        historical = [c.removeprefix("p_") for c in old.columns if c.startswith("p_")]
        probabilities = {name: old["p_" + name].to_numpy(dtype=float) for name in historical}
        lr_p = predict(models["lr"], validation, audit, locked["logistic_spec"])
        cat_p = predict(models["catboost"], validation, audit, locked["catboost_spec"])
        probabilities |= {"logistic_train_tuned": lr_p,
                          "logistic_train_tuned_calibrated": calibrate(calibrators["lr"], lr_p),
                          "catboost": cat_p, "catboost_calibrated": calibrate(calibrators["catboost"], cat_p),
                          "lr_tuned_cat_50_50": .5 * lr_p + .5 * cat_p}
        constant = {e["scenario"]: e["score_usd"] for e in evaluate(validation, audit, probabilities["constant_train_rate"])[1]}
        metric_rows, economic_rows, weekly, consolidated = [], [], [], []
        for name, p in probabilities.items():
            metrics, economics = evaluate(validation, audit, p)
            metric_rows.append({"model": name, **metrics})
            row = {"model": name, "historical_row": name in historical, **metrics}
            for e in economics:
                const = constant[e["scenario"]]
                economic_rows.append({"model": name, **e, "constant_usd": const, "delta_vs_constant_usd": e["score_usd"] - const})
                prefix = "" if e["scenario"] == "eligible" else "observed_"
                row |= {prefix + "score_usd": e["score_usd"], prefix + "evaluated_rows": e["rows"],
                        prefix + "delta_always_usd": e["delta_vs_always_usd"],
                        prefix + "delta_constant_usd": e["score_usd"] - const, prefix + "fight_rows": e["fight_rows"]}
            row["uplift_vs_always_percent"] = 100 * economics[0]["delta_vs_always_usd"] / economics[0]["always_usd"]
            weeks = segment_scores(validation, audit, p, validation.dispute_week)
            row |= {f"score_week_{w}": s for w, s in zip(weeks.segment, weeks.score_usd.fillna(0.), strict=True)}
            weekly.append(pd.DataFrame({"model": name, "week": weeks.segment.astype(int), "rows": weeks.eligible_rows,
                                        "score_usd": weeks.score_usd.fillna(0.)}))
            consolidated.append(row)
        economics = pd.DataFrame(economic_rows)
        old_economics = proto.verified_csv(root / "comparison_outputs" / "validation_economics.csv",
                                           snapshot["validation_economics.csv"])
        reproduced = old_economics.merge(economics, on=["model", "scenario"], suffixes=("_old", ""), validate="one_to_one")
        assert np.allclose(reproduced.score_usd, reproduced.score_usd_old)
        assert reproduced.rows.eq(reproduced.rows_old).all()
        pairwise, decision_groups, details, segments = disagreements(validation, audit, probabilities)
        prediction_table = validation[["dispute_id", "won", "amount_usd", "dispute_week"]].assign(
            score_eligible=audit.score_eligible, score_observed_only=audit.score_observed_only,
            **{"p_" + name: p for name, p in probabilities.items()})
        summary = pd.DataFrame(consolidated).sort_values("score_usd", ascending=False)
        tables = {"validation_summary": summary, "validation_metrics": pd.DataFrame(metric_rows),
                  "validation_economics": economics, "validation_weekly": pd.concat(weekly, ignore_index=True),
                  "validation_predictions": prediction_table, "pairwise_disagreement": pairwise,
                  "decision_disagreement_groups": decision_groups, "disagreement_details": details,
                  "disagreement_segments": segments, "train_catboost_candidates": candidates,
                  "train_catboost_oof": oof}
        hashes = proto.write_tables(output, tables)
        models["catboost"].save_model(str(output / "catboost_train_full.cbm"))
        joblib.dump(models["lr"], output / "logistic_train_tuned.joblib")
        joblib.dump(calibrators, output / "train_calibrators.joblib")
        proto.write_json(output / "run_evidence.json", {
            "protocol_sha256": protocol_hash, "csv_open_events": opened,
            "test_opened": False, "validation_previously_used": True, "versions": proto.runtime_versions(),
            "code_sha256": proto.code_hashes("validation_extension.py", "train_only_study.py", "backtest.py", "models.py",
                                             "evaluation.py", "economics.py", "preparation.py", "config.py"),
            "train_parameter_lock_sha256": proto.sha256(lock),
            "prior_validation_predictions_sha256": proto.sha256(root / "comparison_outputs" / "validation_predictions.csv"),
            "result_sha256": hashes})
    print("Validation extension complete; test was not opened", flush=True)
    return summary, pairwise, decision_groups, locked


if __name__ == "__main__":
    proto.main(freeze, run)
