"""Доповнення validation: параметри тільки з train, діагностика розбіжностей."""
from pathlib import Path
import hashlib
import json
import itertools

import numpy as np
import pandas as pd

from preparation import load_table
import train_only_study as train_study
import catboost_study as cat_study
from model_comparison import predict, logits, evaluate, policy, profit_rows

PROTOCOL = {
    "version": "1.0", "input_splits": ["train", "validation"], "test_opened": False,
    "catboost_search": {"candidates": cat_study.SPECS, "folds": train_study.OUTER_FOLDS,
                        "choice": "highest summed train-fold Score; pooled log loss; candidate name"},
    "logistic_choice": "highest summed train-fold Score from verified C_grid_summary; log loss; smaller C",
    "calibration": "sigmoid on out-of-time train predictions of selected candidate only; original sample rule",
    "new_models": ["logistic_train_tuned", "logistic_train_tuned_calibrated", "catboost", "catboost_calibrated", "lr_tuned_cat_50_50"],
    "blend": "0.5 uncalibrated train-tuned logistic + 0.5 uncalibrated train-tuned CatBoost",
    "old_validation_rows": "preserve frozen predictions from notebook 04; verify hashes and reproduce their Scores",
    "decision": "p * 0.2 * amount_usd > 6; missing reliable amount -> review_amount",
    "diagnostics": {"probability_threshold_for_class": .5, "near_economic_boundary_usd": 1.,
                    "pairs": "all model pairs; detailed rows for logistic_C1 vs CatBoost and tuned logistic vs CatBoost",
                    "no_per_row_model_choice_using_outcome": True, "no_weight_or_threshold_search_on_validation": True},
    "practical_margin": "50 USD per 1000 eligible disputes, unchanged; descriptive validation comparison only",
    "validation_status": "previously used for selection, not an independent final test",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(root):
    output = Path(root) / "validation_extension_outputs"
    output.mkdir(exist_ok=True)
    path = output / "protocol.json"
    value = json.loads(json.dumps(PROTOCOL))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Протокол доповнення validation уже зафіксовано й він відрізняється")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
    return output, digest(path)


def access_guard(root):
    import sys
    import os
    inputs = (Path(root) / "outputs").resolve()
    allowed = {"train_clean.csv", "train_audit.csv", "validation_clean.csv", "validation_audit.csv"}
    opened = []
    def hook(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix == ".csv" and (path.name == "dataset.csv" or path.name.startswith(("test_", "unknown_time_"))):
            raise PermissionError("Test / весь початковий датасет не дозволені в цьому експерименті")
        if path.parent == inputs and path.suffix == ".csv":
            if path.name not in allowed:
                raise PermissionError(f"Таблиця {path.name} не дозволена")
            opened.append(path.name)
    sys.addaudithook(hook)
    return opened


def verified_csv(path, expected, **kwargs):
    if digest(path) != expected:
        raise ValueError(f"Контрольна сума {Path(path).name} не збігається")
    return pd.read_csv(path, **kwargs)


def fit_train_parameters(root):
    """Повністю завершити вибір параметрів і навчання до відкриття validation."""
    normalized, source_audit, _ = train_study.load_train_only(root)
    prior = json.loads((root / "train_only_outputs" / "run_evidence.json").read_text(encoding="utf-8"))
    if digest(root / "outputs" / "train_audit.csv") != prior["train_audit_sha256"]:
        raise ValueError("Train відрізняється від попереднього підбору C")
    ranking = verified_csv(root / "train_only_outputs" / "C_grid_summary.csv", prior["result_sha256"]["C_grid_summary"])
    ranking = ranking.sort_values(["score_usd", "log_loss", "C"], ascending=[False, True, True])
    selected_c = float(ranking.C.iloc[0])
    lr_spec = {"name": f"lr_C_{selected_c:g}", "kind": "logistic", "C": selected_c}
    lr_oof = verified_csv(root / "train_only_outputs" / "C_grid_predictions.csv", prior["result_sha256"]["C_grid_predictions"], dtype={"dispute_id": "string"})
    lr_oof = lr_oof.loc[lr_oof.C.eq(selected_c)].copy()
    lr_oof["inner_period"] = lr_oof.outer_period
    lr_oof["base_train_max_week"] = lr_oof.outer_period.str.split("-").str[0].astype(int) - 1
    lr_calibrator, lr_status = train_study.train_calibrator(lr_oof, 17)
    cat_spec, cat_candidates, cat_oof = cat_study.search(normalized, source_audit, train_study.OUTER_FOLDS)
    cat_calibrator, cat_status = train_study.train_calibrator(cat_oof, 17)
    train = load_table(root / "outputs", "train_clean.csv")
    audit = load_table(root / "outputs", "train_audit.csv")
    if train.dispute_id.tolist() != normalized.dispute_id.tolist():
        raise ValueError("Train ID відрізняються")
    lr = train_study.fit(train, audit, lr_spec)
    cat = cat_study.fit_cat(train, audit, cat_spec)
    locked = {"logistic_spec": lr_spec, "catboost_spec": cat_spec,
              "logistic_calibration": lr_status, "catboost_calibration": cat_status,
              "blend_weights": [.5, .5], "threshold_tuned": False,
              "parameters_selected_without_validation": True}
    return train, audit, lr, cat, lr_calibrator, cat_calibrator, locked, cat_candidates, cat_oof


def pair_details(data, audit, p_a, p_b, name_a, name_b):
    a, b = policy(data, audit, p_a), policy(data, audit, p_b)
    amount = data.amount_usd.to_numpy(dtype=float)
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    y = data.won.to_numpy(dtype=int)
    correct_a, correct_b = (p_a >= .5) == y, (p_b >= .5) == y
    groups = np.select([~eligible, (a == "fight") & (b == "fight"), (a == "fight") & (b == "skip"),
                        (a == "skip") & (b == "fight")],
                       ["review_amount", "both_fight", "a_fight_b_skip", "a_skip_b_fight"], default="both_skip")
    patterns = np.select([correct_a & correct_b, correct_a & ~correct_b, ~correct_a & correct_b],
                         ["both_correct", "only_a_correct", "only_b_correct"], default="both_wrong")
    frame = data[["dispute_id", "dispute_week", "won", "amount_usd", "psp", "reason_category", "payment_method"]].copy()
    frame["model_a"], frame["model_b"] = name_a, name_b
    frame["p_a"], frame["p_b"] = p_a, p_b
    frame["class_a"], frame["class_b"] = p_a >= .5, p_b >= .5
    frame["action_a"], frame["action_b"] = a, b
    frame["classification_pattern"], frame["decision_group"] = patterns, groups
    frame["score_eligible"] = eligible
    frame["score_observed_only"] = audit.score_observed_only.to_numpy(dtype=bool)
    frame["break_even_p"] = np.where(eligible, 30 / amount, np.nan)
    frame["expected_profit_a"] = np.where(eligible, .2 * amount * p_a - 6, np.nan)
    frame["expected_profit_b"] = np.where(eligible, .2 * amount * p_b - 6, np.nan)
    frame["profit_a"] = profit_rows(data, audit, p_a)
    frame["profit_b"] = profit_rows(data, audit, p_b)
    frame["delta_b_minus_a_usd"] = frame.profit_b - frame.profit_a
    frame["action_disagreement"] = eligible & (a != b)
    frame["near_boundary"] = eligible & (np.minimum(np.abs(frame.expected_profit_a), np.abs(frame.expected_profit_b)) <= 1.)
    return frame


def pair_summary(details, flag):
    m = details[flag].to_numpy(dtype=bool)
    group = details.loc[m]
    different = group.action_disagreement
    a_correct = details.class_a.to_numpy() == details.won.to_numpy()
    b_correct = details.class_b.to_numpy() == details.won.to_numpy()
    residual_a, residual_b = details.won - details.p_a, details.won - details.p_b
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
            "residual_correlation": correlation(residual_a, residual_b)}


def disagreements(data, audit, probabilities):
    rows, groups, detailed, segments = [], [], [], []
    main = {("logistic", "catboost"), ("logistic_train_tuned", "catboost"),
            ("catboost", "catboost_calibrated"), ("logistic_train_tuned", "lr_tuned_cat_50_50")}
    for a, b in itertools.combinations(probabilities, 2):
        frame = pair_details(data, audit, probabilities[a], probabilities[b], a, b)
        for flag in ["score_eligible", "score_observed_only"]:
            rows.append(pair_summary(frame, flag))
        if (a, b) in main:
            detailed.append(frame)
            for flag in ["score_eligible", "score_observed_only"]:
                for name, g in frame.loc[frame[flag]].groupby("decision_group"):
                    groups.append({"model_a": a, "model_b": b, "scenario": flag, "group": name,
                                   "rows": len(g), "win_rate": g.won.mean(), "profit_a": g.profit_a.sum(),
                                   "profit_b": g.profit_b.sum(), "delta_b_minus_a_usd": g.delta_b_minus_a_usd.sum()})
            for field in ["psp", "reason_category", "payment_method"]:
                labels = frame[field].astype("string").fillna("MISSING")
                for value in labels.unique():
                    g = frame.loc[labels.eq(value) & frame.score_eligible]
                    if len(g):
                        segments.append({"model_a": a, "model_b": b, "field": field, "group": value,
                                         "rows": len(g), "disagreements": int(g.action_disagreement.sum()),
                                         "delta_b_minus_a_usd": float(g.delta_b_minus_a_usd.sum())})
    return pd.DataFrame(rows), pd.DataFrame(groups), pd.concat(detailed, ignore_index=True), pd.DataFrame(segments)


def run(root):
    root = Path(root)
    output, protocol_hash = freeze(root)
    opened = access_guard(root)
    train, train_audit, lr, cat, lr_cal, cat_cal, locked, candidates, oof = fit_train_parameters(root)
    # Параметри й коефіцієнти калібрування записані ДО першого читання validation.
    lock = output / "train_parameter_lock.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != json.loads(json.dumps(locked)):
        raise ValueError("Параметри з train відрізняються від зафіксованого запуску")
    lock.write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")
    assert not any(name.startswith("validation") for name in opened)
    validation = load_table(root / "outputs", "validation_clean.csv")
    audit = load_table(root / "outputs", "validation_audit.csv")
    assert train.dispute_week.max() < validation.dispute_week.min()
    assert set(train.dispute_id).isdisjoint(validation.dispute_id)
    snapshot = json.loads((root / "comparison_outputs" / "first_run_sha256.json").read_text(encoding="utf-8"))
    old = verified_csv(root / "comparison_outputs" / "validation_predictions.csv", snapshot["validation_predictions.csv"], dtype={"dispute_id": "string"})
    if old.dispute_id.tolist() != validation.dispute_id.tolist():
        raise ValueError("Історичні validation ID не відповідають поточним")
    np.testing.assert_allclose(old.amount_usd, validation.amount_usd, equal_nan=True)
    np.testing.assert_array_equal(old.won, validation.won)
    probabilities = {c[2:]: old[c].to_numpy(dtype=float) for c in old.columns if c.startswith("p_")}
    lr_p = predict(lr, validation, audit, locked["logistic_spec"])
    cat_p = predict(cat, validation, audit, locked["catboost_spec"])
    probabilities.update({"logistic_train_tuned": lr_p,
                          "logistic_train_tuned_calibrated": lr_p if lr_cal is None else lr_cal.predict_proba(logits(lr_p))[:, 1],
                          "catboost": cat_p,
                          "catboost_calibrated": cat_p if cat_cal is None else cat_cal.predict_proba(logits(cat_p))[:, 1],
                          "lr_tuned_cat_50_50": .5 * lr_p + .5 * cat_p})
    metric_rows, economic_rows, weekly, consolidated = [], [], [], []
    constant = evaluate(validation, audit, probabilities["constant_train_rate"])[1]
    for name, p in probabilities.items():
        metrics, economics = evaluate(validation, audit, p)
        metric_rows.append({"model": name, **metrics})
        row = {"model": name, "historical_row": name in [c[2:] for c in old.columns if c.startswith("p_")], **metrics}
        for scenario, e in zip(["eligible", "observed_only"], economics):
            const = next(c["score_usd"] for c in constant if c["scenario"] == scenario)
            economic_rows.append({"model": name, **e, "constant_usd": const, "delta_vs_constant_usd": e["score_usd"] - const})
            prefix = "" if scenario == "eligible" else "observed_"
            row.update({prefix + "score_usd": e["score_usd"], prefix + "evaluated_rows": e["rows"],
                        prefix + "delta_always_usd": e["delta_vs_always_usd"], prefix + "delta_constant_usd": e["score_usd"] - const,
                        prefix + "fight_rows": e["fight_rows"]})
            if scenario == "eligible":
                row["uplift_vs_always_percent"] = 100 * e["delta_vs_always_usd"] / e["always_usd"]
        profits = profit_rows(validation, audit, p)
        for week in sorted(validation.dispute_week.unique()):
            mask = (validation.dispute_week.eq(week) & audit.score_eligible).to_numpy(dtype=bool)
            score = float(profits[mask].sum())
            row[f"score_week_{week}"] = score
            weekly.append({"model": name, "week": int(week), "rows": int(mask.sum()), "score_usd": score})
        consolidated.append(row)
    economics = pd.DataFrame(economic_rows)
    old_econ = verified_csv(root / "comparison_outputs" / "validation_economics.csv", snapshot["validation_economics.csv"])
    for _, old_row in old_econ.iterrows():
        now = economics.loc[economics.model.eq(old_row.model) & economics.scenario.eq(old_row.scenario)].iloc[0]
        assert np.isclose(now.score_usd, old_row.score_usd)
        assert now.rows == old_row.rows
    pairwise, decision_groups, details, segments = disagreements(validation, audit, probabilities)
    prediction_table = validation[["dispute_id", "won", "amount_usd", "dispute_week"]].copy()
    prediction_table["score_eligible"] = audit.score_eligible
    prediction_table["score_observed_only"] = audit.score_observed_only
    for name, p in probabilities.items():
        prediction_table["p_" + name] = p
    summary = pd.DataFrame(consolidated).sort_values("score_usd", ascending=False)
    tables = {"validation_summary": summary, "validation_metrics": pd.DataFrame(metric_rows),
              "validation_economics": economics, "validation_weekly": pd.DataFrame(weekly),
              "validation_predictions": prediction_table, "pairwise_disagreement": pairwise,
              "decision_disagreement_groups": decision_groups, "disagreement_details": details,
              "disagreement_segments": segments, "train_catboost_candidates": candidates,
              "train_catboost_oof": oof}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    import joblib
    cat.save_model(str(output / "catboost_train_full.cbm"))
    joblib.dump(lr, output / "logistic_train_tuned.joblib")
    joblib.dump({"lr": lr_cal, "catboost": cat_cal}, output / "train_calibrators.joblib")
    evidence = {"protocol_sha256": protocol_hash, "actual_input_csv_open_events": opened,
                "test_opened": False, "validation_previously_used": True,
                "code_sha256": digest(root / "validation_extension.py"),
                "train_parameter_lock_sha256": digest(lock),
                "prior_validation_predictions_sha256": digest(root / "comparison_outputs" / "validation_predictions.csv"),
                "result_sha256": {name: digest(output / f"{name}.csv") for name in tables}}
    (output / "run_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("Validation extension complete; test was not opened", flush=True)
    return summary, pairwise, decision_groups, locked


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent
    if "--freeze-only" in sys.argv:
        print(freeze(root)[1])
    else:
        run(root)
