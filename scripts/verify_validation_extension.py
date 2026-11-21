"""Verify notebook 07: rerun match, row-level Score, disagreement summaries and saved models."""
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from disputes import protocol as proto
from disputes.models import calibrate, predict
from disputes.paths import PROJECT
from disputes.preparation import load_table
from disputes.validation_extension import access_guard
from disputes.verification import check_executed_notebook, check_repeat, rule_score, write_result

output = PROJECT / "validation_extension_outputs"
with access_guard(PROJECT) as opened:
    same = check_repeat(output)
    first, final = (proto.read_json(output / f"{name}.json") for name in ["first_run_evidence", "run_evidence"])
    assert first["train_parameter_lock_sha256"] == final["train_parameter_lock_sha256"]
    assert first["prior_validation_predictions_sha256"] == proto.sha256(
        PROJECT / "comparison_outputs" / "validation_predictions.csv")
    assert set(final["csv_open_events"]) == {"train_audit.csv", "train_clean.csv", "validation_clean.csv",
                                             "validation_audit.csv"}

    predictions = pd.read_csv(output / "validation_predictions.csv")
    summary = pd.read_csv(output / "validation_summary.csv").set_index("model")
    for name in summary.index:
        for flag, column in [("score_eligible", "score_usd"), ("score_observed_only", "observed_score_usd")]:
            rows = predictions.loc[predictions[flag]]
            assert np.isclose(rule_score(rows, rows["p_" + name]), summary.loc[name, column]), (name, flag)

    details = pd.read_csv(output / "disagreement_details.csv")
    pairs = pd.read_csv(output / "pairwise_disagreement.csv").set_index(["model_a", "model_b", "scenario"])
    focused = details.loc[details.model_a.eq("logistic_train_tuned") & details.model_b.eq("catboost")
                          & details.action_disagreement]
    focused.to_csv(output / "disagreements_lr_vs_cat.csv", index=False)
    for (a, b), frame in details.groupby(["model_a", "model_b"]):
        assert frame.dispute_id.is_unique
        assert frame.loc[~frame.score_eligible, ["profit_a", "profit_b"]].isna().all().all()
        for flag in ["score_eligible", "score_observed_only"]:
            group, expected = frame.loc[frame[flag]], pairs.loc[(a, b, flag)]
            assert np.isclose(group.delta_b_minus_a_usd.sum(), expected.delta_b_minus_a_usd)
            assert group.action_disagreement.sum() == expected.action_disagreements
            assert group.loc[~group.action_disagreement, "delta_b_minus_a_usd"].eq(0).all()

    # Saved models and calibrators reproduce the saved probabilities.
    data = load_table(PROJECT / "outputs", "validation_clean.csv")
    audit = load_table(PROJECT / "outputs", "validation_audit.csv")
    lock = proto.read_json(output / "train_parameter_lock.json")
    calibrators = joblib.load(output / "train_calibrators.joblib")
    lp = predict(joblib.load(output / "logistic_train_tuned.joblib"), data, audit, lock["logistic_spec"])
    cp = predict(CatBoostClassifier().load_model(str(output / "catboost_train_full.cbm")), data, audit, lock["catboost_spec"])
    expected = {"p_logistic_train_tuned": lp, "p_catboost": cp, "p_lr_tuned_cat_50_50": .5 * lp + .5 * cp,
                "p_logistic_train_tuned_calibrated": calibrate(calibrators["lr"], lp),
                "p_catboost_calibrated": calibrate(calibrators["catboost"], cp)}
    for column, values in expected.items():
        np.testing.assert_allclose(values, predictions[column], atol=1e-12)

write_result(output, {"fresh_runs": 2, "identical_result_sha256": same, "train_parameter_lock_identical": True,
                      "historical_validation_predictions_unchanged": True, "score_checks": 2 * len(summary),
                      "detailed_pair_checks": 2 * details.groupby(["model_a", "model_b"]).ngroups,
                      "saved_models_reproduce_probabilities": True,
                      "executed_code_cells": check_executed_notebook("07_validation_and_disagreements.ipynb"),
                      "test_opened": False, "validation_reused": True, "verification_csv_open_events": opened,
                      "focused_disagreement_rows": len(focused),
                      "focused_disagreement_sha256": proto.sha256(output / "disagreements_lr_vs_cat.csv")})
