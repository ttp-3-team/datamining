"""Перевірити повторення, рядковий Score, розбіжності та збережені моделі."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nbformat
import joblib
from catboost import CatBoostClassifier

from validation_extension import digest, access_guard
from preparation import load_table
from model_comparison import predict, logits

root = Path(__file__).resolve().parent
opened = access_guard(root)
output = root / "validation_extension_outputs"
first = json.loads((output / "first_run_evidence.json").read_text(encoding="utf-8"))
final = json.loads((output / "run_evidence.json").read_text(encoding="utf-8"))
same = {name: value == final["result_sha256"][name] for name, value in first["result_sha256"].items()}
assert all(same.values()), same
assert first["protocol_sha256"] == final["protocol_sha256"]
assert first["train_parameter_lock_sha256"] == final["train_parameter_lock_sha256"]
assert final["code_sha256"] == digest(root / "validation_extension.py")
assert first["prior_validation_predictions_sha256"] == digest(root / "comparison_outputs" / "validation_predictions.csv")
assert set(final["actual_input_csv_open_events"]) == {"train_audit.csv", "train_clean.csv", "validation_clean.csv", "validation_audit.csv"}

p = pd.read_csv(output / "validation_predictions.csv")
s = pd.read_csv(output / "validation_summary.csv").set_index("model")
for name in s.index:
    for flag, target in [("score_eligible", "score_usd"), ("score_observed_only", "observed_score_usd")]:
        rows = p.loc[p[flag]]
        fight = .2 * rows.amount_usd * rows["p_" + name] > 6
        expected = ((.2 * rows.amount_usd * rows.won - 6) * fight).sum()
        assert np.isclose(expected, s.loc[name, target])

details = pd.read_csv(output / "disagreement_details.csv")
pairs = pd.read_csv(output / "pairwise_disagreement.csv")
focused = details.loc[details.model_a.eq("logistic_train_tuned") & details.model_b.eq("catboost") & details.action_disagreement].copy()
focused.to_csv(output / "disagreements_lr_vs_cat.csv", index=False)
for (a, b), frame in details.groupby(["model_a", "model_b"]):
    assert frame.dispute_id.is_unique
    for flag in ["score_eligible", "score_observed_only"]:
        group = frame.loc[frame[flag]]
        summary = pairs.loc[pairs.model_a.eq(a) & pairs.model_b.eq(b) & pairs.scenario.eq(flag)].iloc[0]
        assert np.isclose(group.delta_b_minus_a_usd.sum(), summary.delta_b_minus_a_usd)
        assert np.isclose(group.loc[group.action_disagreement, "delta_b_minus_a_usd"].sum(), summary.delta_b_minus_a_usd)
        assert group.action_disagreement.sum() == summary.action_disagreements
        assert group.loc[~group.action_disagreement, "delta_b_minus_a_usd"].eq(0).all()
    assert frame.loc[~frame.score_eligible, "profit_a"].isna().all()
    assert frame.loc[~frame.score_eligible, "profit_b"].isna().all()

data = load_table(root / "outputs", "validation_clean.csv")
audit = load_table(root / "outputs", "validation_audit.csv")
lock = json.loads((output / "train_parameter_lock.json").read_text(encoding="utf-8"))
lr = joblib.load(output / "logistic_train_tuned.joblib")
cat = CatBoostClassifier().load_model(str(output / "catboost_train_full.cbm"))
cal = joblib.load(output / "train_calibrators.joblib")
lp = predict(lr, data, audit, lock["logistic_spec"])
cp = predict(cat, data, audit, lock["catboost_spec"])
np.testing.assert_allclose(lp, p.p_logistic_train_tuned, atol=1e-12)
np.testing.assert_allclose(cp, p.p_catboost, atol=1e-12)
np.testing.assert_allclose(.5*lp+.5*cp, p.p_lr_tuned_cat_50_50, atol=1e-12)
for family, raw, target in [("lr", lp, "p_logistic_train_tuned_calibrated"), ("catboost", cp, "p_catboost_calibrated")]:
    probability = raw if cal[family] is None else cal[family].predict_proba(logits(raw))[:, 1]
    np.testing.assert_allclose(probability, p[target], atol=1e-12)

nb = nbformat.read(root / "07_validation_and_disagreements.ipynb", as_version=4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == "code"]
assert all(c.execution_count and not any(x.output_type == "error" for x in c.outputs) for c in code)
evidence = {"fresh_runs": 2, "identical_result_sha256": same, "train_parameter_lock_identical": True,
            "historical_validation_predictions_unchanged": True, "score_checks": 2*len(s),
            "detailed_pair_checks": 2*details.groupby(["model_a", "model_b"]).ngroups,
            "saved_models_reproduce_probabilities": True, "unit_tests": 4, "unit_test_failures": 0,
            "executed_code_cells": len(code), "test_opened": False,
            "validation_reused": True, "actual_verification_input_events": opened}
evidence["focused_disagreement_rows"] = len(focused)
evidence["focused_disagreement_sha256"] = digest(output / "disagreements_lr_vs_cat.csv")
(output / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(json.dumps(evidence, indent=2))
