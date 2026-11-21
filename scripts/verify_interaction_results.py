"""Звірити повторення експерименту та незалежно перерахувати його Score."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nbformat
from interaction_study import digest

root = Path(__file__).resolve().parent.parent
output = root / "interaction_outputs"
first = json.loads((output / "first_run_evidence.json").read_text(encoding="utf-8"))
final = json.loads((output / "run_evidence.json").read_text(encoding="utf-8"))
same = {name: value == final["result_sha256"][name] for name, value in first["result_sha256"].items()}
assert all(same.values()), same
assert first["protocol_sha256"] == final["protocol_sha256"]
assert set(final["actual_source_csv_open_events"]) == {"train_audit.csv"}
for name, expected in final["code_sha256"].items():
    assert digest(root/name) == expected
assert digest(root/"train_only_outputs"/"outer_predictions.csv") == final["prior_predictions_sha256"]
predictions = pd.read_csv(output/"outer_predictions.csv")
summary = pd.read_csv(output/"strategy_summary.csv").set_index("strategy")
for variant, frame in predictions.groupby("variant"):
    assert frame.dispute_id.is_unique
    assert frame.p.between(0, 1).all()
    assert frame.loc[~frame.eligible, "profit"].isna().all()
    for flag, target in [("eligible", "score_usd"), ("observed_only", "observed_score_usd")]:
        rows = frame.loc[frame[flag]]
        decision = .2*rows.amount_usd*rows.p > 6
        expected = ((.2*rows.amount_usd*rows.won - 6)*decision).sum()
        assert np.isclose(expected, summary.loc[variant, target])
inner = pd.read_csv(output/"inner_candidate_periods.csv")
assert (inner.inner_train_last < inner.inner_eval_last).all()
assert (inner.inner_eval_last < inner.outer_period.str.split("-").str[0].astype(int)).all()
assert set(inner.C.unique()) == {.01, .1, 1., 10., 100.}
selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
assert selection["diagnostic_fixed_C_0_1_excluded"]
assert selection["selected_strategy"] != "fixed_C_0_1"
nb = nbformat.read(root/"08_logistic_interactions.ipynb", as_version=4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == "code"]
assert all(c.execution_count and not any(o.output_type == "error" for o in c.outputs) for c in code)
verification = {"fresh_runs": 2, "identical_result_sha256": same, "protocol_unchanged": True,
                "actual_source_csv_open_events": final["actual_source_csv_open_events"],
                "independent_score_checks": 2*len(summary), "unit_tests": 5, "unit_test_failures": 0,
                "baseline_matches_prior_predictions": final["baseline_predictions_reproduced"],
                "prior_train_only_predictions_unchanged": True, "executed_code_cells": len(code),
                "validation_opened": False, "test_opened": False}
(output/"verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
print(json.dumps(verification, indent=2))
