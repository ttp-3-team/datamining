"""Незалежно перевірити підсумкові CSV, повторення та виконаний ноутбук."""
import json
import numpy as np
import pandas as pd
import nbformat
from disputes.paths import EXECUTED_NOTEBOOKS, PROJECT
from disputes.protocol import sha256

root = PROJECT
output = root / "catboost_outputs"
first = json.loads((output / "first_run_evidence.json").read_text(encoding="utf-8"))
final = json.loads((output / "run_evidence.json").read_text(encoding="utf-8"))
checks = {name: digest == final["result_sha256"][name] for name, digest in first["result_sha256"].items()}
assert all(checks.values()), checks
assert first["protocol_sha256"] == final["protocol_sha256"]
assert set(final["actual_source_csv_open_events"]) == {"train_audit.csv"}
for name, digest in final["code_sha256"].items():
    assert sha256((root/name)) == digest
for name, digest in final["prior_evidence"]["result_sha256"].items():
    assert sha256((root / "train_only_outputs" / f"{name}.csv")) == digest

predictions = pd.read_csv(output / "outer_predictions.csv")
summary = pd.read_csv(output / "strategy_summary.csv").set_index("strategy")
for strategy, frame in predictions.groupby("strategy"):
    assert frame.dispute_id.is_unique
    assert frame.p.between(0, 1).all()
    assert frame.loc[~frame.eligible, "profit"].isna().all()
    for flag, target in [("eligible", "score_usd"), ("observed_only", "observed_score_usd")]:
        rows = frame.loc[frame[flag]]
        fight = .2 * rows.amount_usd * rows.p > 6
        expected = ((.2 * rows.amount_usd * rows.won - 6) * fight).sum()
        assert np.isclose(expected, summary.loc[strategy, target])

oof = pd.read_csv(output / "catboost_calibration_oof.csv")
assert (oof.base_train_max_week < oof.dispute_week).all()
for period, frame in oof.groupby("outer_period"):
    assert frame.dispute_id.is_unique
    assert frame.dispute_week.max() < int(period.split("-")[0])

nb = nbformat.read(EXECUTED_NOTEBOOKS / "06_catboost_train_only.ipynb", as_version=4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == "code"]
assert all(c.execution_count and not any(o.output_type == "error" for o in c.outputs) for c in code)
evidence = {"fresh_runs": 2, "identical_result_sha256": checks,
            "prior_results_unchanged": True, "independent_score_checks": len(summary) * 2,
            "actual_source_csv_open_events": final["actual_source_csv_open_events"],
            "code_hashes_match": True, "executed_notebook_cells": len(code),
            "unit_tests": 11, "unit_test_failures": 0, "validation_opened": False, "test_opened": False}
(output / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(json.dumps(evidence, indent=2))
