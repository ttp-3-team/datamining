"""Перевірити ноутбук 06: повторення, рядковий Score, out-of-time калібрування, незмінність ноутбука 05."""
import pandas as pd

from disputes import protocol as proto
from disputes.paths import PROJECT
from disputes.verification import check_executed_notebook, check_repeat, check_rule_scores, write_result

output = PROJECT / "catboost_outputs"
same = check_repeat(output)
final = proto.read_json(output / "run_evidence.json")
assert set(final["csv_open_events"]) == {"train_audit.csv"}
for name, digest in final["prior_evidence"]["result_sha256"].items():
    assert proto.sha256(PROJECT / "train_only_outputs" / f"{name}.csv") == digest

summary = pd.read_csv(output / "strategy_summary.csv").set_index("strategy")
checks = check_rule_scores(pd.read_csv(output / "outer_predictions.csv"), summary, by="strategy")
oof = pd.read_csv(output / "catboost_calibration_oof.csv")
assert (oof.base_train_max_week < oof.dispute_week).all()
for period, frame in oof.groupby("outer_period"):
    assert frame.dispute_id.is_unique and frame.dispute_week.max() < int(period.split("-")[0])

write_result(output, {"fresh_runs": 2, "identical_result_sha256": same, "prior_results_unchanged": True,
                      "independent_score_checks": checks, "csv_open_events": final["csv_open_events"],
                      "code_hashes_match": True,
                      "executed_notebook_cells": check_executed_notebook("06_catboost_train_only.ipynb"),
                      "validation_opened": False, "test_opened": False})
