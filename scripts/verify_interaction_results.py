"""Перевірити ноутбук 08: повторення, рядковий Score і межі внутрішніх зрізів."""
import pandas as pd

from disputes import protocol as proto
from disputes.config import C_VALUES
from disputes.paths import PROJECT
from disputes.verification import check_executed_notebook, check_repeat, check_rule_scores, write_result

output = PROJECT / "interaction_outputs"
same = check_repeat(output)
final = proto.read_json(output / "run_evidence.json")
assert set(final["csv_open_events"]) == {"train_audit.csv"}
assert proto.sha256(PROJECT / "train_only_outputs" / "outer_predictions.csv") == final["prior_predictions_sha256"]

summary = pd.read_csv(output / "strategy_summary.csv").set_index("strategy")
checks = check_rule_scores(pd.read_csv(output / "outer_predictions.csv"), summary, by="variant")
inner = pd.read_csv(output / "inner_candidate_periods.csv")
assert (inner.inner_train_last < inner.inner_eval_last).all()
assert (inner.inner_eval_last < inner.outer_period.str.split("-").str[0].astype(int)).all()
assert set(inner.C.unique()) == set(C_VALUES)
selection = proto.read_json(output / "selection.json")
assert selection["diagnostic_fixed_C_0_1_excluded"] and selection["selected_strategy"] != "fixed_C_0_1"

write_result(output, {"fresh_runs": 2, "identical_result_sha256": same, "protocol_unchanged": True,
                      "csv_open_events": final["csv_open_events"], "independent_score_checks": checks,
                      "baseline_matches_prior_predictions": final["baseline_predictions_reproduced"],
                      "prior_train_only_predictions_unchanged": True,
                      "executed_code_cells": check_executed_notebook("08_logistic_interactions.ipynb"),
                      "validation_opened": False, "test_opened": False})
