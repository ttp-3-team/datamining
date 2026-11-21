"""Verify notebook 09: row-level Score, time boundaries, weights and retraining of two control weeks."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from disputes import protocol as proto
from disputes import weighted_weekly_study as study
from disputes.backtest import time_window
from disputes.models import logits
from disputes.paths import PROJECT
from disputes.verification import check_executed_notebook, check_rule_scores, write_result

output = PROJECT / "weighted_weekly_outputs"
evidence = proto.read_json(output / "run_evidence.json")
assert set(evidence["csv_open_events"]) == {"train_audit.csv"}
assert proto.code_hashes_match(evidence["code_sha256"])
assert all(proto.sha256(output / f"{name}.csv") == digest for name, digest in evidence["result_sha256"].items())

checks, predictions = 0, {}
for regime in ["primary", "weekly"]:
    predictions[regime] = pd.read_csv(output / f"{regime}_predictions.csv", dtype={"dispute_id": "string"})
    summary = pd.read_csv(output / f"{regime}_summary.csv").set_index("strategy")
    checks += check_rule_scores(predictions[regime], summary, by="strategy")
    inner = pd.read_csv(output / f"{regime}_inner_candidates.csv")
    assert (inner.inner_train_last < inner.inner_eval_last).all() and (inner.inner_eval_last <= inner.outer_cutoff).all()
    calibration = pd.read_csv(output / f"{regime}_calibration.csv")
    assert (calibration.calibration_max_week <= calibration.cutoff).all()

# Same cutoff and inner scheme must give the same p for week 10 in both regimes.
primary, weekly = predictions["primary"], predictions["weekly"]
for method in primary.strategy.unique():
    a = primary.loc[primary.strategy.eq(method) & primary.outer_period.eq("10-13") & primary.dispute_week.eq(10)]
    b = weekly.loc[weekly.strategy.eq(method) & weekly.dispute_week.eq(10)]
    assert a.dispute_id.tolist() == b.dispute_id.tolist()
    np.testing.assert_allclose(a.p, b.p, atol=1e-12)

fit_log = pd.read_csv(output / "fit_log.csv")
assert (fit_log.max_fit_week <= fit_log.cutoff).all()
weighted = fit_log.loc[fit_log["mode"].isin(["amount_p99", "sqrt_amount_p99"])]
np.testing.assert_allclose(weighted["mean"], 1., atol=1e-12)
assert (weighted["max"] > 0).all() and (weighted.weight_effective_n <= weighted.rows + 1e-8).all()

with proto.train_only_access(PROJECT) as opened:
    data, audit, _ = study.base.load_train_only(PROJECT)
    # Week 13 gain by PSP × currency: aggregates only, no new rules.
    week13 = time_window(data, audit, 12, 13).eval[["dispute_id", "psp", "currency"]]
    reference = weekly.loc[weekly.strategy.eq("logistic") & weekly.dispute_week.eq(13), ["dispute_id", "profit", "p"]]
    groups = []
    for method in ["catboost", "boosting", "lr_psp_currency"]:
        frame = (weekly.loc[weekly.strategy.eq(method) & weekly.dispute_week.eq(13) & weekly.eligible]
                 .merge(reference.rename(columns={"profit": "lr_profit", "p": "lr_p"}), on="dispute_id", validate="one_to_one")
                 .merge(week13, on="dispute_id", validate="one_to_one"))
        frame["delta"] = frame.profit - frame.lr_profit
        groups.append(frame.groupby(["psp", "currency"], dropna=False).agg(
            rows=("won", "size"), win_rate=("won", "mean"), mean_method_p=("p", "mean"), mean_lr_p=("lr_p", "mean"),
            delta_vs_logistic_usd=("delta", "sum")).reset_index().assign(method=method, week=13))
    (pd.concat(groups, ignore_index=True).sort_values("delta_vs_logistic_usd", ascending=False)
     .to_csv(output / "week13_group_diagnostic.csv", index=False))

    # The first and last weeks are retrained with an empty cache and must reproduce all strategies.
    choices, calibration = (pd.read_csv(output / f"weekly_{name}.csv") for name in ["choices", "calibration"])
    anchors = []
    for week in [10, 17]:
        cache, window = study.ModelCache(), time_window(data, audit, week - 1, week)
        raw = {row.family: cache.predict(window, json.loads(row.spec))
               for row in choices.loc[choices.cutoff.eq(week - 1)].itertuples()}
        probabilities = dict(raw)
        # Calibration from the stored slope and intercept: sigmoid(slope · logit(p) + intercept).
        for row in calibration.loc[calibration.cutoff.eq(week - 1)].itertuples():
            p = raw[row.family]
            probabilities[f"{row.family}_calibrated"] = (
                expit(row.slope * logits(p).ravel() + row.intercept) if row.adequate_by_protocol else p)
        probabilities |= {"constant_train_rate": np.full(len(window.eval), window.train.won.mean()),
                          "blend_50_50": (raw["logistic"] + raw["boosting"]) / 2,
                          "lr_cat_50_50": (raw["logistic"] + raw["catboost"]) / 2,
                          study.FIXED: cache.predict(window, study.lr_specs()[1])}
        assert set(probabilities) == set(study.COMPLEXITY)
        for method, p in probabilities.items():
            old = weekly.loc[weekly.strategy.eq(method) & weekly.dispute_week.eq(week)]
            assert window.eval.dispute_id.tolist() == old.dispute_id.tolist()
            np.testing.assert_allclose(p, old.p, atol=1e-10)
        anchors.append({"week": week, "methods_reproduced": len(probabilities)})
        print(f"Fresh anchor week {week}: all methods reproduced", flush=True)

write_result(output, {"independent_score_checks": checks, "fresh_model_anchor_repeats": anchors,
                      "same_cutoff_horizon_invariance": True, "train_weight_normalization_verified": True,
                      "csv_open_events": opened,
                      "executed_report_cells": check_executed_notebook("09_weighted_weekly_study.ipynb"),
                      "report_reads_verified_results": True, "full_training_runs": 1,
                      "validation_opened": False, "test_opened": False})
