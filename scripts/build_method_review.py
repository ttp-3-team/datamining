"""Build the method review (method_review_outputs/): CSV tables and an HTML report.

Reads only saved study results verified by checksum; raw train/validation/test CSVs are blocked by the audit hook.
Ukrainian report text lives in templates/method_review_text.json.
"""
import json
from pathlib import Path

import jinja2
import numpy as np
import pandas as pd
from markupsafe import Markup

from disputes import protocol as proto
from disputes.evaluation import probability_metrics
from disputes.paths import PROJECT

OUTPUT = PROJECT / "method_review_outputs"
TEMPLATE = Path(__file__).parent / "templates" / "method_review.html.j2"
TEXT = json.loads((Path(__file__).parent / "templates" / "method_review_text.json").read_text(encoding="utf-8"))
LABELS = TEXT["labels"]
REGIMES = TEXT["regimes"]
COLUMN_NAMES = TEXT["columns"]
METRIC_COLUMNS = ["method", "score_usd", "evaluated_rows", "uplift_vs_always_percent", "uplift_vs_constant_percent",
                  "uplift_vs_logistic_percent", "score_per1000_usd", "log_loss", "brier", "roc_auc",
                  "observed_score_usd", "observed_uplift_vs_logistic_percent"]
SUMMARY_COLUMNS = ["evaluation", "method_id", "method", "classification_rows", "evaluated_rows", "score_usd", "always_usd",
                   "constant_usd", "delta_vs_logistic_usd", "uplift_vs_always_percent", "uplift_vs_constant_percent",
                   "uplift_vs_logistic_percent", "score_index_logistic_100", "score_per1000_usd", "log_loss", "brier",
                   "roc_auc", "observed_rows", "observed_score_usd", "observed_delta_vs_logistic_usd",
                   "observed_uplift_vs_logistic_percent", "periods", "periods_above_always", "periods_above_logistic",
                   "worst_delta_vs_logistic_per1000", "evidence_status"]
CONCENTRATION_METHODS = ["catboost", "boosting", "lr_psp_currency"]


def label(method_id: str) -> str:
    return LABELS.get(method_id, method_id)


def relative(frame: pd.DataFrame, baseline_id: str = "logistic") -> pd.DataFrame:
    """Scores relative to always-fight, the constant probability and logistic regression of the same regime."""
    ref = frame.set_index("method_id").loc[baseline_id]
    return frame.assign(
        delta_vs_logistic_usd=frame.score_usd - ref.score_usd,
        uplift_vs_logistic_percent=100 * (frame.score_usd - ref.score_usd) / ref.score_usd,
        uplift_vs_always_percent=100 * (frame.score_usd - frame.always_usd) / frame.always_usd,
        uplift_vs_constant_percent=100 * (frame.score_usd - frame.constant_usd) / frame.constant_usd,
        score_per1000_usd=1000 * frame.score_usd / frame.evaluated_rows,
        score_index_logistic_100=100 * frame.score_usd / ref.score_usd,
        observed_delta_vs_logistic_usd=frame.observed_score_usd - ref.observed_score_usd,
        observed_uplift_vs_logistic_percent=100 * (frame.observed_score_usd - ref.observed_score_usd) / ref.observed_score_usd)


def gain_concentration(predictions: pd.DataFrame) -> pd.DataFrame:
    """Whether the gain over LR comes from a single week or a few large disputes."""
    reference = predictions.loc[predictions.strategy.eq("logistic"), ["dispute_id", "profit"]]
    rows = []
    for method, part in predictions.groupby("strategy"):
        frame = part.merge(reference.rename(columns={"profit": "lr_profit"}), on="dispute_id", validate="one_to_one")
        delta = frame.profit - frame.lr_profit
        by_week = delta.groupby(frame.dispute_week).sum()
        positive = delta[delta > 0].sort_values(ascending=False)
        net, best_gain = float(delta.sum()), float(by_week.max())
        rows.append({"method_id": method, "method": label(method), "net_delta_usd": net,
                     "positive_cases": len(positive), "negative_cases": int((delta < 0).sum()),
                     "largest_positive_case_usd": float(positive.iloc[0]) if len(positive) else 0.,
                     "top3_positive_cases_usd": float(positive.head(3).sum()), "best_week": int(by_week.idxmax()),
                     "best_week_delta_usd": best_gain,
                     "best_week_share_net_percent": 100 * best_gain / net if net > 0 else np.nan,
                     "net_without_best_week_usd": net - best_gain})
    return pd.DataFrame(rows)


def hgb_feature_diagnostics(load) -> pd.DataFrame:
    trials = load("comparison_outputs", "cv_scores.csv")
    rows = []
    for candidate, group in trials.loc[trials.candidate.str.startswith("hgb_")].groupby("candidate"):
        eligible, observed = (group.loc[group.scenario.eq(s)] for s in ["eligible", "observed_only"])
        oof = load("comparison_outputs", f"oof_{candidate}.csv")
        rows.append({"candidate": candidate, "score_usd": eligible.score_usd.sum(), "evaluated_rows": int(eligible.rows.sum()),
                     "observed_score_usd": observed.score_usd.sum(),
                     "always_usd": (eligible.score_usd - eligible.delta_vs_always_usd).sum(),
                     **probability_metrics(oof.won, oof.p),
                     "status": "fixed candidate on reused train folds; feature-search diagnostic"})
    frame = pd.DataFrame(rows)
    return frame.assign(uplift_vs_always_percent=100 * (frame.score_usd - frame.always_usd) / frame.always_usd)


def html_table(frame: pd.DataFrame, columns) -> str:
    existing = [c for c in columns if c in frame]
    return frame[existing].rename(columns=COLUMN_NAMES).to_html(
        index=False, na_rep="—", float_format=lambda x: f"{x:,.3f}", border=0, escape=True)


def build():
    OUTPUT.mkdir(exist_ok=True)
    sources = []
    manifests = {folder: proto.read_json(PROJECT / folder / "run_evidence.json") for folder in
                 ["catboost_outputs", "interaction_outputs", "weighted_weekly_outputs", "validation_extension_outputs"]}

    def load(folder: str, filename: str, verify: bool = True) -> pd.DataFrame:
        """Read a CSV; study tables are checked against the checksum in their run evidence."""
        path = PROJECT / folder / filename
        if verify and folder in manifests:
            frame = proto.verified_csv(path, manifests[folder]["result_sha256"][filename.removesuffix(".csv")])
        else:
            frame = pd.read_csv(path)
        sources.append({"path": path.relative_to(PROJECT).as_posix(), "sha256": proto.sha256(path)})
        return frame

    # The report reads aggregated scores only, never raw train/validation/test rows.
    guard = proto.data_access({PROJECT / "outputs": set(),
                               PROJECT / "validation_extension_outputs": {"validation_summary.csv"}},
                              forbidden_names={"dataset.csv"}, forbidden_prefixes=("test_",))
    with guard:
        interactions = load("interaction_outputs", "strategy_summary.csv").query("strategy != 'baseline'")
        interactions = interactions.assign(method_id="lr_" + interactions.strategy)
        weighted = load("weighted_weekly_outputs", "primary_summary.csv").query("strategy.str.startswith('lr_weight')")
        nested = pd.concat([load("catboost_outputs", "strategy_summary.csv").rename(columns={"strategy": "method_id"}),
                            interactions, weighted.rename(columns={"strategy": "method_id"})], ignore_index=True)
        if nested.method_id.duplicated().any():
            raise ValueError("Duplicate methods in the nested train table")
        nested = relative(nested.assign(evaluation="nested_train_3"))
        weekly = load("weighted_weekly_outputs", "weekly_summary.csv").rename(columns={"strategy": "method_id"})
        weekly = relative(weekly.assign(evaluation="weekly_train_8"))
        validation = load("validation_extension_outputs", "validation_summary.csv")
        renamed = {"logistic": "lr_fixed_C_1", "logistic_train_tuned": "logistic",
                   "logistic_train_tuned_calibrated": "logistic_calibrated", "lr_tuned_cat_50_50": "lr_cat_50_50",
                   "blend_50_50": "historical_lr_hgb_50_50"}
        validation = relative(validation.assign(
            method_id=validation.model.replace(renamed), evaluation="reused_validation",
            always_usd=validation.score_usd - validation.delta_always_usd,
            constant_usd=validation.score_usd - validation.delta_constant_usd,
            classification_rows=validation.rows, observed_rows=validation.observed_evaluated_rows))
        all_methods = pd.concat([nested, weekly, validation], ignore_index=True)
        all_methods["method"] = all_methods.method_id.map(label)
        all_methods["evidence_status"] = np.where(all_methods.evaluation.eq("reused_validation"),
                                                  "already reused validation; historical aggregates only",
                                                  "reused train; no independent final test")
        concentration = gain_concentration(load("weighted_weekly_outputs", "weekly_predictions.csv"))
        cgrid = load("train_only_outputs", "C_grid_summary.csv")
        cgrid = relative(cgrid.assign(method_id=cgrid.C.map(lambda c: f"lr_fixed_C_{c:g}"),
                                      method=cgrid.C.map(lambda c: TEXT["fixed_C_label"].format(C=f"{c:g}")),
                                      evaluation="fixed_C_train_diagnostic"), "lr_fixed_C_1")
        hgb = hgb_feature_diagnostics(load)
        # Written by scripts/verify_weighted_weekly.py, not by the study, so it has no checksum in the run evidence.
        week13 = load("weighted_weekly_outputs", "week13_group_diagnostic.csv", verify=False)
        selections = {regime: proto.read_json(PROJECT / "weighted_weekly_outputs" / f"{regime}_selection.json")
                      for regime in ["primary", "weekly"]}

    # Empty overview cells: the method was not evaluated in that regime.
    methods = pd.Series(list(dict.fromkeys(all_methods.method_id)))
    overview = pd.DataFrame({TEXT["overview"]["method"]: methods.map(label), "method_id": methods})
    for regime, prefix in REGIMES.items():
        part = all_methods.loc[all_methods.evaluation.eq(regime)].set_index("method_id")
        for column in ["score_usd", "uplift_vs_always_percent", "uplift_vs_logistic_percent"]:
            title = TEXT["overview"][column]
            overview[f"{prefix}: {title}"] = overview.method_id.map(part[column])
        if regime == "weekly_train_8":
            overview[TEXT["overview"]["weeks_above_logistic"]] = overview.method_id.map(part.periods_above_logistic)

    tables = {"all_methods_summary": all_methods.reindex(columns=SUMMARY_COLUMNS),
              "weekly_gain_concentration": concentration, "nested_train_table": nested, "weekly_train_table": weekly,
              "validation_table": validation, "C_grid_diagnostics": cgrid, "HGB_feature_diagnostics": hgb,
              "all_methods_overview": overview}
    proto.write_tables(OUTPUT, tables)

    weights = nested.loc[nested.method_id.str.startswith("lr_weight")].sort_values("score_usd", ascending=False)
    best_weight = weights.iloc[0][["method_id", "score_usd", "delta_vs_logistic_usd", "uplift_vs_logistic_percent"]].to_dict()
    weekly_best = weekly.sort_values("score_usd", ascending=False).iloc[0][
        ["method_id", "score_usd", "delta_vs_logistic_usd", "periods_above_logistic"]].to_dict()
    chosen = {regime: selection["selected_strategy"] for regime, selection in selections.items()}
    if chosen["primary"] == chosen["weekly"] == "logistic":
        recommendation = TEXT["recommendation"]["both_logistic"]
    elif chosen["primary"] == chosen["weekly"]:
        recommendation = TEXT["recommendation"]["same_choice"].format(method=label(chosen["primary"]))
    else:
        recommendation = TEXT["recommendation"]["different_choice"]
    focus = concentration.loc[concentration.method_id.isin(CONCENTRATION_METHODS)]
    proto.write_json(OUTPUT / "conclusions.json", {
        "primary_weighted_best": best_weight, "weekly_best": weekly_best,
        "primary_selection": selections["primary"], "weekly_selection": selections["weekly"],
        "no_test_evaluation": True, "recommendation": recommendation,
        "weekly_concentration": focus.to_dict(orient="records")})

    gainers = focus.loc[focus.method_id.isin(["catboost", "lr_psp_currency"])]
    html = jinja2.Template(TEMPLATE.read_text(encoding="utf-8"), autoescape=True).render(
        recommendation=recommendation,
        best_weight={**best_weight, "label": label(best_weight["method_id"])},
        weekly_best={**weekly_best, "label": label(weekly_best["method_id"])},
        nested_rows=int(nested.evaluated_rows.iloc[0]), weekly_rows=int(weekly.evaluated_rows.iloc[0]),
        concentration=[{**row, "label": label(row["method_id"])} for row in gainers.to_dict(orient="records")],
        largest_case=float(gainers.largest_positive_case_usd.max()), sources=sources,
        tables={name: Markup(table) for name, table in {
            "overview": html_table(overview, [c for c in overview if c != "method_id"]),
            "nested": html_table(nested.assign(method=nested.method_id.map(label)).sort_values("score_usd", ascending=False),
                                 METRIC_COLUMNS),
            "weekly": html_table(weekly.assign(method=weekly.method_id.map(label)).sort_values("score_usd", ascending=False),
                                 METRIC_COLUMNS + ["periods_above_always", "periods_above_logistic",
                                                   "worst_delta_vs_logistic_per1000", "sd_delta_vs_logistic_per1000"]),
            "concentration": html_table(concentration.loc[concentration.method_id.isin(
                CONCENTRATION_METHODS + ["lr_weight_amount", "lr_weight_sqrt"])],
                ["method", "net_delta_usd", "positive_cases", "negative_cases", "largest_positive_case_usd",
                 "top3_positive_cases_usd", "best_week", "best_week_delta_usd", "best_week_share_net_percent",
                 "net_without_best_week_usd"]),
            "week13": html_table(week13.loc[week13.delta_vs_logistic_usd.abs().gt(1)],
                                 ["method", "psp", "currency", "rows", "win_rate", "mean_method_p", "mean_lr_p",
                                  "delta_vs_logistic_usd"]),
            "validation": html_table(validation.assign(method=validation.method_id.map(label))
                                     .sort_values("score_usd", ascending=False), METRIC_COLUMNS),
            "cgrid": html_table(cgrid, ["method", "score_usd", "uplift_vs_always_percent", "log_loss", "brier", "roc_auc",
                                        "observed_score_usd"]),
            "hgb": html_table(hgb, ["candidate", "score_usd", "uplift_vs_always_percent", "log_loss", "brier", "roc_auc",
                                    "observed_score_usd"]),
        }.items()})
    (OUTPUT / "method_review.html").write_text(html, encoding="utf-8")
    proto.write_json(OUTPUT / "report_evidence.json", {
        "sources": sources, "test_opened": False, "new_validation_predictions": False,
        "output_sha256": {p.name: proto.sha256(p) for p in sorted(OUTPUT.glob("*")) if p.name != "report_evidence.json"}})
    print(OUTPUT / "method_review.html")


if __name__ == "__main__":
    build()
