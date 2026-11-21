"""Вкладене часове порівняння стратегій. Доступ лише до train_audit.csv."""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from disputes.preparation import SOURCE_COLUMNS, load_table, normalize_source, fit_scale_reference, prepare_amounts
from disputes.model_comparison import CANDIDATES, features, new_model, predict, logits, profit_rows
from disputes.protocol import sha256
from disputes import protocol as proto
from disputes.paths import PROJECT

C_VALUES = [.01, .1, 1., 10., 100.]
LR_SPECS = [{"name": f"lr_C_{c:g}", "kind": "logistic", "C": c} for c in C_VALUES]
GB_SPECS = [dict(s) for s in CANDIDATES if s["kind"] == "boosting"]
OUTER_FOLDS = [(5, 9), (9, 13), (13, 17)]
INNER_FOLDS = {5: [(1, 3), (3, 5)], 9: [(3, 6), (6, 9)], 13: [(5, 9), (9, 13)]}
COMPLEXITY = {"constant_train_rate": 0, "logistic": 1, "logistic_calibrated": 2,
              "boosting": 3, "boosting_calibrated": 4, "blend_50_50": 5}
PROTOCOL = {
    "version": "1.0", "input_allowlist": ["train_audit.csv"],
    "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
    "logistic_C": C_VALUES, "boosting_candidates": GB_SPECS,
    "hyperparameter_choice": "highest summed inner eligible Score; tie: pooled log loss; then smaller C / candidate name",
    "calibration": {"method": "sigmoid on logit out-of-time train predictions", "C": 1.,
                    "minimum_rows": 1000, "minimum_each_class": 50, "minimum_periods": 2,
                    "if_insufficient": "raw probabilities, explicitly flagged; exclude strategy from final selection"},
    "ensemble": "0.5 * uncalibrated tuned logistic + 0.5 * uncalibrated tuned boosting",
    "decision": "p * 0.2 * amount_usd > 6; unavailable amount -> review_amount",
    "selection": {"primary": "pooled eligible Score", "practical_gap_usd_per_1000": 50.,
                  "near_best": "within 0.05 USD * pooled eligible rows of highest Score",
                  "tie_order": ["lower complexity", "higher worst-period uplift per eligible dispute vs constant",
                                "lower SD of period uplift per dispute vs constant", "lower pooled log loss", "strategy name"],
                  "complexity": COMPLEXITY, "include_constant_baseline": True,
                  "observed_only": "reported sensitivity, not an additional tuning objective"},
    "random_seed": 42, "threads": 4, "validation_opened": False, "test_opened": False,
    "limitations": ["earlier EDA and model choices already used these train periods",
                    "validation was used in previous experiments and remains reused data",
                    "no outcome availability date; no verified label maturity",
                    "3 external periods do not establish future stability",
                    "calibration sample criteria are heuristics, not a reliability guarantee"],
}


def freeze_protocol(root):
    """Зафіксувати протокол до читання даних."""
    return proto.freeze(Path(root) / "train_only_outputs", PROTOCOL, "Протокол train-only")


def load_train_only(root):
    filename = "train_audit.csv"
    audit = load_table(Path(root) / "outputs", filename)
    raw = audit[["raw__" + c for c in SOURCE_COLUMNS]].rename(columns=lambda c: c[5:])
    data, source_audit = normalize_source(raw)
    if not data.dispute_id.tolist() == audit.dispute_id.tolist():
        raise ValueError("Відновлені train-ID відрізняються від аудиту")
    if not data.dispute_week.between(0, 17).all():
        raise ValueError("У train є тижні поза 0–17")
    return data, source_audit, [filename]


def install_data_access_guard(root):
    """Python audit hook блокує доступ до інших таблиць, журналює фактичні open."""
    import os
    import sys
    inputs = (Path(root) / "outputs").resolve()
    events = []
    def hook(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.parent == inputs and path.suffix == ".csv":
            if path.name != "train_audit.csv":
                raise PermissionError(f"Train-only протокол забороняє відкривати {path.name}")
            events.append(path.name)
        if path.name in {"dataset.csv", "validation_predictions.csv"} or (
            path.suffix == ".csv" and path.name.startswith(("test_", "validation_", "unknown_time_"))
        ):
            raise PermissionError(f"Train-only протокол забороняє відкривати {path.name}")
    sys.addaudithook(hook)
    return events


def prepare_window(data, source_audit, last_train, last_eval):
    a = data.loc[data.dispute_week.le(last_train)].copy()
    b = data.loc[data.dispute_week.gt(last_train) & data.dispute_week.le(last_eval)].copy()
    if a.empty or b.empty or a.dispute_week.max() >= b.dispute_week.min():
        raise ValueError("Порожній або неправильно впорядкований часовий зріз")
    reference = fit_scale_reference(a)
    ca, aa = prepare_amounts(a, source_audit.loc[a.index], reference)
    cb, ab = prepare_amounts(b, source_audit.loc[b.index], reference)
    return ca, aa, cb, ab


def fit(data, audit, spec):
    x, _ = features(data, audit, spec)
    order = np.argsort(data.dispute_week.to_numpy(), kind="stable")
    model = new_model(spec)
    if spec["kind"] == "logistic":
        model.set_params(classifier__C=spec["C"])
    with warnings.catch_warnings(), threadpool_limits(limits=4):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x.iloc[order], data.won.iloc[order].astype(int))
    return model


def probability_metrics(y, p):
    return {"classification_rows": len(y), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan}


def prediction_rows(data, audit, p, constant_p):
    result = data[["dispute_id", "dispute_week", "won", "amount_usd"]].copy()
    result["p"] = p
    result["eligible"] = audit.score_eligible.to_numpy(dtype=bool)
    result["observed_only"] = audit.score_observed_only.to_numpy(dtype=bool)
    result["profit"] = profit_rows(data, audit, p)
    result["constant_profit"] = profit_rows(data, audit, np.full(len(data), constant_p))
    payoff = .2 * data.amount_usd.to_numpy(dtype=float) * data.won.to_numpy(dtype=int) - 6
    result["always_profit"] = np.where(result.eligible, payoff, np.nan)
    return result.reset_index(drop=True)


def summarize_rows(rows, flag="eligible"):
    mask = rows[flag].to_numpy(dtype=bool)
    result = probability_metrics(rows.won.to_numpy(dtype=int), rows.p.to_numpy(dtype=float))
    if mask.any():
        score = float(rows.loc[mask, "profit"].sum())
        always = float(rows.loc[mask, "always_profit"].sum())
        constant = float(rows.loc[mask, "constant_profit"].sum())
    else:
        score = always = constant = np.nan
    result.update({"evaluated_rows": int(mask.sum()), "score_usd": score,
                   "always_usd": always, "constant_usd": constant,
                   "delta_always_usd": score - always, "delta_constant_usd": score - constant})
    return result


def inner_search(training, training_audit, folds, specs):
    """Не приймає зовнішніх eval-даних навіть як аргумент."""
    rows, oof = [], {s["name"]: [] for s in specs}
    for cutoff, end in folds:
        a, aa, b, ab = prepare_window(training, training_audit, cutoff, end)
        for spec in specs:
            model = fit(a, aa, spec)
            with threadpool_limits(limits=4):
                p = predict(model, b, ab, spec)
            predictions = prediction_rows(b, ab, p, float(a.won.mean()))
            predictions["inner_period"] = f"{cutoff + 1}-{end}"
            predictions["base_train_max_week"] = cutoff
            oof[spec["name"]].append(predictions)
            rows.append({"candidate": spec["name"], "inner_train_last": cutoff, "inner_eval_last": end,
                         **summarize_rows(predictions)})
    oof = {name: pd.concat(parts, ignore_index=True) for name, parts in oof.items()}
    rankings = []
    for spec in specs:
        rankings.append({"candidate": spec["name"], "C": spec.get("C", np.nan),
                         **summarize_rows(oof[spec["name"]])})
    rank = pd.DataFrame(rankings).sort_values(["score_usd", "log_loss", "C", "candidate"],
                                              ascending=[False, True, True, True], na_position="last")
    selected = next(s for s in specs if s["name"] == rank.candidate.iloc[0])
    return selected, pd.DataFrame(rows), oof, rank


def train_calibrator(oof, outer_train_last):
    if not oof.dispute_id.is_unique:
        raise ValueError("Повторні калібрувальні ID")
    if not (oof.base_train_max_week < oof.dispute_week).all():
        raise ValueError("Прогноз для калібратора не є out-of-time")
    if oof.dispute_week.max() > outer_train_last:
        raise ValueError("Калібратор отримав мітки зовнішнього періоду")
    counts = oof.won.value_counts()
    rule = PROTOCOL["calibration"]
    enough = (len(oof) >= rule["minimum_rows"]
              and min(int(counts.get(0, 0)), int(counts.get(1, 0))) >= rule["minimum_each_class"]
              and oof.inner_period.nunique() >= rule["minimum_periods"])
    status = {"calibration_rows": len(oof), "calibration_lost": int(counts.get(0, 0)),
              "calibration_won": int(counts.get(1, 0)), "calibration_periods": int(oof.inner_period.nunique()),
              "calibration_max_week": int(oof.dispute_week.max()), "outer_train_last": outer_train_last,
              "adequate_by_protocol": bool(enough),
              "status": "fitted_heuristic_sample_rule" if enough else "insufficient_sample_raw_fallback"}
    if not enough:
        return None, status
    calibrator = LogisticRegression(C=rule["C"], max_iter=2000, random_state=42)
    calibrator.fit(logits(oof.p), oof.won.astype(int))
    status.update({"slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])})
    return calibrator, status


def select_from_summary(summary):
    """Заздалегідь визначене правило, без p-value та післяфактум зміни порога."""
    candidates = summary.loc[summary.selectable].copy()
    if candidates.empty:
        raise ValueError("Немає придатної стратегії")
    n = int(candidates.evaluated_rows.iloc[0])
    if candidates.evaluated_rows.nunique() != 1:
        raise ValueError("Стратегії оцінені на різних кількостях рядків")
    margin = PROTOCOL["selection"]["practical_gap_usd_per_1000"] / 1000 * n
    maximum = float(candidates.score_usd.max())
    candidates["gap_to_max_usd"] = maximum - candidates.score_usd
    near = candidates.loc[candidates.gap_to_max_usd.le(margin + 1e-9)].sort_values(
        ["complexity", "worst_period_delta_constant_per_row", "period_delta_constant_sd", "log_loss", "strategy"],
        ascending=[True, False, True, True, True])
    return {"selected_strategy": str(near.strategy.iloc[0]), "maximum_score_strategy": str(candidates.loc[candidates.score_usd.idxmax(), "strategy"]),
            "maximum_score_usd": maximum, "practical_margin_usd": margin,
            "evaluated_rows": n, "near_best_strategies": near.strategy.tolist(),
            "selected_gap_to_max_usd": float(near.gap_to_max_usd.iloc[0])}


def run_outer(data, source_audit, cutoff, end, progress=print):
    """Спочатку вибір і калібрування тільки в training; потім зовнішнє оцінювання."""
    training = data.loc[data.dispute_week.le(cutoff)].copy()
    audit = source_audit.loc[training.index].copy()
    lr_spec, lr_results, lr_oof, lr_rank = inner_search(training, audit, INNER_FOLDS[cutoff], LR_SPECS)
    gb_spec, gb_results, gb_oof, gb_rank = inner_search(training, audit, INNER_FOLDS[cutoff], GB_SPECS)
    lr_cal, lr_status = train_calibrator(lr_oof[lr_spec["name"]], cutoff)
    gb_cal, gb_status = train_calibrator(gb_oof[gb_spec["name"]], cutoff)
    # Зовнішні мітки до цього моменту не використовувалися.
    a, aa, b, ab = prepare_window(data, source_audit, cutoff, end)
    period = f"{cutoff + 1}-{end}"
    model_lr, model_gb = fit(a, aa, lr_spec), fit(a, aa, gb_spec)
    with threadpool_limits(limits=4):
        lr_p = predict(model_lr, b, ab, lr_spec)
        gb_p = predict(model_gb, b, ab, gb_spec)
    lr_cp = lr_p if lr_cal is None else lr_cal.predict_proba(logits(lr_p))[:, 1]
    gb_cp = gb_p if gb_cal is None else gb_cal.predict_proba(logits(gb_p))[:, 1]
    probabilities = {"constant_train_rate": np.full(len(b), float(a.won.mean())),
                     "logistic": lr_p, "logistic_calibrated": lr_cp,
                     "boosting": gb_p, "boosting_calibrated": gb_cp,
                     "blend_50_50": .5 * lr_p + .5 * gb_p}
    predictions = []
    for name, p in probabilities.items():
        frame = prediction_rows(b, ab, p, float(a.won.mean()))
        frame["strategy"] = name
        frame["outer_period"] = period
        predictions.append(frame)
    # Окремо фіксована C-сітка на тих самих зовнішніх зрізах: діагностика,
    # не підміна вкладеної стратегії глобальним переможцем цієї сітки.
    grid = []
    for spec in LR_SPECS:
        model = model_lr if spec["C"] == lr_spec["C"] else fit(a, aa, spec)
        with threadpool_limits(limits=4):
            p = predict(model, b, ab, spec)
        frame = prediction_rows(b, ab, p, float(a.won.mean()))
        frame["C"] = spec["C"]
        frame["outer_period"] = period
        grid.append(frame)
    status = pd.DataFrame([{"model": "logistic", **lr_status}, {"model": "boosting", **gb_status}])
    status["outer_period"] = period
    choice = {"outer_period": period, "outer_train_last": cutoff, "outer_eval_last": end,
              "train_rows": len(a), "eval_rows": len(b), "selected_C": lr_spec["C"],
              "selected_boosting": gb_spec["name"], "logistic_spec": lr_spec, "boosting_spec": gb_spec}
    inner = pd.concat([lr_results.assign(family="logistic"), gb_results.assign(family="boosting")])
    inner["outer_period"] = period
    oof = pd.concat([lr_oof[lr_spec["name"]].assign(family="logistic"),
                     gb_oof[gb_spec["name"]].assign(family="boosting")], ignore_index=True)
    oof["outer_period"] = period
    progress(f"Завершено зовнішній період {period}: C={lr_spec['C']}, {gb_spec['name']}", flush=True)
    return pd.concat(predictions, ignore_index=True), pd.concat(grid, ignore_index=True), status, choice, inner, oof


def consolidate(predictions, calibration):
    summaries, periods = [], []
    for strategy, frame in predictions.groupby("strategy", sort=False):
        main, observed = summarize_rows(frame), summarize_rows(frame, "observed_only")
        row = {"strategy": strategy, **main, "observed_rows": observed["evaluated_rows"],
               "observed_score_usd": observed["score_usd"],
               "observed_delta_always_usd": observed["delta_always_usd"],
               "observed_delta_constant_usd": observed["delta_constant_usd"],
               "complexity": COMPLEXITY[strategy], "selectable": True}
        if strategy.endswith("_calibrated"):
            family = "logistic" if strategy.startswith("logistic") else "boosting"
            row["selectable"] = bool(calibration.loc[calibration.model.eq(family), "adequate_by_protocol"].all())
        uplifts = []
        for period, group in frame.groupby("outer_period", sort=False):
            m, o = summarize_rows(group), summarize_rows(group, "observed_only")
            periods.append({"strategy": strategy, "outer_period": period, **m,
                            "observed_rows": o["evaluated_rows"], "observed_score_usd": o["score_usd"],
                            "observed_delta_always_usd": o["delta_always_usd"],
                            "observed_delta_constant_usd": o["delta_constant_usd"]})
            row[f"score_weeks_{period}"] = m["score_usd"]
            row[f"rows_weeks_{period}"] = m["evaluated_rows"]
            uplifts.append(m["delta_constant_usd"] / m["evaluated_rows"])
        row["worst_period_delta_constant_per_row"] = min(uplifts)
        row["period_delta_constant_sd"] = float(np.std(uplifts))
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("score_usd", ascending=False)
    # Порівнюємо ідентичні спори й суми між усіма стратегіями.
    canonical = None
    for _, frame in predictions.groupby("strategy"):
        view = frame[["dispute_id", "outer_period", "amount_usd", "eligible", "observed_only"]].reset_index(drop=True)
        if canonical is None:
            canonical = view
        else:
            pd.testing.assert_frame_equal(view, canonical)
    return summary, pd.DataFrame(periods)


def run(root):
    root = Path(root)
    output, protocol_hash = freeze_protocol(root)
    observed_access = install_data_access_guard(root)
    data, source_audit, access_log = load_train_only(root)
    runs = [run_outer(data, source_audit, cutoff, end) for cutoff, end in OUTER_FOLDS]
    predictions, grid, calibration, inner, calibration_oof = [pd.concat([r[i] for r in runs], ignore_index=True)
                                                             for i in [0, 1, 2, 4, 5]]
    choices = [r[3] for r in runs]
    summary, periods = consolidate(predictions, calibration)
    grid_rows, grid_periods = [], []
    for c, frame in grid.groupby("C"):
        main, observed = summarize_rows(frame), summarize_rows(frame, "observed_only")
        row = {"C": c, **main, "observed_score_usd": observed["score_usd"], "observed_rows": observed["evaluated_rows"]}
        for period, group in frame.groupby("outer_period", sort=False):
            result = summarize_rows(group)
            row[f"score_weeks_{period}"] = result["score_usd"]
            grid_periods.append({"C": c, "outer_period": period, **result})
        grid_rows.append(row)
    grid_summary = pd.DataFrame(grid_rows).sort_values("score_usd", ascending=False)
    selection = select_from_summary(summary)
    selection.update({"protocol_sha256": protocol_hash, "outer_choices": choices,
                      "opened_data_tables": access_log, "validation_opened": False, "test_opened": False,
                      "claim": "train-only model selection; no independently confirmed winner"})
    tables = {"strategy_summary": summary, "strategy_periods": periods, "C_grid_summary": grid_summary,
              "C_grid_periods": pd.DataFrame(grid_periods), "calibration_status": calibration,
              "inner_candidate_periods": inner, "outer_predictions": predictions,
              "C_grid_predictions": grid, "calibration_oof": calibration_oof,
              "outer_choices": pd.DataFrame([{k: v for k, v in c.items() if not k.endswith("_spec")} for c in choices])}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    import sklearn
    import sys
    evidence = {"protocol_sha256": protocol_hash, "opened_data_tables": access_log,
                "observed_input_csv_open_events": observed_access,
                "train_audit_sha256": sha256((root / "outputs" / "train_audit.csv")),
                "code_sha256": proto.code_hashes("train_only_study.py", "model_comparison.py", "preparation.py"),
                "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__},
                "result_sha256": {name: sha256((output / f"{name}.csv")) for name in tables}}
    (output / "run_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("Обрана стратегія:", selection["selected_strategy"], flush=True)
    return summary, periods, grid_summary, calibration, selection


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    root = PROJECT
    if args.freeze_only:
        _, digest = freeze_protocol(root)
        print("Protocol frozen:", digest)
    else:
        run(root)
