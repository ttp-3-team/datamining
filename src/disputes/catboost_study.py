"""CatBoost у тій самій вкладеній train-only схемі, без зміни попередніх результатів."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from disputes import train_only_study as previous
from disputes.model_comparison import features, predict, logits
from disputes.protocol import sha256
from disputes import protocol as proto
from disputes.paths import PROJECT

SPECS = [{"name": f"cat_d{depth}_l2_{l2}", "kind": "catboost", "depth": depth, "l2": l2}
         for depth in [4, 6] for l2 in [3., 10.]]
COMPLEXITY = {**previous.COMPLEXITY, "catboost": 3, "catboost_calibrated": 4, "lr_cat_50_50": 5}
PROTOCOL = {
    "version": "1.0", "outer_folds": previous.OUTER_FOLDS, "inner_folds": previous.INNER_FOLDS,
    "candidates": SPECS, "base_features": "same categorical features and amount_usd as logistic regression",
    "parameters": {"iterations": 500, "learning_rate": .05, "boosting_type": "Ordered",
                   "has_time": True, "thread_count": 4, "random_seed": 42, "early_stopping": False},
    "tuning": "highest pooled inner eligible Score; tie: pooled log loss then candidate name",
    "calibration": previous.PROTOCOL["calibration"], "blend": "0.5 raw tuned logistic + 0.5 raw tuned CatBoost",
    "selection": {**previous.PROTOCOL["selection"], "complexity": COMPLEXITY},
    "baselines": "verified frozen train-only out-of-time predictions from notebook 05",
    "input_tables": ["outputs/train_audit.csv", "train_only_outputs/outer_predictions.csv"],
    "validation_opened": False, "test_opened": False,
    "limitations": previous.PROTOCOL["limitations"] + ["within-week event order is unknown",
        "ordered boosting does not establish outcome availability at prediction time"],
}


def freeze(root):
    """Зафіксувати протокол до читання даних."""
    return proto.freeze(Path(root) / "catboost_outputs", PROTOCOL, "Протокол CatBoost")


def fit_cat(data, audit, spec):
    x, cats = features(data, audit, spec)
    order = np.argsort(data.dispute_week.to_numpy(), kind="stable")
    model = CatBoostClassifier(iterations=500, learning_rate=.05, depth=spec["depth"],
                              l2_leaf_reg=spec["l2"], loss_function="Logloss", boosting_type="Ordered",
                              has_time=True, thread_count=4, random_seed=42,
                              allow_writing_files=False, verbose=False)
    model.fit(x.iloc[order], data.won.iloc[order].astype(int), cat_features=cats)
    return model


def search(training, audit, folds):
    records, oof = [], {s["name"]: [] for s in SPECS}
    for cutoff, end in folds:
        a, aa, b, ab = previous.prepare_window(training, audit, cutoff, end)
        for spec in SPECS:
            model = fit_cat(a, aa, spec)
            p = predict(model, b, ab, spec)
            frame = previous.prediction_rows(b, ab, p, float(a.won.mean()))
            frame["inner_period"] = f"{cutoff + 1}-{end}"
            frame["base_train_max_week"] = cutoff
            oof[spec["name"]].append(frame)
            records.append({"candidate": spec["name"], "inner_train_last": cutoff,
                            "inner_eval_last": end, **previous.summarize_rows(frame)})
    oof = {k: pd.concat(v, ignore_index=True) for k, v in oof.items()}
    ranking = pd.DataFrame([{"candidate": spec["name"], **previous.summarize_rows(oof[spec["name"]])}
                            for spec in SPECS]).sort_values(["score_usd", "log_loss", "candidate"], ascending=[False, True, True])
    selected = next(s for s in SPECS if s["name"] == ranking.candidate.iloc[0])
    return selected, pd.DataFrame(records), oof[selected["name"]]


def load_baselines(root):
    folder = Path(root) / "train_only_outputs"
    evidence = json.loads((folder / "run_evidence.json").read_text(encoding="utf-8"))
    path = folder / "outer_predictions.csv"
    if sha256(path) != evidence["result_sha256"]["outer_predictions"]:
        raise ValueError("Попередні train-only прогнози змінилися")
    if sha256((Path(root) / "outputs" / "train_audit.csv")) != evidence["train_audit_sha256"]:
        raise ValueError("Train-аудит відрізняється від попереднього експерименту")
    return pd.read_csv(path, dtype={"dispute_id": "string", "outer_period": "string"}), evidence


def summarize(predictions, calibration):
    rows, periods = [], []
    for name, frame in predictions.groupby("strategy", sort=False):
        main, observed = previous.summarize_rows(frame), previous.summarize_rows(frame, "observed_only")
        row = {"strategy": name, **main, "observed_rows": observed["evaluated_rows"],
               "observed_score_usd": observed["score_usd"],
               "observed_delta_always_usd": observed["delta_always_usd"],
               "observed_delta_constant_usd": observed["delta_constant_usd"],
               "complexity": COMPLEXITY[name], "selectable": True}
        if name.endswith("_calibrated"):
            family = name.removesuffix("_calibrated")
            row["selectable"] = bool(calibration.loc[calibration.model.eq(family), "adequate_by_protocol"].all())
        ups = []
        for period, group in frame.groupby("outer_period", sort=False):
            m, o = previous.summarize_rows(group), previous.summarize_rows(group, "observed_only")
            periods.append({"strategy": name, "outer_period": period, **m, "observed_score_usd": o["score_usd"], "observed_rows": o["evaluated_rows"]})
            row[f"score_weeks_{period}"] = m["score_usd"]
            ups.append(m["delta_constant_usd"] / m["evaluated_rows"])
        row["worst_period_delta_constant_per_row"] = min(ups)
        row["period_delta_constant_sd"] = float(np.std(ups))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score_usd", ascending=False), pd.DataFrame(periods)


def run(root):
    root = Path(root)
    output, digest = freeze(root)
    access = previous.install_data_access_guard(root)
    data, audit, _ = previous.load_train_only(root)
    baselines, old_evidence = load_baselines(root)
    calibration_file = root / "train_only_outputs" / "calibration_status.csv"
    if sha256(calibration_file) != old_evidence["result_sha256"]["calibration_status"]:
        raise ValueError("Попередній статус калібрування змінився")
    old_calibration = pd.read_csv(calibration_file)
    frames, calibrations, records, choices, oof_frames = [], [], [], [], []
    for cutoff, end in previous.OUTER_FOLDS:
        training = data.loc[data.dispute_week.le(cutoff)].copy()
        spec, candidate_rows, oof = search(training, audit.loc[training.index], previous.INNER_FOLDS[cutoff])
        calibrator, status = previous.train_calibrator(oof, cutoff)
        a, aa, b, ab = previous.prepare_window(data, audit, cutoff, end)
        model = fit_cat(a, aa, spec)
        p = predict(model, b, ab, spec)
        cp = p if calibrator is None else calibrator.predict_proba(logits(p))[:, 1]
        period = f"{cutoff + 1}-{end}"
        lr = baselines.loc[baselines.strategy.eq("logistic") & baselines.outer_period.eq(period)].copy()
        if lr.dispute_id.tolist() != b.dispute_id.tolist():
            raise ValueError("ID CatBoost і попередніх стратегій не збігаються")
        for col in ["amount_usd", "won"]:
            np.testing.assert_allclose(lr[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), equal_nan=True)
        for name, probs in {"catboost": p, "catboost_calibrated": cp, "lr_cat_50_50": .5*p + .5*lr.p.to_numpy()}.items():
            frame = previous.prediction_rows(b, ab, probs, float(a.won.mean()))
            frame["strategy"], frame["outer_period"] = name, period
            frames.append(frame)
        calibrations.append({"model": "catboost", "outer_period": period, **status})
        records.append(candidate_rows.assign(outer_period=period))
        oof_frames.append(oof.assign(outer_period=period))
        choices.append({"outer_period": period, **spec})
        # Контроль збереження кожної зовнішньої моделі без повторного навчання.
        path = output / f"catboost_weeks_{period}.cbm"
        model.save_model(str(path))
        restored = CatBoostClassifier().load_model(str(path))
        np.testing.assert_allclose(predict(restored, b, ab, spec), p, atol=1e-12)
        print(f"Completed train-only outer weeks {period}: {spec['name']}", flush=True)
    predictions = pd.concat([baselines, *frames], ignore_index=True)
    calibration = pd.concat([old_calibration, pd.DataFrame(calibrations)], ignore_index=True)
    # Однакові спори, результати й суми для всіх дев'яти стратегій.
    canonical = None
    for _, frame in predictions.groupby("strategy"):
        view = frame[["dispute_id", "outer_period", "won", "amount_usd", "eligible", "observed_only"]].reset_index(drop=True)
        if canonical is None:
            canonical = view
        else:
            pd.testing.assert_frame_equal(view, canonical, check_dtype=False)
    summary, periods = summarize(predictions, calibration)
    selection = previous.select_from_summary(summary)
    selection.update({"protocol_sha256": digest, "outer_choices": choices,
                      "validation_opened": False, "test_opened": False,
                      "prior_protocol_sha256": old_evidence["protocol_sha256"]})
    tables = {"strategy_summary": summary, "strategy_periods": periods, "outer_predictions": predictions,
              "calibration_status": calibration, "inner_candidate_periods": pd.concat(records),
              "catboost_calibration_oof": pd.concat(oof_frames), "outer_choices": pd.DataFrame(choices)}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    import catboost
    evidence = {"protocol_sha256": digest, "actual_source_csv_open_events": access,
                "catboost_version": catboost.__version__, "prior_evidence": old_evidence,
                "code_sha256": proto.code_hashes("catboost_study.py", "train_only_study.py", "model_comparison.py", "preparation.py"),
                "result_sha256": {n: sha256((output/f"{n}.csv")) for n in tables}}
    (output / "run_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("Selected:", selection["selected_strategy"], flush=True)
    return summary, periods, calibration, selection


if __name__ == "__main__":
    import sys
    if "--freeze-only" in sys.argv:
        print(freeze(PROJECT)[1])
    else:
        run(PROJECT)
