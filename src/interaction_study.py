"""Контрольоване додавання взаємодій до логістичної регресії, лише train."""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import train_only_study as previous
from model_comparison import BASE_CATEGORIES, features, validate_probabilities

PAIRS = {"psp_currency": ("psp", "currency"), "psp_reason": ("psp", "reason_category")}
VARIANTS = {"baseline": [], "psp_currency": ["psp_currency"], "psp_reason": ["psp_reason"],
            "both_interactions": ["psp_currency", "psp_reason"]}
PROTOCOL = {
    "version": "1.0", "input_allowlist": ["train_audit.csv"],
    "variants": VARIANTS, "pairs": PAIRS, "C_values": previous.C_VALUES,
    "outer_folds": previous.OUTER_FOLDS, "inner_folds": previous.INNER_FOLDS,
    "encoding": "retain base features; add JSON-encoded categorical pairs; train-only one-hot; unknown=ignore",
    "tuning": "highest pooled inner Score for each variant; pooled log loss then smaller C break ties",
    "fixed_C_0_1": "diagnostic only, previously selected on these train periods; excluded from selection",
    "calibration": "none; only interaction hypothesis is changed",
    "selection": {**previous.PROTOCOL["selection"], "complexity": {k: len(v) for k, v in VARIANTS.items()}},
    "decision": "p * 0.2 * amount_usd > 6; missing amount -> review_amount",
    "random_seed": 42, "threads": 4, "validation_opened": False, "test_opened": False,
    "limitations": previous.PROTOCOL["limitations"],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(root):
    output = Path(root) / "interaction_outputs"
    output.mkdir(exist_ok=True)
    path = output / "protocol.json"
    canonical = json.loads(json.dumps(PROTOCOL))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != canonical:
            raise ValueError("Протокол взаємодій уже зафіксований і відрізняється")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(canonical, stream, ensure_ascii=False, indent=2)
    return output, digest(path)


def make_features(data, audit, variant):
    x, _ = features(data, audit, {"kind": "logistic"})
    cats = list(BASE_CATEGORIES)
    for pair in VARIANTS[variant]:
        a, b = PAIRS[pair]
        column = "interaction_" + pair
        # JSON-список зберігає межі категорій навіть за наявності розділювачів у значеннях.
        x[column] = [json.dumps([left, right], ensure_ascii=False) for left, right in zip(x[a], x[b])]
        cats.append(column)
    return x, cats


def new_model(variant, c):
    cats = list(BASE_CATEGORIES) + ["interaction_" + pair for pair in VARIANTS[variant]]
    preprocessing = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), cats),
        ("amount", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), ["amount_usd"]),
    ])
    return Pipeline([("preprocess", preprocessing),
                     ("classifier", LogisticRegression(C=c, max_iter=2000, random_state=42))])


def fit(data, audit, variant, c):
    x, _ = make_features(data, audit, variant)
    order = np.argsort(data.dispute_week.to_numpy(), kind="stable")
    model = new_model(variant, c)
    with warnings.catch_warnings(), threadpool_limits(limits=4):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x.iloc[order], data.won.iloc[order].astype(int))
    return model


def predict(model, data, audit, variant):
    x, _ = make_features(data, audit, variant)
    with threadpool_limits(limits=4):
        p = model.predict_proba(x)[:, list(model.classes_).index(1)]
    validate_probabilities(p, len(data))
    return p


def inner_search(data, audit, folds):
    predictions = {(variant, c): [] for variant in VARIANTS for c in previous.C_VALUES}
    records = []
    for cutoff, end in folds:
        a, aa, b, ab = previous.prepare_window(data, audit, cutoff, end)
        for variant, c in predictions:
            model = fit(a, aa, variant, c)
            p = predict(model, b, ab, variant)
            frame = previous.prediction_rows(b, ab, p, float(a.won.mean()))
            predictions[(variant, c)].append(frame)
            records.append({"variant": variant, "C": c, "inner_train_last": cutoff,
                            "inner_eval_last": end, **previous.summarize_rows(frame)})
    ranked = []
    for (variant, c), frames in predictions.items():
        ranked.append({"variant": variant, "C": c,
                       **previous.summarize_rows(pd.concat(frames, ignore_index=True))})
    rank = pd.DataFrame(ranked)
    chosen = {variant: float(group.sort_values(["score_usd", "log_loss", "C"],
                                                ascending=[False, True, True]).C.iloc[0])
              for variant, group in rank.groupby("variant")}
    return chosen, pd.DataFrame(records), rank


def run_outer(data, audit, cutoff, end):
    training = data.loc[data.dispute_week.le(cutoff)].copy()
    chosen, inner, ranks = inner_search(training, audit.loc[training.index], previous.INNER_FOLDS[cutoff])
    # Усі параметри визначено до створення зовнішньої оцінюваної частини.
    a, aa, b, ab = previous.prepare_window(data, audit, cutoff, end)
    period = f"{cutoff+1}-{end}"
    predictions, choices, coverage, coefficients = [], [], [], []
    for variant, c in [*chosen.items(), ("fixed_C_0_1", .1)]:
        feature_variant = "baseline" if variant == "fixed_C_0_1" else variant
        model = fit(a, aa, feature_variant, c)
        p = predict(model, b, ab, feature_variant)
        frame = previous.prediction_rows(b, ab, p, float(a.won.mean()))
        frame["variant"], frame["outer_period"] = variant, period
        predictions.append(frame)
        x_train, _ = make_features(a, aa, feature_variant)
        x_eval, _ = make_features(b, ab, feature_variant)
        choices.append({"variant": variant, "outer_period": period, "C": c, "train_rows": len(a),
                        "eval_rows": len(b), "encoded_features": len(model.named_steps["classifier"].coef_[0])})
        for pair in VARIANTS[feature_variant]:
            col = "interaction_" + pair
            unknown = ~x_eval[col].isin(x_train[col])
            coverage.append({"variant": variant, "outer_period": period, "pair": pair,
                             "train_combinations": x_train[col].nunique(), "eval_rows": len(b),
                             "unseen_combination_rows": int(unknown.sum())})
        names = model.named_steps["preprocess"].get_feature_names_out()
        coefficients.extend({"variant": variant, "outer_period": period, "feature": name,
                             "coefficient": float(value)} for name, value in
                            zip(names, model.named_steps["classifier"].coef_[0]))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(choices), inner.assign(outer_period=period), ranks.assign(outer_period=period), pd.DataFrame(coverage), pd.DataFrame(coefficients)


def summarize(predictions):
    summaries, periods = [], []
    for variant, frame in predictions.groupby("variant", sort=False):
        main, observed = previous.summarize_rows(frame), previous.summarize_rows(frame, "observed_only")
        row = {"strategy": variant, **main, "observed_rows": observed["evaluated_rows"],
               "observed_score_usd": observed["score_usd"],
               "observed_delta_always_usd": observed["delta_always_usd"],
               "observed_delta_constant_usd": observed["delta_constant_usd"],
               "selectable": variant != "fixed_C_0_1",
               "complexity": len(VARIANTS.get(variant, []))}
        uplift = []
        for period, g in frame.groupby("outer_period", sort=False):
            m, o = previous.summarize_rows(g), previous.summarize_rows(g, "observed_only")
            periods.append({"strategy": variant, "outer_period": period, **m,
                            "observed_rows": o["evaluated_rows"], "observed_score_usd": o["score_usd"]})
            row["score_weeks_" + period] = m["score_usd"]
            uplift.append(m["delta_constant_usd"] / m["evaluated_rows"])
        row["worst_period_delta_constant_per_row"] = min(uplift)
        row["period_delta_constant_sd"] = float(np.std(uplift))
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("score_usd", ascending=False)
    baseline = summary.loc[summary.strategy.eq("baseline")].iloc[0]
    summary["delta_vs_baseline_usd"] = summary.score_usd - baseline.score_usd
    summary["observed_delta_vs_baseline_usd"] = summary.observed_score_usd - baseline.observed_score_usd
    fixed = summary.loc[summary.strategy.eq("fixed_C_0_1")].iloc[0]
    summary["delta_vs_fixed_C_0_1_usd"] = summary.score_usd - fixed.score_usd
    return summary, pd.DataFrame(periods)


def run(root):
    root = Path(root)
    output, protocol_hash = freeze(root)
    access = previous.install_data_access_guard(root)
    data, audit, _ = previous.load_train_only(root)
    outcomes = []
    for cutoff, end in previous.OUTER_FOLDS:
        result = run_outer(data, audit, cutoff, end)
        outcomes.append(result)
        print(f"Completed interaction outer weeks {cutoff+1}-{end}", flush=True)
    predictions, choices, inner, ranks, coverage, coefficients = [pd.concat([r[i] for r in outcomes], ignore_index=True) for i in range(6)]
    summary, periods = summarize(predictions)
    # Ідентичні оцінювані рядки, суми та підмножини для всіх варіантів.
    canonical = None
    for _, frame in predictions.groupby("variant"):
        view = frame[["dispute_id", "outer_period", "won", "amount_usd", "eligible", "observed_only"]].reset_index(drop=True)
        if canonical is None:
            canonical = view
        else:
            pd.testing.assert_frame_equal(view, canonical)
    # Звірити базову модель з попереднім вкладеним train-only дослідженням.
    evidence = json.loads((root / "train_only_outputs" / "run_evidence.json").read_text(encoding="utf-8"))
    old_path = root / "train_only_outputs" / "outer_predictions.csv"
    if digest(old_path) != evidence["result_sha256"]["outer_predictions"]:
        raise ValueError("Попередні train-only прогнози змінилися")
    old = pd.read_csv(old_path, dtype={"dispute_id": "string"})
    old = old.loc[old.strategy.eq("logistic")].reset_index(drop=True)
    baseline = predictions.loc[predictions.variant.eq("baseline")].reset_index(drop=True)
    assert old.dispute_id.tolist() == baseline.dispute_id.tolist()
    np.testing.assert_allclose(old.p, baseline.p, atol=1e-10)
    selection = previous.select_from_summary(summary)
    selection.update({"protocol_sha256": protocol_hash, "validation_opened": False, "test_opened": False,
                      "diagnostic_fixed_C_0_1_excluded": True})
    tables = {"strategy_summary": summary, "strategy_periods": periods, "outer_predictions": predictions,
              "outer_choices": choices, "inner_candidate_periods": inner, "inner_candidate_ranking": ranks,
              "interaction_coverage": coverage, "coefficients": coefficients}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    import sklearn
    run_evidence = {"protocol_sha256": protocol_hash, "actual_source_csv_open_events": access,
                    "baseline_predictions_reproduced": True, "train_audit_sha256": digest(root / "outputs" / "train_audit.csv"),
                    "prior_predictions_sha256": digest(old_path),
                    "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
                    "code_sha256": {name: digest(root / "src" / name) for name in ["interaction_study.py", "train_only_study.py", "model_comparison.py", "preparation.py"]},
                    "result_sha256": {name: digest(output / f"{name}.csv") for name in tables}}
    (output / "run_evidence.json").write_text(json.dumps(run_evidence, indent=2), encoding="utf-8")
    print("Selected:", selection["selected_strategy"], flush=True)
    return summary, periods, choices, coverage, selection


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent.parent
    if "--freeze-only" in sys.argv:
        print(freeze(root)[1])
    else:
        run(root)
