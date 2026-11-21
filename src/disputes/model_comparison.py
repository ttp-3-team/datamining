"""Часове порівняння моделей спорів. Test не завантажується."""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, OrdinalEncoder

from disputes.preparation import (SOURCE_COLUMNS, fit_scale_reference, load_table,
                         normalize_source, prepare_amounts, calculate_baseline)
from disputes import protocol as proto
from disputes.protocol import sha256

BASE_CATEGORIES = ["merchant_id", "psp", "payment_method", "card_brand",
                   "reason_category", "reason_code", "currency", "dispute_weekday"]
QUALITY = ["currency_dollar_assumed_usd", "amount_invalid_removed", "amount_usd_invalid_removed",
           "amount_restored", "amount_usd_restored", "amount_usd_scale_corrected"]
FOLDS = [(5, 9), (9, 13), (13, 17)]
CANDIDATES = [
    {"name": "logistic", "kind": "logistic"},
    {"name": "hgb_depth4", "kind": "boosting", "depth": 4},
    {"name": "hgb_depth6", "kind": "boosting", "depth": 6},
    {"name": "hgb_add_amount", "kind": "boosting", "depth": 4, "add_amount": True},
    {"name": "hgb_add_quality", "kind": "boosting", "depth": 4, "add_quality": True},
    {"name": "hgb_without_merchant", "kind": "boosting", "depth": 4, "drop_merchant": True},
    {"name": "hgb_add_week", "kind": "boosting", "depth": 4, "add_week": True},
]


def aligned(data, audit):
    if not data.dispute_id.equals(audit.dispute_id):
        raise ValueError("Рядки даних і аудиту не відповідають одне одному")


def features(data, audit, spec):
    """Лише ознаки, доступні до результату; жодних target-агрегатів вручну."""
    aligned(data, audit)
    cats = [c for c in BASE_CATEGORIES if not (spec.get("drop_merchant") and c == "merchant_id")]
    nums = ["amount_usd"]
    if spec.get("add_amount"):
        nums.append("amount")
    if spec.get("add_week"):
        nums.append("dispute_week")
    x = data[cats + nums].copy()
    for col in cats:
        x[col] = x[col].astype("string").fillna("MISSING").astype(str)
    for col in nums:
        x[col] = x[col].astype(float)
    if spec.get("add_quality"):
        for col in QUALITY:
            x[col] = audit[col].to_numpy(dtype=np.int8)
    if "won" in x or "dispute_id" in x or any(c.startswith("raw__") for c in x):
        raise AssertionError("Витік цілі або службових полів")
    return x, cats


def new_model(spec, seed=42):
    if spec["kind"] == "logistic":
        preprocess = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), BASE_CATEGORIES),
            ("amount", Pipeline([
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), ["amount_usd"]),
        ])
        return Pipeline([("preprocess", preprocess),
                         ("classifier", LogisticRegression(C=1., max_iter=2000, random_state=seed))])
    cats = [c for c in BASE_CATEGORIES if not (spec.get("drop_merchant") and c == "merchant_id")]
    nums = ["amount_usd"] + (["amount"] if spec.get("add_amount") else [])
    nums += ["dispute_week"] if spec.get("add_week") else []
    nums += QUALITY if spec.get("add_quality") else []
    preprocessing = ColumnTransformer([
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats),
        ("numeric", "passthrough", nums),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=400, max_depth=spec["depth"], max_leaf_nodes=31,
            learning_rate=.05, l2_regularization=5., min_samples_leaf=30,
            categorical_features=list(range(len(cats))), early_stopping=False, random_state=seed)),
    ])


def fit_model(data, audit, spec, seed=42):
    x, cats = features(data, audit, spec)
    order = data.dispute_week.sort_values(kind="stable").index
    # iloc через позиції: початковий індекс CSV не є часовим порядком.
    positions = data.index.get_indexer(order)
    model = new_model(spec, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x.iloc[positions], data.won.iloc[positions].astype(int))
    return model


def predict(model, data, audit, spec):
    x, _ = features(data, audit, spec)
    p = model.predict_proba(x)[:, list(model.classes_).index(1)]
    validate_probabilities(p, len(data))
    return p


def validate_probabilities(p, n):
    p = np.asarray(p, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Некоректні ймовірності")


def policy(data, audit, p, cost=6.):
    aligned(data, audit)
    validate_probabilities(p, len(data))
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    amount = data.amount_usd.to_numpy(dtype=float)
    if not (np.isfinite(amount[eligible]) & (amount[eligible] > 0)).all():
        raise ValueError("Прапорець Score суперечить сумам")
    fight = eligible & (.2 * amount * np.asarray(p) > cost)
    return np.where(~eligible, "review_amount", np.where(fight, "fight", "skip"))


def profit_rows(data, audit, p, cost=6.):
    actions = policy(data, audit, p, cost)
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    payoff = .2 * data.amount_usd.to_numpy(dtype=float) * data.won.to_numpy(dtype=int) - cost
    return np.where(eligible, np.where(actions == "fight", payoff, 0.), np.nan)


def evaluate(data, audit, p):
    y = data.won.to_numpy(dtype=int)
    classification = {
        "rows": len(data), "win_rate": float(y.mean()), "roc_auc": roc_auc_score(y, p),
        "average_precision_won": average_precision_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p), "accuracy": accuracy_score(y, p >= .5),
        "precision_won": precision_score(y, p >= .5, zero_division=0),
        "recall_won": recall_score(y, p >= .5, zero_division=0),
        "precision_lost": precision_score(y, p >= .5, pos_label=0, zero_division=0),
        "recall_lost": recall_score(y, p >= .5, pos_label=0, zero_division=0),
    }
    profits = profit_rows(data, audit, p)
    actions = policy(data, audit, p)
    economics = []
    for scenario, flag in [("eligible", "score_eligible"), ("observed_only", "score_observed_only")]:
        mask = audit[flag].to_numpy(dtype=bool)
        baseline = calculate_baseline(data, audit, observed_only=scenario == "observed_only")
        score = float(profits[mask].sum())
        economics.append({"scenario": scenario, "rows": int(mask.sum()), "coverage": float(mask.mean()),
                          "fight_rows": int(((actions == "fight") & mask).sum()), "score_usd": score,
                          "always_usd": baseline["score_usd"],
                          "delta_vs_always_usd": score - baseline["score_usd"]})
    return classification, economics


def load_inputs(root):
    inputs = root / "outputs"
    train = load_table(inputs, "train_clean.csv")
    audit = load_table(inputs, "train_audit.csv")
    validation = load_table(inputs, "validation_clean.csv")
    val_audit = load_table(inputs, "validation_audit.csv")
    aligned(train, audit)
    aligned(validation, val_audit)
    assert train.dispute_week.max() < validation.dispute_week.min()
    assert set(train.dispute_id).isdisjoint(validation.dispute_id)
    # Відновлюємо лише train із початкових значень аудиту. Test не читається.
    raw_train = audit[["raw__" + c for c in SOURCE_COLUMNS]].rename(columns=lambda c: c[5:])
    normalized, source_audit = normalize_source(raw_train)
    assert normalized.dispute_id.tolist() == train.dispute_id.tolist()
    return train, audit, validation, val_audit, normalized, source_audit


def temporal_fold(data, source_audit, last_train, last_eval):
    train_mask = data.dispute_week.le(last_train)
    eval_mask = data.dispute_week.gt(last_train) & data.dispute_week.le(last_eval)
    a, b = data.loc[train_mask].copy(), data.loc[eval_mask].copy()
    assert len(a) and len(b) and a.dispute_week.max() < b.dispute_week.min()
    reference = fit_scale_reference(a)
    ca, aa = prepare_amounts(a, source_audit.loc[a.index], reference)
    cb, ab = prepare_amounts(b, source_audit.loc[b.index], reference)
    return ca, aa, cb, ab, reference


def cross_validate(normalized, source_audit):
    rows, predictions = [], {}
    fold_frames = []
    for last_train, last_eval in FOLDS:
        a, aa, b, ab, _ = temporal_fold(normalized, source_audit, last_train, last_eval)
        fold_name = f"train_0_{last_train}__eval_{last_train + 1}_{last_eval}"
        fold_frames.append({"fold": fold_name, "train_rows": len(a), "eval_rows": len(b),
                            "train_max_week": int(a.dispute_week.max()),
                            "eval_min_week": int(b.dispute_week.min()),
                            "eligible_rows": int(ab.score_eligible.sum())})
        for spec in CANDIDATES:
            model = fit_model(a, aa, spec)
            p = predict(model, b, ab, spec)
            metrics, economics = evaluate(b, ab, p)
            for economic in economics:
                rows.append({"candidate": spec["name"], "fold": fold_name, **metrics, **economic})
            predictions.setdefault(spec["name"], []).append(pd.DataFrame({
                "dispute_id": b.dispute_id.to_numpy(), "won": b.won.to_numpy(),
                "dispute_week": b.dispute_week.to_numpy(), "p": p,
                "fold": fold_name,
            }))
            print(f"{fold_name}: {spec['name']} Score={economics[0]['score_usd']:.2f}, "
                  f"logloss={metrics['log_loss']:.4f}", flush=True)
    scores = pd.DataFrame(rows)
    ranked = scores.query("scenario == 'eligible'").groupby("candidate").agg(
        total_score_usd=("score_usd", "sum"), total_delta_usd=("delta_vs_always_usd", "sum"),
        mean_log_loss=("log_loss", "mean"), mean_roc_auc=("roc_auc", "mean"),
        min_fold_delta=("delta_vs_always_usd", "min"),
    ).sort_values(["total_score_usd", "mean_log_loss"], ascending=[False, True])
    hgb_name = next(name for name in ranked.index if name.startswith("hgb_"))
    best = next(s for s in CANDIDATES if s["name"] == hgb_name)
    return scores, ranked, best, {k: pd.concat(v, ignore_index=True) for k, v in predictions.items()}, pd.DataFrame(fold_frames)


def logits(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_finalists(train, audit, validation, val_audit, spec, oof):
    logistic_spec = CANDIDATES[0]
    logistic = fit_model(train, audit, logistic_spec)
    boosting = fit_model(train, audit, spec)
    lp = predict(logistic, validation, val_audit, logistic_spec)
    cp = predict(boosting, validation, val_audit, spec)
    calibration_rows = oof[spec["name"]]
    # Калібратор бачить лише прогнози на пізніших train-фолдах, не validation.
    calibrator = LogisticRegression(C=1., max_iter=1000, random_state=42)
    calibrator.fit(logits(calibration_rows.p), calibration_rows.won.astype(int))
    calibrated = calibrator.predict_proba(logits(cp))[:, 1]
    probabilities = {"constant_train_rate": np.full(len(validation), train.won.mean()),
                     "logistic": lp, "boosting": cp, "boosting_calibrated": calibrated,
                     "blend_50_50": (lp + cp) / 2}
    metrics, econ = [], []
    for name, p in probabilities.items():
        m, e = evaluate(validation, val_audit, p)
        metrics.append({"model": name, **m})
        econ.extend({"model": name, **r} for r in e)
    metrics, econ = pd.DataFrame(metrics), pd.DataFrame(econ)
    # Validation дозволено використати для вибору серед чотирьох фіксованих варіантів.
    ranking = econ.query("scenario == 'eligible' and model != 'constant_train_rate'").merge(
        metrics[["model", "log_loss"]], on="model").sort_values(
            ["score_usd", "log_loss"], ascending=[False, True])
    selected = ranking.model.iloc[0]
    return logistic, boosting, calibrator, probabilities, metrics, econ, selected


def week_diagnostics(data, audit, probabilities):
    rows = []
    for name, p in probabilities.items():
        profit = profit_rows(data, audit, p)
        for week in sorted(data.dispute_week.unique()):
            mask = (data.dispute_week.eq(week) & audit.score_eligible).to_numpy(dtype=bool)
            always = (.2 * data.amount_usd.to_numpy(dtype=float) * data.won.to_numpy() - 6)[mask].sum()
            rows.append({"model": name, "week": int(week), "rows": int(mask.sum()),
                         "score_usd": float(profit[mask].sum()), "always_usd": float(always),
                         "delta_vs_always_usd": float(profit[mask].sum() - always)})
    return pd.DataFrame(rows)


def paired_block_intervals(data, audit, probabilities, selected, repeats=2000):
    """Описова невизначеність після вибору на validation; не незалежний тест."""
    mask = audit.score_eligible.to_numpy(dtype=bool)
    selected_profit = profit_rows(data, audit, probabilities[selected])[mask]
    always = (.2 * data.amount_usd.to_numpy(dtype=float) * data.won.to_numpy() - 6)[mask]
    rng = np.random.default_rng(42)
    rows = []
    for baseline in ["always_fight", "logistic"]:
        other = always if baseline == "always_fight" else profit_rows(data, audit, probabilities[baseline])[mask]
        delta = selected_profit - other
        for block in ["dispute_week", "merchant_id"]:
            groups = data.loc[mask, block].astype("string").fillna("MISSING").to_numpy()
            totals = pd.DataFrame({"block": groups, "delta": delta}).groupby("block").delta.sum().to_numpy()
            samples = np.array([rng.choice(totals, size=len(totals), replace=True).sum() for _ in range(repeats)])
            rows.append({"reference": baseline, "block": block, "blocks": len(totals),
                         "delta_usd": float(delta.sum()), "p025_usd": float(np.quantile(samples, .025)),
                         "p975_usd": float(np.quantile(samples, .975)), "repeats": repeats})
    return pd.DataFrame(rows)


def permutation_diagnostics(model, data, audit, spec, repeats=5):
    """Вплив перемішування ознак на log loss HistGradientBoosting, без повторного вибору ознак."""
    x, _ = features(data, audit, spec)
    y = data.won.astype(int)
    baseline = log_loss(y, model.predict_proba(x)[:, 1], labels=[0, 1])
    rng, rows = np.random.default_rng(42), []
    for col in x:
        effects = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            effects.append(log_loss(y, model.predict_proba(shuffled)[:, 1], labels=[0, 1]) - baseline)
        rows.append({"feature": col, "log_loss_increase": float(np.mean(effects)),
                     "repeat_sd": float(np.std(effects)), "repeats": repeats})
    return pd.DataFrame(rows).sort_values("log_loss_increase", ascending=False)


def segment_diagnostics(data, audit, p, train):
    mask = audit.score_eligible.to_numpy(dtype=bool)
    profits = profit_rows(data, audit, p)
    always = .2 * data.amount_usd.to_numpy(dtype=float) * data.won.to_numpy() - 6
    actions = policy(data, audit, p)
    groups = {
        "payment_method": data.payment_method.astype("string").fillna("MISSING"),
        "reason_category": data.reason_category.astype("string").fillna("MISSING"),
        "amount_band": pd.cut(data.amount_usd, [0, 30, 60, 100, np.inf], include_lowest=True).astype("string").fillna("MISSING"),
        "merchant_seen": pd.Series(np.where(data.merchant_id.isna(), "missing",
                                     np.where(data.merchant_id.isin(train.merchant_id.dropna()), "seen", "unseen")), index=data.index),
    }
    rows = []
    for field, groups_series in groups.items():
        for value in groups_series.unique():
            group = groups_series.eq(value).to_numpy(dtype=bool)
            scored = group & mask
            rows.append({"field": field, "group": str(value), "rows": int(group.sum()),
                         "evaluated_rows": int(scored.sum()), "win_rate": float(data.loc[group, "won"].mean()),
                         "mean_p": float(np.mean(p[group])), "fight_rows": int(((actions == "fight") & scored).sum()),
                         "score_usd": float(profits[scored].sum()) if scored.any() else np.nan,
                         "delta_vs_always_usd": float(profits[scored].sum() - always[scored].sum()) if scored.any() else np.nan})
    return pd.DataFrame(rows)


def export_run(root, spec, models, probabilities, metrics, econ, selected,
               cv_scores, cv_rank, folds, oof, week, intervals, importance, segments, validation, val_audit):
    import joblib
    import sklearn
    import scipy
    import sys
    output = root / "comparison_outputs"
    output.mkdir(exist_ok=True)
    tables = {"cv_scores": cv_scores, "cv_ranking": cv_rank.reset_index(), "cv_folds": folds,
              "validation_metrics": metrics, "validation_economics": econ, "validation_weekly": week,
              "paired_block_intervals": intervals, "boosting_permutation_importance": importance,
              "selected_segments": segments}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    for name, frame in oof.items():
        frame.to_csv(output / f"oof_{name}.csv", index=False)
    predictions = validation[["dispute_id", "dispute_week", "won", "amount_usd"]].copy()
    predictions["score_eligible"] = val_audit.score_eligible
    predictions["score_observed_only"] = val_audit.score_observed_only
    for name, p in probabilities.items():
        predictions["p_" + name] = p
    predictions["selected_action"] = policy(validation, val_audit, probabilities[selected])
    predictions["selected_profit_usd"] = profit_rows(validation, val_audit, probabilities[selected])
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    # Перший запуск фіксує контрольні суми; validation_extension звіряє з ними.
    snapshot = output / "first_run_sha256.json"
    if not snapshot.exists():
        hashes = {name: sha256(output / name) for name in ["validation_predictions.csv", "validation_economics.csv"]}
        snapshot.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    joblib.dump(models[1], output / "boosting.joblib")
    joblib.dump(models[0], output / "logistic.joblib")
    joblib.dump(models[2], output / "calibrator.joblib")
    manifest = json.loads((root / "outputs" / "manifest.json").read_text(encoding="utf-8"))
    config = {"selected_validation_model": selected, "boosting_spec": spec,
              "boosting_parameters": {k: v for k, v in models[1].named_steps["classifier"].get_params().items()}, "candidates": CANDIDATES, "folds": FOLDS,
              "candidate_selection": "highest pooled temporal CV Score; mean log loss breaks ties",
              "final_selection": "highest validation Score among logistic, HistGradientBoosting, calibrated HistGradientBoosting, fixed 50/50 blend",
              "decision_cost": 6., "benefit_fraction": .2, "threshold_tuned": False,
              "test_evaluated": False, "validation_used_for_selection": True,
              "input_sha256": {name: manifest["files"][name]["sha256"] for name in
                               ["train_clean.csv", "train_audit.csv", "validation_clean.csv", "validation_audit.csv"]},
              "code_sha256": proto.code_hashes("model_comparison.py"),
              "versions": {"python": sys.version.split()[0],
                           "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                           "scipy": scipy.__version__}}
    (output / "selection.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    # Незалежний рядковий розрахунок вибраної стратегії.
    expected = econ.query("scenario == 'eligible' and model == @selected").score_usd.iloc[0]
    assert np.isclose(predictions.selected_profit_usd.sum(), expected)
    # Повторне завантаження моделей повинне відновити ті самі прогнози.
    restored_cb = joblib.load(output / "boosting.joblib")
    assert np.allclose(predict(restored_cb, validation, val_audit, spec), probabilities["boosting"], atol=1e-12)
    restored_lr = joblib.load(output / "logistic.joblib")
    assert np.allclose(predict(restored_lr, validation, val_audit, CANDIDATES[0]), probabilities["logistic"], atol=1e-12)
    restored_cal = joblib.load(output / "calibrator.joblib")
    assert np.allclose(restored_cal.predict_proba(logits(probabilities["boosting"]))[:, 1],
                       probabilities["boosting_calibrated"], atol=1e-12)
    return output, config
