"""First model comparison on validation (notebook 04). Test is not loaded.

The HGB candidate is chosen by temporal validation on train; the final variant by Score on
validation (so validation counts as used afterwards).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.inspection import permutation_importance

from disputes import protocol as proto
from disputes.backtest import time_window
from disputes.config import ACTION_COST, BENEFIT_SHARE, HGB_SPECS, OUTER_FOLDS, SEED
from disputes.economics import check_aligned, payoff, policy, profit_rows
from disputes.evaluation import evaluate, merchant_status, segment_scores
from disputes.models import calibrate, features, fit, fit_platt, predict
from disputes.preparation import SOURCE_COLUMNS, load_table, normalize_source

CANDIDATES = [{"name": "logistic", "kind": "logistic"}, *HGB_SPECS]
FOLDS = OUTER_FOLDS
OUTPUT = "comparison_outputs"


def load_inputs(root: Path):
    inputs = Path(root) / "outputs"
    train, audit = load_table(inputs, "train_clean.csv"), load_table(inputs, "train_audit.csv")
    validation, val_audit = load_table(inputs, "validation_clean.csv"), load_table(inputs, "validation_audit.csv")
    check_aligned(train, audit)
    check_aligned(validation, val_audit)
    assert train.dispute_week.max() < validation.dispute_week.min()
    assert set(train.dispute_id).isdisjoint(validation.dispute_id)
    # Train is restored from the original audit values for temporal folds. Test is not read.
    raw_train = audit[["raw__" + c for c in SOURCE_COLUMNS]].rename(columns=lambda c: c.removeprefix("raw__"))
    normalized, source_audit = normalize_source(raw_train)
    assert normalized.dispute_id.tolist() == train.dispute_id.tolist()
    return train, audit, validation, val_audit, normalized, source_audit


def cross_validate(normalized: pd.DataFrame, source_audit: pd.DataFrame):
    """Temporal validation of all candidates on three train windows; HGB leader by total Score."""
    rows, predictions, fold_frames = [], {}, []
    for last_train, last_eval in FOLDS:
        window = time_window(normalized, source_audit, last_train, last_eval)
        fold = f"train_0_{last_train}__eval_{last_train + 1}_{last_eval}"
        fold_frames.append({"fold": fold, "train_rows": len(window.train), "eval_rows": len(window.eval),
                            "train_max_week": int(window.train.dispute_week.max()),
                            "eval_min_week": int(window.eval.dispute_week.min()),
                            "eligible_rows": int(window.eval_audit.score_eligible.sum())})
        for spec in CANDIDATES:
            p = predict(fit(window.train, window.train_audit, spec), window.eval, window.eval_audit, spec)
            metrics, economics = evaluate(window.eval, window.eval_audit, p)
            rows += [{"candidate": spec["name"], "fold": fold, **metrics, **economic} for economic in economics]
            predictions.setdefault(spec["name"], []).append(pd.DataFrame({
                "dispute_id": window.eval.dispute_id.to_numpy(), "won": window.eval.won.to_numpy(),
                "dispute_week": window.eval.dispute_week.to_numpy(), "p": p, "fold": fold}))
            print(f"{fold}: {spec['name']} Score={economics[0]['score_usd']:.2f}, "
                  f"logloss={metrics['log_loss']:.4f}", flush=True)
    scores = pd.DataFrame(rows)
    ranked = scores.query("scenario == 'eligible'").groupby("candidate").agg(
        total_score_usd=("score_usd", "sum"), total_delta_usd=("delta_vs_always_usd", "sum"),
        mean_log_loss=("log_loss", "mean"), mean_roc_auc=("roc_auc", "mean"),
        min_fold_delta=("delta_vs_always_usd", "min"),
    ).sort_values(["total_score_usd", "mean_log_loss"], ascending=[False, True])
    best = next(spec for spec in CANDIDATES if spec["name"] == next(n for n in ranked.index if n.startswith("hgb_")))
    oof = {name: pd.concat(frames, ignore_index=True) for name, frames in predictions.items()}
    return scores, ranked, best, oof, pd.DataFrame(fold_frames)


def fit_finalists(train, audit, validation, val_audit, spec, oof):
    """Logistic, HGB, calibrated HGB and a 50/50 ensemble on validation; selection by Score."""
    logistic_spec = CANDIDATES[0]
    logistic, boosting = fit(train, audit, logistic_spec), fit(train, audit, spec)
    lp, cp = predict(logistic, validation, val_audit, logistic_spec), predict(boosting, validation, val_audit, spec)
    # The calibrator sees only predictions on later train folds, never validation.
    calibrator = fit_platt(oof[spec["name"]].p, oof[spec["name"]].won)
    probabilities = {"constant_train_rate": np.full(len(validation), train.won.mean()),
                     "logistic": lp, "boosting": cp, "boosting_calibrated": calibrate(calibrator, cp),
                     "blend_50_50": (lp + cp) / 2}
    metrics, econ = [], []
    for name, p in probabilities.items():
        m, e = evaluate(validation, val_audit, p)
        metrics.append({"model": name, **m})
        econ += [{"model": name, **r} for r in e]
    metrics, econ = pd.DataFrame(metrics), pd.DataFrame(econ)
    ranking = econ.query("scenario == 'eligible' and model != 'constant_train_rate'").merge(
        metrics[["model", "log_loss"]], on="model").sort_values(["score_usd", "log_loss"], ascending=[False, True])
    return logistic, boosting, calibrator, probabilities, metrics, econ, ranking.model.iloc[0]


def week_diagnostics(data, audit, probabilities) -> pd.DataFrame:
    frames = [segment_scores(data, audit, p, data.dispute_week).assign(model=name) for name, p in probabilities.items()]
    weeks = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({"model": weeks.model, "week": weeks.segment.astype(int), "rows": weeks.eligible_rows,
                         "score_usd": weeks.score_usd.fillna(0.), "always_usd": weeks.always_usd.fillna(0.),
                         "delta_vs_always_usd": weeks.delta_vs_always_usd.fillna(0.)})


def paired_block_intervals(data, audit, probabilities, selected, repeats=2000) -> pd.DataFrame:
    """Descriptive 95% block bootstrap intervals (week / merchant) of the Score difference; not an independent test."""
    mask = audit.score_eligible.to_numpy(dtype=bool)
    selected_profit = profit_rows(data, audit, probabilities[selected])[mask]
    references = {"always_fight": payoff(data)[mask],
                  "logistic": profit_rows(data, audit, probabilities["logistic"])[mask]}
    rows = []
    for reference, other in references.items():
        delta = selected_profit - other
        for block in ["dispute_week", "merchant_id"]:
            groups = data.loc[mask, block].astype("string").fillna("MISSING").to_numpy()
            totals = pd.Series(delta).groupby(groups).sum().to_numpy()
            interval = bootstrap((totals,), np.sum, n_resamples=repeats, method="percentile",
                                 rng=np.random.default_rng(SEED)).confidence_interval
            rows.append({"reference": reference, "block": block, "blocks": len(totals), "delta_usd": float(delta.sum()),
                         "p025_usd": float(interval.low), "p975_usd": float(interval.high), "repeats": repeats})
    return pd.DataFrame(rows)


def permutation_diagnostics(model, data, audit, spec, repeats=5) -> pd.DataFrame:
    """Log loss increase after permuting each feature (sklearn permutation importance)."""
    x, _ = features(data, audit, spec)
    result = permutation_importance(model, x, data.won.astype(int), scoring="neg_log_loss",
                                    n_repeats=repeats, random_state=SEED)
    return pd.DataFrame({"feature": x.columns, "log_loss_increase": result.importances_mean,
                         "repeat_sd": result.importances_std, "repeats": repeats}
                        ).sort_values("log_loss_increase", ascending=False)


def segment_diagnostics(data, audit, p, train) -> pd.DataFrame:
    groups = {"payment_method": data.payment_method, "reason_category": data.reason_category,
              "amount_band": pd.cut(data.amount_usd, [0, 30, 60, 100, np.inf], include_lowest=True).astype("string"),
              "merchant_seen": merchant_status(data, train)}
    frames = [segment_scores(data, audit, p, values).assign(field=field) for field, values in groups.items()]
    result = pd.concat(frames, ignore_index=True)
    return result.rename(columns={"segment": "group", "eligible_rows": "evaluated_rows", "mean_probability": "mean_p"})[
        ["field", "group", "rows", "evaluated_rows", "win_rate", "mean_p", "fight_rows", "score_usd", "delta_vs_always_usd"]]


def export_run(root, spec, models, probabilities, metrics, econ, selected,
               cv_scores, cv_rank, folds, oof, week, intervals, importance, segments, validation, val_audit):
    output = Path(root) / OUTPUT
    output.mkdir(exist_ok=True)
    tables = {"cv_scores": cv_scores, "cv_ranking": cv_rank.reset_index(), "cv_folds": folds,
              "validation_metrics": metrics, "validation_economics": econ, "validation_weekly": week,
              "paired_block_intervals": intervals, "boosting_permutation_importance": importance,
              "selected_segments": segments, **{f"oof_{name}": frame for name, frame in oof.items()}}
    predictions = validation[["dispute_id", "dispute_week", "won", "amount_usd"]].assign(
        score_eligible=val_audit.score_eligible, score_observed_only=val_audit.score_observed_only,
        **{"p_" + name: p for name, p in probabilities.items()},
        selected_action=policy(validation, val_audit, probabilities[selected]),
        selected_profit_usd=profit_rows(validation, val_audit, probabilities[selected]))
    tables["validation_predictions"] = predictions
    hashes = proto.write_tables(output, tables)
    # The first run records checksums; validation_extension verifies against them.
    snapshot = output / "first_run_sha256.json"
    if not snapshot.exists():
        proto.write_json(snapshot, {f"{name}.csv": hashes[name] for name in ["validation_predictions", "validation_economics"]})
    logistic, boosting, calibrator = models
    for name, model in {"logistic": logistic, "boosting": boosting, "calibrator": calibrator}.items():
        joblib.dump(model, output / f"{name}.joblib")
    manifest = proto.read_json(Path(root) / "outputs" / "manifest.json")
    config = {"selected_validation_model": selected, "boosting_spec": spec,
              "boosting_parameters": boosting.named_steps["classifier"].get_params(),
              "candidates": CANDIDATES, "folds": FOLDS,
              "candidate_selection": "highest pooled temporal CV Score; mean log loss breaks ties",
              "final_selection": "highest validation Score among logistic, HistGradientBoosting, "
                                 "calibrated HistGradientBoosting, fixed 50/50 blend",
              "decision_cost": ACTION_COST, "benefit_fraction": BENEFIT_SHARE, "threshold_tuned": False,
              "test_evaluated": False, "validation_used_for_selection": True,
              "input_sha256": {name: manifest["files"][name]["sha256"] for name in
                               ["train_clean.csv", "train_audit.csv", "validation_clean.csv", "validation_audit.csv"]},
              "code_sha256": proto.code_hashes("model_comparison.py", "models.py", "evaluation.py", "economics.py"),
              "versions": proto.runtime_versions()}
    proto.write_json(output / "selection.json", config)
    # Independent per-row recomputation of the selected strategy and reload of saved models.
    expected = econ.query("scenario == 'eligible' and model == @selected").score_usd.iloc[0]
    assert np.isclose(predictions.selected_profit_usd.sum(), expected)
    restored = {name: joblib.load(output / f"{name}.joblib") for name in ["logistic", "boosting", "calibrator"]}
    assert np.allclose(predict(restored["boosting"], validation, val_audit, spec), probabilities["boosting"], atol=1e-12)
    assert np.allclose(predict(restored["logistic"], validation, val_audit, CANDIDATES[0]),
                       probabilities["logistic"], atol=1e-12)
    assert np.allclose(calibrate(restored["calibrator"], probabilities["boosting"]),
                       probabilities["boosting_calibrated"], atol=1e-12)
    return output, config
