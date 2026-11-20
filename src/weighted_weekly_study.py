"""Зважена логістична регресія та щотижневий вкладений backtest, тільки train."""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import train_only_study as base
import interaction_study as interactions
import catboost_study as cats
from model_comparison import fit_model, predict as generic_predict, logits

WEIGHT_MODES = ["none", "amount_p99", "sqrt_amount_p99"]
WEEKLY_WEEKS = list(range(10, 18))
WEEKLY_INNER = {t: [(t-6, t-3), (t-3, t)] for t in range(9, 17)}
COMPLEXITY = {"constant_train_rate": 0, "logistic": 1, "logistic_calibrated": 2,
    "lr_weight_amount": 2, "lr_weight_amount_calibrated": 3,
    "lr_weight_sqrt": 2, "lr_weight_sqrt_calibrated": 3,
    "boosting": 3, "boosting_calibrated": 4, "catboost": 3, "catboost_calibrated": 4,
    "lr_psp_currency": 2, "lr_psp_reason": 2, "lr_both_interactions": 3,
    "blend_50_50": 5, "lr_cat_50_50": 5, "lr_fixed_C_0_1": 0}
PROTOCOL = {
    "version": "1.0", "source_allowlist": ["train_audit.csv"],
    "weights": {"modes": WEIGHT_MODES, "cap_quantile": .99,
                "missing": "use median positive training amount for importance weight only",
                "normalization": "divide weights by their training mean",
                "preprocessing": "unweighted; weights go only to classifier loss",
                "score": "original unweighted monetary formula; original cleaned USD amounts"},
    "C_values": base.C_VALUES, "primary_outer_folds": base.OUTER_FOLDS,
    "primary_inner_folds": base.INNER_FOLDS, "weekly_eval_weeks": WEEKLY_WEEKS,
    "weekly_inner_folds": WEEKLY_INNER, "weekly_methods": list(COMPLEXITY),
    "hgb_candidates": base.GB_SPECS, "catboost_candidates": cats.SPECS,
    "interactions": interactions.VARIANTS,
    "parameter_selection": "highest summed inner raw Score; pooled log loss; C or candidate name",
    "calibration": base.PROTOCOL["calibration"],
    "selection": {**base.PROTOCOL["selection"], "complexity": COMPLEXITY},
    "fixed_C_0_1": "previously selected diagnostic only; not selectable",
    "blends": "fixed 50/50 uncalibrated tuned logistic with HGB / CatBoost",
    "weekly_stability": "per-week uplift vs always, constant, and same-week logistic; counts and USD/1000",
    "no_class_balance": True, "seed": 42, "threads": 4,
    "validation_opened": False, "test_opened": False,
    "limitations": base.PROTOCOL["limitations"] + [
        "weekly cases overlap previously evaluated weeks 10-17; not new independent data",
        "training windows overlap and outcomes may not have matured by next week",
        "clipped amount weighting is a hypothesis, not direct optimization of Score"],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(root):
    output = Path(root)/"weighted_weekly_outputs"
    output.mkdir(exist_ok=True)
    p = output/"protocol.json"
    value = json.loads(json.dumps(PROTOCOL))
    if p.exists():
        if json.loads(p.read_text(encoding="utf-8")) != value:
            raise ValueError("Протокол уже зафіксований і відрізняється")
    else:
        with p.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
    return output, digest(p)


def sample_weights(data, mode):
    amount = data.amount_usd.to_numpy(dtype=float)
    valid = np.isfinite(amount) & (amount > 0)
    if mode == "none":
        return np.ones(len(data)), {"mode": mode, "rows": len(data), "mean": 1., "max": 1., "weight_effective_n": float(len(data))}
    if mode not in WEIGHT_MODES or not valid.any():
        raise ValueError("Немає опори для ваг або невідомий режим")
    median, cap = float(np.median(amount[valid])), float(np.quantile(amount[valid], .99))
    values = np.minimum(np.where(valid, amount, median), cap)
    if mode == "sqrt_amount_p99":
        values = np.sqrt(values)
    weights = values / values.mean()
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Некоректні ваги")
    return weights, {"mode": mode, "rows": len(data), "median_amount": median, "cap_amount": cap,
                     "missing_weight_rows": int((~valid).sum()), "mean": float(weights.mean()),
                     "max": float(weights.max()), "weight_effective_n": float(weights.sum()**2 / (weights**2).sum())}


def lr_specs(mode="none", variant="baseline"):
    return [{"kind": "lr", "variant": variant, "weight": mode, "C": c,
             "name": f"lr_{variant}_{mode}_C{c:g}"} for c in base.C_VALUES]


class Engine:
    """Кеш тільки за cutoff і фіксованою специфікацією; майбутні мітки не входять у fit."""
    def __init__(self, data, audit):
        self.data, self.audit = data, audit
        self.windows, self.models, self.predictions = {}, {}, {}
        self.fit_log = []

    def window(self, cutoff, end):
        key = (cutoff, end)
        if key not in self.windows:
            self.windows[key] = base.prepare_window(self.data, self.audit, cutoff, end)
        return self.windows[key]

    def predict(self, cutoff, end, spec):
        text = json.dumps(spec, sort_keys=True)
        key = (cutoff, text)
        a, aa, b, ab = self.window(cutoff, end)
        if key not in self.models:
            assert a.dispute_week.max() <= cutoff < b.dispute_week.min()
            if spec["kind"] == "lr":
                x, _ = interactions.make_features(a, aa, spec["variant"])
                model = interactions.new_model(spec["variant"], spec["C"])
                weights, stats = sample_weights(a, spec["weight"])
                order = np.argsort(a.dispute_week.to_numpy(), kind="stable")
                kwargs = {} if spec["weight"] == "none" else {"classifier__sample_weight": weights[order]}
                with warnings.catch_warnings(), threadpool_limits(limits=4):
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(x.iloc[order], a.won.iloc[order].astype(int), **kwargs)
            elif spec["kind"] == "catboost":
                model, stats = cats.fit_cat(a, aa, spec), {}
            else:
                with threadpool_limits(limits=4):
                    model = fit_model(a, aa, spec)
                stats = {}
            self.models[key] = model
            self.fit_log.append({"cutoff": cutoff, "max_fit_week": int(a.dispute_week.max()),
                                 "candidate": spec["name"], **stats})
        prediction_key = (cutoff, end, text)
        if prediction_key not in self.predictions:
            model = self.models[key]
            if spec["kind"] == "lr":
                p = interactions.predict(model, b, ab, spec["variant"])
            else:
                with threadpool_limits(limits=4):
                    p = generic_predict(model, b, ab, spec)
            self.predictions[prediction_key] = p
        return self.predictions[prediction_key]


def search(engine, folds, specs):
    oof, ranks, records = {}, [], []
    for spec in specs:
        frames = []
        for cutoff, end in folds:
            a, aa, b, ab = engine.window(cutoff, end)
            p = engine.predict(cutoff, end, spec)
            frame = base.prediction_rows(b, ab, p, float(a.won.mean()))
            frame["inner_period"], frame["base_train_max_week"] = f"{cutoff+1}-{end}", cutoff
            frames.append(frame)
            records.append({"candidate": spec["name"], "inner_train_last": cutoff,
                            "inner_eval_last": end, **base.summarize_rows(frame)})
        merged = pd.concat(frames, ignore_index=True)
        oof[spec["name"]] = merged
        ranks.append({"candidate": spec["name"], "C": spec.get("C", np.nan), **base.summarize_rows(merged)})
    ranking = pd.DataFrame(ranks).sort_values(["score_usd", "log_loss", "C", "candidate"],
                                              ascending=[False, True, True, True], na_position="last")
    spec = next(s for s in specs if s["name"] == ranking.candidate.iloc[0])
    return spec, oof[spec["name"]], pd.DataFrame(records)


def evaluate_cutoff(engine, cutoff, end, inner_folds, all_methods):
    families = [("logistic", lr_specs()), ("lr_weight_amount", lr_specs("amount_p99")),
                ("lr_weight_sqrt", lr_specs("sqrt_amount_p99"))]
    if all_methods:
        families += [("boosting", base.GB_SPECS), ("catboost", cats.SPECS)]
        families += [("lr_"+v, lr_specs(variant=v)) for v in ["psp_currency", "psp_reason", "both_interactions"]]
    probabilities, choices, cal_status, inner_records = {}, [], [], []
    a, aa, b, ab = engine.window(cutoff, end)
    for family, specs in families:
        selected, oof, records = search(engine, inner_folds, specs)
        raw = engine.predict(cutoff, end, selected)
        probabilities[family] = raw
        choices.append({"family": family, "cutoff": cutoff, "eval_end": end, "spec": json.dumps(selected, sort_keys=True)})
        inner_records.append(records.assign(family=family, outer_cutoff=cutoff, outer_eval_end=end))
        if not family.startswith("lr_psp") and family != "lr_both_interactions":
            calibrator, status = base.train_calibrator(oof, cutoff)
            probabilities[family+"_calibrated"] = raw if calibrator is None else calibrator.predict_proba(logits(raw))[:, 1]
            cal_status.append({"family": family, "cutoff": cutoff, "eval_end": end, **status})
    probabilities["constant_train_rate"] = np.full(len(b), a.won.mean())
    if all_methods:
        probabilities["blend_50_50"] = .5*probabilities["logistic"] + .5*probabilities["boosting"]
        probabilities["lr_cat_50_50"] = .5*probabilities["logistic"] + .5*probabilities["catboost"]
        probabilities["lr_fixed_C_0_1"] = engine.predict(cutoff, end, lr_specs()[1])
    rows = []
    for method, p in probabilities.items():
        frame = base.prediction_rows(b, ab, p, float(a.won.mean()))
        frame["strategy"], frame["outer_period"] = method, f"{cutoff+1}-{end}"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(choices), pd.DataFrame(cal_status), pd.concat(inner_records, ignore_index=True)


def summarize(predictions, calibration):
    rows, periods = [], []
    for method, frame in predictions.groupby("strategy", sort=False):
        main, observed = base.summarize_rows(frame), base.summarize_rows(frame, "observed_only")
        row = {"strategy": method, **main, "observed_rows": observed["evaluated_rows"],
               "observed_score_usd": observed["score_usd"], "observed_always_usd": observed["always_usd"],
               "observed_constant_usd": observed["constant_usd"], "complexity": COMPLEXITY[method],
               "selectable": method != "lr_fixed_C_0_1"}
        if method.endswith("_calibrated"):
            family = method.removesuffix("_calibrated")
            row["selectable"] = bool(calibration.loc[calibration.family.eq(family), "adequate_by_protocol"].all())
        uplift = []
        for period, g in frame.groupby("outer_period", sort=False):
            m, o = base.summarize_rows(g), base.summarize_rows(g, "observed_only")
            periods.append({"strategy": method, "outer_period": period, **m, "observed_score_usd": o["score_usd"], "observed_rows": o["evaluated_rows"]})
            row["score_weeks_"+period] = m["score_usd"]
            uplift.append(m["delta_constant_usd"]/m["evaluated_rows"])
        row["worst_period_delta_constant_per_row"] = min(uplift)
        row["period_delta_constant_sd"] = float(np.std(uplift))
        rows.append(row)
    summary, periods = pd.DataFrame(rows), pd.DataFrame(periods)
    reference = periods.loc[periods.strategy.eq("logistic"), ["outer_period", "score_usd"]].rename(columns={"score_usd": "logistic_usd"})
    periods = periods.merge(reference, on="outer_period", validate="many_to_one")
    periods["delta_vs_logistic_usd"] = periods.score_usd-periods.logistic_usd
    periods["delta_vs_logistic_per1000"] = 1000*periods.delta_vs_logistic_usd/periods.evaluated_rows
    for method, group in periods.groupby("strategy"):
        mask = summary.strategy.eq(method)
        summary.loc[mask, "periods"] = len(group)
        summary.loc[mask, "periods_above_always"] = int(group.delta_always_usd.gt(0).sum())
        summary.loc[mask, "periods_above_logistic"] = int(group.delta_vs_logistic_usd.gt(1e-8).sum())
        summary.loc[mask, "worst_delta_vs_logistic_per1000"] = group.delta_vs_logistic_per1000.min()
        summary.loc[mask, "sd_delta_vs_logistic_per1000"] = group.delta_vs_logistic_per1000.std(ddof=0)
    baseline = summary.loc[summary.strategy.eq("logistic"), "score_usd"].iloc[0]
    summary["delta_vs_logistic_usd"] = summary.score_usd-baseline
    summary["uplift_vs_always_percent"] = 100*summary.delta_always_usd/summary.always_usd
    summary["uplift_vs_constant_percent"] = 100*summary.delta_constant_usd/summary.constant_usd
    summary["uplift_vs_logistic_percent"] = 100*summary.delta_vs_logistic_usd/baseline
    summary["score_per1000_usd"] = 1000*summary.score_usd/summary.evaluated_rows
    return summary.sort_values("score_usd", ascending=False), periods


def run(root):
    root = Path(root)
    output, protocol_hash = freeze(root)
    access = base.install_data_access_guard(root)
    data, audit, _ = base.load_train_only(root)
    engine = Engine(data, audit)
    tables = {}
    for regime, windows in [("primary", base.OUTER_FOLDS), ("weekly", [(w-1, w) for w in WEEKLY_WEEKS])]:
        results = []
        for cutoff, end in windows:
            folds = base.INNER_FOLDS[cutoff] if regime == "primary" else WEEKLY_INNER[cutoff]
            results.append(evaluate_cutoff(engine, cutoff, end, folds, regime == "weekly"))
            print(f"Completed {regime}: train through {cutoff}, evaluate through {end}", flush=True)
        predictions, choices, calibration, inner = [pd.concat([r[i] for r in results], ignore_index=True) for i in range(4)]
        summary, periods = summarize(predictions, calibration)
        for name, frame in {"predictions": predictions, "choices": choices, "calibration": calibration,
                            "inner_candidates": inner, "summary": summary, "periods": periods}.items():
            tables[regime+"_"+name] = frame
        # Усі методи мають ті самі спори та маски.
        reference = None
        for _, frame in predictions.groupby("strategy"):
            view = frame[["dispute_id", "outer_period", "amount_usd", "won", "eligible", "observed_only"]].reset_index(drop=True)
            if reference is None:
                reference = view
            else:
                pd.testing.assert_frame_equal(reference, view)
        if regime == "primary":
            old_manifest = json.loads((root/"train_only_outputs"/"run_evidence.json").read_text(encoding="utf-8"))
            old_path = root/"train_only_outputs"/"outer_predictions.csv"
            assert digest(old_path) == old_manifest["result_sha256"]["outer_predictions"]
            old = pd.read_csv(old_path, dtype={"dispute_id": "string"})
            for method in ["logistic", "logistic_calibrated"]:
                current = predictions.loc[predictions.strategy.eq(method)].reset_index(drop=True)
                historical = old.loc[old.strategy.eq(method)].reset_index(drop=True)
                assert current.dispute_id.tolist() == historical.dispute_id.tolist()
                np.testing.assert_allclose(current.p, historical.p, atol=1e-10)
        selection = base.select_from_summary(summary)
        (output/f"{regime}_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    tables["fit_log"] = pd.DataFrame(engine.fit_log)
    for name, frame in tables.items():
        frame.to_csv(output/f"{name}.csv", index=False)
    evidence = {"protocol_sha256": protocol_hash, "actual_source_csv_open_events": access,
                "baseline_predictions_reproduced": True, "validation_opened": False, "test_opened": False,
                "code_sha256": {n: digest(root/n) for n in ["weighted_weekly_study.py", "train_only_study.py", "interaction_study.py", "catboost_study.py", "model_comparison.py", "preparation.py"]},
                "result_sha256": {n: digest(output/f"{n}.csv") for n in tables}}
    (output/"run_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return tables["primary_summary"], tables["weekly_summary"], tables["weekly_periods"]


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent
    if "--freeze-only" in sys.argv:
        print(freeze(root)[1])
    else:
        run(root)
