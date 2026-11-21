"""Метрики, грошові підсумки та правило вибору стратегії."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score

from disputes.config import PRACTICAL_GAP_USD_PER_1000
from disputes.economics import payoff, policy, profit_rows
from disputes.preparation import calculate_baseline


def classification_metrics(y, p) -> dict:
    """Метрики ймовірностей і класу при діагностичному порозі 0,5."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    (tn, fp), (fn, tp) = confusion_matrix(y, p >= .5, labels=[0, 1])
    ratio = lambda a, b: float(a / b) if b else 0.
    return {"rows": len(y), "win_rate": float(y.mean()), "roc_auc": roc_auc_score(y, p),
            "average_precision_won": average_precision_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p), "accuracy": ratio(tp + tn, len(y)),
            "precision_won": ratio(tp, tp + fp), "recall_won": ratio(tp, tp + fn),
            "precision_lost": ratio(tn, tn + fn), "recall_lost": ratio(tn, tn + fp),
            "true_won": int(tp), "false_won": int(fp), "false_lost": int(fn), "true_lost": int(tn)}


def probability_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    return {"classification_rows": len(y), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan}


def evaluate(data: pd.DataFrame, audit: pd.DataFrame, p) -> tuple[dict, list[dict]]:
    """Класифікаційні метрики та Score для двох сценаріїв сум (усі придатні / лише незмінені)."""
    classification = classification_metrics(data.won, p)
    profits, actions = profit_rows(data, audit, p), policy(data, audit, p)
    economics = []
    for scenario, flag in [("eligible", "score_eligible"), ("observed_only", "score_observed_only")]:
        mask = audit[flag].to_numpy(dtype=bool)
        always = calculate_baseline(data, audit, observed_only=scenario == "observed_only")["score_usd"]
        score = float(profits[mask].sum())
        economics.append({"scenario": scenario, "rows": int(mask.sum()), "coverage": float(mask.mean()),
                          "fight_rows": int(((actions == "fight") & mask).sum()), "score_usd": score,
                          "always_usd": always, "delta_vs_always_usd": score - always})
    return classification, economics


def segment_scores(data: pd.DataFrame, audit: pd.DataFrame, p, segment) -> pd.DataFrame:
    """Score моделі й «боротися завжди» за групами; NaN, якщо в групі немає придатних сум."""
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    frame = pd.DataFrame({
        "segment": pd.Series(np.asarray(segment), dtype="string").fillna("MISSING").to_numpy(),
        "won": data.won.to_numpy(dtype=int), "p": np.asarray(p, dtype=float), "eligible": eligible,
        "fight": (policy(data, audit, p) == "fight") & eligible,
        "profit": profit_rows(data, audit, p), "always": np.where(eligible, payoff(data), np.nan)})
    frame["delta"] = frame.profit - frame.always
    result = frame.groupby("segment", sort=True).agg(
        rows=("won", "size"), eligible_rows=("eligible", "sum"), win_rate=("won", "mean"),
        mean_probability=("p", "mean"), fight_rows=("fight", "sum"),
        score_usd=("profit", lambda s: s.sum(min_count=1)), always_usd=("always", lambda s: s.sum(min_count=1)),
        delta_vs_always_usd=("delta", lambda s: s.sum(min_count=1))).reset_index()
    result["fight_rate"] = result.fight_rows / result.eligible_rows.replace(0, np.nan)
    return result


def merchant_status(data: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Чи бачила модель цього мерчанта в train: seen / unseen / missing."""
    seen = data.merchant_id.isin(train.merchant_id.dropna())
    return pd.Series(np.where(data.merchant_id.isna(), "missing", np.where(seen, "seen", "unseen")), index=data.index)


def calibration_bins(y, p, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"won": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.cut(frame.p, np.linspace(0, 1, bins + 1), include_lowest=True)
    result = frame.groupby("bin", observed=True).agg(rows=("won", "size"), mean_probability=("p", "mean"),
                                                     win_rate=("won", "mean")).reset_index()
    return result.assign(bin=result.bin.astype(str))


# --------------------------------------------------------------------------- рядкові прогнози стратегій
def prediction_rows(data: pd.DataFrame, audit: pd.DataFrame, p, constant_p: float) -> pd.DataFrame:
    """Прогноз, дохід стратегії, сталої ймовірності та «боротися завжди» для кожного спору."""
    eligible = audit.score_eligible.to_numpy(dtype=bool)
    result = data[["dispute_id", "dispute_week", "won", "amount_usd"]].copy()
    result["p"] = p
    result["eligible"] = eligible
    result["observed_only"] = audit.score_observed_only.to_numpy(dtype=bool)
    result["profit"] = profit_rows(data, audit, p)
    result["constant_profit"] = profit_rows(data, audit, np.full(len(data), constant_p))
    result["always_profit"] = np.where(eligible, payoff(data), np.nan)
    return result.reset_index(drop=True)


def summarize_rows(rows: pd.DataFrame, flag: str = "eligible") -> dict:
    mask = rows[flag].to_numpy(dtype=bool)
    result = probability_metrics(rows.won, rows.p.to_numpy(dtype=float))
    total = lambda col: float(rows.loc[mask, col].sum()) if mask.any() else np.nan
    score, always, constant = total("profit"), total("always_profit"), total("constant_profit")
    return {**result, "evaluated_rows": int(mask.sum()), "score_usd": score, "always_usd": always,
            "constant_usd": constant, "delta_always_usd": score - always, "delta_constant_usd": score - constant}


def _with_observed(rows: pd.DataFrame) -> dict:
    observed = summarize_rows(rows, "observed_only")
    return {**summarize_rows(rows), **{"observed_" + k: observed[k] for k in
            ["score_usd", "always_usd", "constant_usd", "delta_always_usd", "delta_constant_usd"]},
            "observed_rows": observed["evaluated_rows"]}


def summarize_strategies(predictions: pd.DataFrame, by: str = "strategy", complexity: dict | None = None,
                         not_selectable=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Підсумок кожної стратегії та кожного зовнішнього періоду, зі стабільністю між періодами."""
    rows, periods = [], []
    for name, frame in predictions.groupby(by, sort=False):
        row = {"strategy": name, **_with_observed(frame),
               "complexity": (complexity or {}).get(name, np.nan), "selectable": name not in not_selectable}
        uplift = []
        for period, group in frame.groupby("outer_period", sort=False):
            summary = _with_observed(group)
            periods.append({"strategy": name, "outer_period": period, **summary})
            row[f"score_weeks_{period}"] = summary["score_usd"]
            row[f"rows_weeks_{period}"] = summary["evaluated_rows"]
            uplift.append(summary["delta_constant_usd"] / summary["evaluated_rows"])
        row["worst_period_delta_constant_per_row"] = min(uplift)
        row["period_delta_constant_sd"] = float(np.std(uplift))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score_usd", ascending=False), pd.DataFrame(periods)


def assert_same_rows(predictions: pd.DataFrame, by: str = "strategy",
                     columns=("dispute_id", "outer_period", "won", "amount_usd", "eligible", "observed_only")) -> None:
    """Усі стратегії оцінені на тих самих спорах, сумах і масках."""
    views = [frame[list(columns)].reset_index(drop=True) for _, frame in predictions.groupby(by)]
    for view in views[1:]:
        pd.testing.assert_frame_equal(view, views[0], check_dtype=False)


def select_from_summary(summary: pd.DataFrame) -> dict:
    """Наперед задане правило: серед стратегій у межах практичного допуску від максимуму — простіша."""
    candidates = summary.loc[summary.selectable].copy()
    if candidates.empty:
        raise ValueError("Немає придатної стратегії")
    if candidates.evaluated_rows.nunique() != 1:
        raise ValueError("Стратегії оцінені на різних кількостях рядків")
    n = int(candidates.evaluated_rows.iloc[0])
    margin = PRACTICAL_GAP_USD_PER_1000 / 1000 * n
    maximum = float(candidates.score_usd.max())
    candidates["gap_to_max_usd"] = maximum - candidates.score_usd
    near = candidates.loc[candidates.gap_to_max_usd.le(margin + 1e-9)].sort_values(
        ["complexity", "worst_period_delta_constant_per_row", "period_delta_constant_sd", "log_loss", "strategy"],
        ascending=[True, False, True, True, True])
    return {"selected_strategy": str(near.strategy.iloc[0]),
            "maximum_score_strategy": str(candidates.loc[candidates.score_usd.idxmax(), "strategy"]),
            "maximum_score_usd": maximum, "practical_margin_usd": margin, "evaluated_rows": n,
            "near_best_strategies": near.strategy.tolist(), "selected_gap_to_max_usd": float(near.gap_to_max_usd.iloc[0])}
