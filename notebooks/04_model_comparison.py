# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Порівняння моделей на validation
#
# Ноутбук порівнює ЛР і HistGradientBoosting за Score до оцінювання на test. Test не завантажується й не оцінюється.
#
# Набір даних конфіденційний: дані, прогнози з ID і ноутбуки з рядками даних не публікуються.
#
# Середовище: `uv sync`; усі комірки виконуються в чистій сесії. Потрібна тека `outputs/`, створена ноутбуком 02. Результати записуються в `comparison_outputs/`.

# %%
import json

import matplotlib
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disputes.economics import policy
from disputes.evaluation import calibration_bins
from disputes.model_comparison import (
    CANDIDATES,
    FOLDS,
    cross_validate,
    export_run,
    fit_finalists,
    load_inputs,
    paired_block_intervals,
    permutation_diagnostics,
    segment_diagnostics,
    week_diagnostics,
)
from disputes.models import features
from disputes.paths import PROJECT as PROJECT_DIR

OUTPUT = PROJECT_DIR / "comparison_outputs"
OUTPUT.mkdir(exist_ok=True)
train, train_audit, validation, validation_audit, normalized, source_audit = load_inputs(PROJECT_DIR)
print("Train:", len(train), "Validation:", len(validation))
print("Test і unknown_time не завантажувалися")
# The plan is written before the search and is not changed after results are seen.
plan = {
    "candidates": CANDIDATES, "folds": FOLDS, "iterations": 400, "learning_rate": .05,
    "cv_selection": "pooled Score, mean log loss as tie breaker",
    "validation_finalists": ["logistic", "boosting", "boosting_calibrated", "blend_50_50"],
    "decision": "p * 0.2 * amount_usd > 6", "test_evaluated": False,
}
(OUTPUT / "experiment_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
display(pd.DataFrame(CANDIDATES))

# %% [markdown]
# ## 1. Часова перевірка на train
#
# Три послідовні зрізи: тижні 0–5 → оцінювання на 6–9; 0–9 → 10–13; 0–13 → 14–17. Для кожного зрізу опорні співвідношення для очищення сум визначаються заново лише на ранній частині; початкові значення train відновлюються з аудиту. Використання очищених сум з опорою на всі тижні 0–17 означало б доступ до статистики майбутніх тижнів.
#
# Перетворення ознак також навчаються лише на ранній частині. HistGradientBoosting: 400 ітерацій, learning rate 0,05, без early stopping. Кожен зріз оцінюється лише на пізніших тижнях. Пропущені категорії отримують значення `MISSING`, пропуски числових ознак модель обробляє сама. Категорії кодуються OrdinalEncoder, а модель отримує явний перелік категоріальних ознак, тому номери кодів не трактуються як числова шкала; нові категорії перетворюються на пропуск. Словник категорій навчається лише на train. Балансування класів і агрегати цілі не використовуються.
#
# Порівнюються глибини 4 і 6, а також додавання суми в оригінальній валюті, прапорців якості, номера тижня та вилучення ID продавця. За фіксованої глибини це окремі перевірки внеску ознак, а не повний перебір комбінацій. Обирається конфігурація HistGradientBoosting з найбільшим сумарним Score на однакових зрізах; за рівності — з меншим середнім log loss. Ці оцінки використано для вибору, тому вони не є незалежним результатом.
#
# Часовий поділ не гарантує доступності міток на момент рішення, оскільки дата завершення спору відсутня.

# %%
cv_scores, cv_ranking, best_spec, oof, folds = cross_validate(normalized, source_audit)
display(folds)
display(cv_ranking.round(4))
print("Обрана конфігурація HistGradientBoosting:", best_spec)
display(cv_scores.query("scenario == 'eligible'").pivot(index="candidate", columns="fold", values="score_usd").round(2))

# %% [markdown]
# ## 2. Фіналісти на validation
#
# Обрана конфігурація HistGradientBoosting і ЛР навчаються на тижнях 0–17. Додатково перевіряється sigmoid-калібрування HistGradientBoosting: калібратор навчається лише на позачасових прогнозах усередині train; перенесення цієї корекції на модель, навчену на більшій вибірці, не гарантоване. Ансамбль — середнє ймовірностей ЛР і HistGradientBoosting з рівними вагами; ваги не підбираються.
#
# Validation використовується для вибору серед чотирьох фіксованих варіантів за Score (за рівності — за log loss). Тому наведені нижче значення є оцінками для вибору, а не незалежним підтвердженням якості. Застосовується правило рішення p × 0,2 × amount_usd > 6; поріг 0,5 використовується лише для класифікаційних метрик.

# %%
logistic, boosting, calibrator, probabilities, metrics, economics, selected = fit_finalists(
    train, train_audit, validation, validation_audit, best_spec, oof)
display(metrics.round(4))
display(economics.round(2))
print("Вибраний на validation варіант:", selected)
display(pd.DataFrame(confusion_matrix(validation.won.astype(int), probabilities[selected] >= .5),
                     index=["actual_lost", "actual_won"], columns=["predicted_lost", "predicted_won"]))

# %% [markdown]
# ## 3. Відтворення базової моделі, стійкість і невизначеність
#
# Перевіряються: відтворення прогнозів базової ЛР, відповідність рядкового Score таблиці, незалежність рішень від `won`, частка спорів із сумою, придатною для Score. Дохід спорів з невідомою сумою не замінюється медіаною чи нулем.
#
# Наведено Score за тижнями та парні блокові bootstrap-інтервали різниці Score (перевибір тижнів або продавців, 2000 повторів). Чотири тижні validation недостатні для оцінки часової невизначеності; перевибір продавців не враховує спільних часових ефектів. Інтервали описові, умовні щодо навчених моделей і не враховують зміщення від вибору найкращого варіанта на тих самих даних.

# %%
baseline_path = PROJECT_DIR / "model_outputs" / "validation_predictions.csv"
if baseline_path.exists():
    old = pd.read_csv(baseline_path, dtype={"dispute_id": "string"})
    assert old.dispute_id.tolist() == validation.dispute_id.tolist()
    np.testing.assert_allclose(old.p_won, probabilities["logistic"], atol=1e-10)
    print("Попередні прогнози логістичної регресії відтворено")
else:
    print("Попереднього експорту немає; історичне зіставлення пропущено")

p_selected = probabilities[selected]
actions = policy(validation, validation_audit, p_selected)
changed_target = validation.copy()
changed_target.won = 1 - changed_target.won
np.testing.assert_array_equal(actions, policy(changed_target, validation_audit, p_selected))
pd.testing.assert_frame_equal(features(validation, validation_audit, best_spec)[0],
                             features(changed_target, validation_audit, best_spec)[0])
weekly = week_diagnostics(validation, validation_audit, probabilities)
intervals = paired_block_intervals(validation, validation_audit, probabilities, selected)
display(weekly.pivot(index="model", columns="week", values="delta_vs_always_usd").round(2))
display(intervals.round(2))
print("Без надійної суми:", int((actions == "review_amount").sum()))

# %% [markdown]
# ## 4. Діагностика ознак і сегментів
#
# Permutation importance — погіршення log loss HistGradientBoosting на validation після перемішування однієї ознаки (5 повторів). Показник описує асоціації в моделі, а не причинний вплив; корельовані ознаки можуть заміщати одна одну. Після перегляду діагностик нові моделі в цьому ноутбуці не підбираються.
#
# Сегменти обраної стратегії: спосіб оплати, причина спору, сума, відомі й нові продавці. Результати на малих групах нестійкі; за відсутності нових продавців узагальнення на них не перевірено.

# %%
importance = permutation_diagnostics(boosting, validation, validation_audit, best_spec)
segments = segment_diagnostics(validation, validation_audit, p_selected, train)
display(importance.round(4))
display(segments.round(3))
calibration = pd.concat([calibration_bins(validation.won, p).assign(model=name)
                         for name, p in probabilities.items()], ignore_index=True)
display(calibration.loc[calibration.model.eq(selected)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name in ["logistic", "boosting", "boosting_calibrated", "blend_50_50"]:
    group = calibration.loc[calibration.model.eq(name)]
    axes[0].plot(group.mean_probability, group.win_rate, "o-", label=name, markersize=4)
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set(xlabel="Predicted win probability", ylabel="Observed win rate", title="Validation calibration")
axes[0].legend(fontsize=8)
econ = economics.query("scenario == 'eligible'")
axes[1].barh(econ.model, econ.score_usd)
axes[1].axvline(econ.always_usd.iloc[0], color="gray", linestyle="--", label="Always fight")
axes[1].set(xlabel="Score, USD", title=f"Same {int(econ.rows.iloc[0]):,} eligible validation disputes")
axes[1].legend()
fig.tight_layout()
figure = OUTPUT / "validation_comparison.png"
fig.savefig(figure, dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## 5. Експорт і зафіксований вибір
#
# Зберігаються журнал усіх кандидатів і всі результати. Після збереження перевіряються повторне завантаження моделей і тотожність прогнозів. Рядки validation з ID і моделі призначені лише для локального відтворення.
#
# Правило рішення в межах припущень задачі: оцінити ймовірність виграшу і порівняти очікуваний дохід із вартістю дії $6; для спорів з відсутньою сумою спершу уточнити суму. Перед практичним застосуванням потрібно перевірити фактичну вартість дії, доступність ознак на момент рішення та зміну складу спорів. Результат є історичною симуляцією на спорах, за які боролися, а не оцінкою причинного ефекту для всіх спорів.

# %%
output, config = export_run(
    PROJECT_DIR, best_spec, (logistic, boosting, calibrator), probabilities, metrics, economics,
    selected, cv_scores, cv_ranking, folds, oof, weekly, intervals, importance, segments,
    validation, validation_audit)
calibration.to_csv(output / "validation_calibration.csv", index=False)
row = economics.query("scenario == 'eligible' and model == @selected").iloc[0]
lr_row = economics.query("scenario == 'eligible' and model == 'logistic'").iloc[0]
summary = (f"Selected on validation: {selected}\n"
           f"HistGradientBoosting candidate selected by temporal CV: {best_spec['name']}\n"
           f"Validation Score: {row.score_usd:.2f} USD on {int(row.rows)} disputes\n"
           f"Delta vs always: {row.delta_vs_always_usd:+.2f} USD\n"
           f"Delta vs logistic: {row.score_usd - lr_row.score_usd:+.2f} USD\n"
           "Validation used for selection; test has not been evaluated.\n")
(output / "summary.txt").write_text(summary, encoding="utf-8")
print(summary)
print("Повторне завантаження моделей і контроль Score: OK")
print("Лідер внутрішнього часового CV:", cv_ranking.index[0])
logistic_intervals = intervals.query("reference == 'logistic'")
if ((logistic_intervals.p025_usd <= 0) & (logistic_intervals.p975_usd >= 0)).any():
    print("Описові інтервали різниці з логістичною регресією охоплюють 0: стійку перевагу не встановлено.")
print("Формальний вибір за validation Score не замінює фінальної перевірки на test.")

# %% [markdown]
# ## Джерела методів
#
# [HistGradientBoosting: категорії та пропуски](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting), [калібрування](https://scikit-learn.org/stable/modules/calibration.html), [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
