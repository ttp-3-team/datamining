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
# # Взаємодії ознак у логістичній регресії — лише train
#
# Гіпотеза: спільна комбінація провайдера й валюти або причини спору може давати
# інформацію понад окремі внески цих категорій. Перевіряємо тільки цей пункт.
# Validation і test не відкриваються; попередні результати не перезаписуються.
#
# Чотири варіанти з однаковими базовими ознаками:
#
# 1. baseline — без взаємодій;
# 2. psp_currency — додано провайдер × валюта;
# 3. psp_reason — додано провайдер × категорія причини;
# 4. both_interactions — додано обидві взаємодії.
#
# Попередній фіксований C=0.1 показано окремим діагностичним контролем.
# Він уже був обраний після перегляду цих train-періодів, тому не бере участі
# в новому виборі як нібито незалежно зафіксований кандидат.
#
# Середовище — `uv sync`. Для запуску потрібні
# outputs/train_audit.csv та результати train_only_outputs/ з notebook 05.
# Виконати Restart and Run All. Усі нові виходи — interaction_outputs/.

# %%
from pathlib import Path
from disputes.paths import PROJECT as PROJECT_DIR
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from disputes.interaction_study import freeze, run, VARIANTS

OUTPUT, protocol_hash = freeze(PROJECT_DIR)
print("Протокол зафіксовано до нових результатів:", protocol_hash)
print("Варіанти:", VARIANTS)

# %% [markdown]
# ## Кодування та чесне порівняння
#
# Взаємодія — додаткова категоріальна ознака, наприклад [провайдер, USD].
# Базові категорії залишаються в моделі. Пари кодуються JSON-списком, щоб
# розділювачі всередині назв не могли випадково склеїти різні комбінації.
# One-hot словник навчається лише на навчальній частині кожного зрізу.
# Нова комбінація у пізнішому періоді не додається в словник; її додатковий
# one-hot блок нульовий, а відомі базові ознаки продовжують працювати.
#
# Для кожного варіанта C=0.01/0.1/1/10/100 підбирається окремо на внутрішніх
# часових зрізах. Вибір — найбільший pooled внутрішній Score, потім менший
# pooled log loss і менший C. Це однаковий бюджет налаштування для всіх варіантів.
# Немає калібрування, нових ваг спостережень, пошуку бізнес-порогів чи ансамблів.
#
# Зовнішні періоди: 0–5 → 6–9, 0–9 → 10–13, 0–13 → 14–17.
# Усередині навчальних частин — ті самі два внутрішні зрізи, що в notebook 05.
# Навіть опора очищення сум визначається заново на ранній частині кожного зрізу.
#
# Правило вибору залишається незмінним: варіанти в межах 50 USD на 1 000
# оцінених спорів від максимуму вважаються практично близькими. Перевага
# меншій кількості доданих взаємодій. За однакової складності — кращий
# найгірший період приросту до постійної стратегії на один спір, менший розкид,
# менший log loss. Це практичне правило, не статистичний тест.

# %%
summary, periods, choices, coverage, selection = run(PROJECT_DIR)
columns = ["strategy", "score_usd", "delta_vs_baseline_usd", "delta_vs_fixed_C_0_1_usd",
           "delta_always_usd", "delta_constant_usd", "score_weeks_6-9", "score_weeks_10-13",
           "score_weeks_14-17", "log_loss", "brier", "roc_auc", "observed_score_usd",
           "observed_delta_vs_baseline_usd", "evaluated_rows", "observed_rows", "selectable"]
display(summary[columns].round(5))
print("Параметри й кількість закодованих ознак для зовнішніх прогнозів")
display(choices)
print("Нові комбінації категорій у пізніших періодах")
display(coverage)
display(periods.round(5))

# %% [markdown]
# ## Інтерпретація результату
#
# Основне порівняння — з baseline, у якого C обирається тільки за минулими
# даними відповідного зовнішнього періоду. Його прогнози автоматично
# звіряються з попереднім вкладеним train-only дослідженням.
# Фіксований C=0.1 — додаткова відома контрольна цифра, а не заміна чесного
# вкладеного baseline. Результат його попереднього глобального підбору на цих
# даних не можна видавати за незалежний зовнішній вибір параметра.
#
# Метрики ймовірностей включають усі зовнішні спори, Score — однакові рядки
# з придатними сумами. Observed-сценарій виключає відновлені/виправлені USD-суми;
# він є перевіркою чутливості, не окремим критерієм підбору.
# Коефіцієнти збережено для перевірки моделі; вони описують внески в log-odds,
# а не причинний ефект провайдера, валюти чи причини спору.

# %%
print("Вибір за протоколом:", selection["selected_strategy"])
print("Максимальний Score серед допустимих варіантів:", selection["maximum_score_strategy"])
print("Поріг практичної близькості, USD:", selection["practical_margin_usd"])
print("Відставання обраного від максимуму, USD:", selection["selected_gap_to_max_usd"])
print("Близькі варіанти:", selection["near_best_strategies"])
if selection["selected_strategy"] == "baseline":
    print("За зафіксованим правилом взаємодії не виправдали додаткової складності.")
else:
    print("Вибрана взаємодія потребує подальшої зовнішньої перевірки за зафіксованим рецептом.")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
for variant, group in periods.groupby("strategy", sort=False):
    axes[0].plot(group.outer_period, group.delta_constant_usd/group.evaluated_rows, "o-", label=variant)
axes[0].set(title="Same nested train-only folds", xlabel="Outer evaluation weeks",
            ylabel="Uplift vs train-rate policy, USD/dispute")
axes[0].legend(fontsize=7)
view = summary.sort_values("delta_vs_baseline_usd")
axes[1].barh(view.strategy, view.delta_vs_baseline_usd)
axes[1].axvline(0, color="gray", linestyle="--")
axes[1].set(title="Difference from nested baseline", xlabel="Score difference, USD")
fig.tight_layout()
figure = OUTPUT / "interaction_comparison.png"
fig.savefig(figure, dpi=150)
plt.close(fig)
display(HTML('<img alt="Train-only logistic interactions" width="1100" src="data:image/png;base64,'
             + base64.b64encode(figure.read_bytes()).decode() + '">'))

# %% [markdown]
# ## Межі висновку та відтворення
#
# Ці train-періоди вже аналізували раніше, зокрема під час вибору гіпотез.
# Вкладена перевірка контролює локальний підбір C, але не скасовує повторного
# використання даних. Три зовнішні періоди не доводять майбутню стійкість.
# Дата отримання результату спору відсутня, тому доступність навчальних міток
# на момент прогнозу не підтверджена. Validation уже використовувався раніше,
# проте тут його файли не відкриваються. Test також не відкривається.
#
# `protocol.json` містить зафіксований план; `strategy_summary.csv` — єдину таблицю;
# `outer_choices.csv` — C й розмір моделей; `interaction_coverage.csv` — нові
# комбінації; `coefficients.csv` — коефіцієнти; `run_evidence.json` — хеші й
# журнал доступу до початкових CSV. Виводи та ID-прогнози не публікувати.
#
# Перевірки: `uv run pytest tests/test_interaction_study.py`.
# Перевірка notebook без Jupyter: `uv run python scripts/check_notebooks.py 08_logistic_interactions`.
# Повторний запуск із тим самим протоколом не змінює правила вибору.
