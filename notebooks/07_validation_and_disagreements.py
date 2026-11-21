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
# # Validation: попередні стратегії, CatBoost і розбіжності рішень
#
# Validation уже використано для оцінювання та вибору в ноутбуці 04. У ноутбуках
# 05–06 застосовувалися лише дані train, тому CatBoost на validation не оцінювався.
# У цьому ноутбуці нові стратегії додаються до спільної таблиці результатів на
# validation. Test не відкривається; validation є повторно використаною вибіркою.
#
# Параметри нових моделей визначаються лише на train (тижні 0–17), після чого моделі
# оцінюються на validation (тижні 18–21). Попередні прогнози на validation не
# перераховуються: їхні контрольні суми та Score перевіряються, а рядки додаються
# до об'єднаної таблиці.
#
# Вхідні дані: `outputs/`, `train_only_outputs/`, `comparison_outputs/`. Результати
# записуються в `validation_extension_outputs/`. Файли з окремими спорами не
# публікуються.

# %%
import matplotlib
import numpy as np
import pandas as pd

from disputes.paths import PROJECT as PROJECT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

from disputes.validation_extension import freeze, run

OUTPUT, protocol_hash = freeze(PROJECT_DIR)
print("Протокол до нових validation-прогнозів:", protocol_hash)

# %% [markdown]
# ## Параметри, визначені на train
#
# - C логістичної регресії (ЛР): найбільший сумарний Score на трьох часових зрізах
#   train із перевіреної сітки C; за рівності — log loss, далі менше C.
# - CatBoost: чотири комбінації depth/l2 на тих самих трьох зрізах train; вибір за
#   сумарним Score, далі за pooled log loss.
# - Sigmoid-калібратори: навчаються лише на out-of-time прогнозах train для обраних
#   параметрів; мітки validation не використовуються.
# - Ансамбль: середнє некаліброваних прогнозів ЛР і CatBoost з вагами 0,5 і 0,5.
#   Ваги та поріг правила рішення за validation не підбираються.
#
# Параметри та коефіцієнти калібрування записуються в `train_parameter_lock.json`
# до відкриття validation. Рядок `logistic` відповідає попередній ЛР із C = 1;
# рядок `logistic_train_tuned` — ЛР із C, обраним на train. Для всіх стратегій
# застосовується правило рішення p × 0,2 × amount_usd > 6; за невідомої суми —
# `review_amount`.
#
# Повторне використання validation не робить його незалежною фінальною перевіркою.
# Наведені нижче розрізи є діагностикою й не використовуються для вибору моделі.

# %%
summary, pairs, groups, locked = run(PROJECT_DIR)
print("Параметри, визначені лише на train:")
print("Логістична:", locked["logistic_spec"])
print("CatBoost:", locked["catboost_spec"])
columns = ["model", "historical_row", "score_usd", "delta_always_usd", "delta_constant_usd",
           "uplift_vs_always_percent", "log_loss", "brier", "roc_auc",
           "observed_score_usd", "evaluated_rows", "observed_evaluated_rows",
           "score_week_18", "score_week_19", "score_week_20", "score_week_21"]
display(summary[columns].round(5))
print("historical_row=True: збережений рядок попередньої validation-таблиці")

# %% [markdown]
# ## Види розбіжностей між моделями
#
# Розрізняються три рівні розбіжностей:
#
# 1. Різні ймовірності (наприклад, 0,7 і 0,8). Дія за правилом рішення при цьому може
#    збігатися, тож різниця ймовірностей не обов'язково змінює Score.
# 2. Різні класи за порогу 0,5. Цей поріг використовується для аналізу помилок
#    класифікації й не є правилом рішення.
# 3. Різні дії за правилом рішення p × 0,2 × amount_usd > 6. Лише ці спори
#    зумовлюють різницю Score; у групах з однаковими діями різниця Score дорівнює нулю.
#
# Для кожної пари моделей наводяться кількість розбіжностей класів і дій, кореляція
# ймовірностей і залишків won − p, спільні та відмінні помилки, а також внесок
# розбіжних рішень у Score. Класифікаційні підрахунки охоплюють усі спори validation,
# грошові — лише спори із сумою, придатною для Score; окремо наведено сценарій
# observed. Кореляція ймовірностей не вимірює різниці Score.

# %%
main_pairs = pairs.loc[
    pairs.scenario.eq("score_eligible") &
    (((pairs.model_a == "logistic") & (pairs.model_b == "catboost")) |
     ((pairs.model_a == "logistic_train_tuned") & (pairs.model_b == "catboost")) |
     ((pairs.model_a == "logistic_train_tuned") & (pairs.model_b == "lr_tuned_cat_50_50")) |
     ((pairs.model_a == "catboost") & (pairs.model_b == "catboost_calibrated")))
]
display(main_pairs[["model_a", "model_b", "classification_rows", "class_disagreements_at_0_5",
                    "evaluated_rows", "action_disagreements", "a_fight_b_skip", "a_skip_b_fight",
                    "delta_b_minus_a_usd", "near_boundary_disagreements",
                    "only_a_correct", "only_b_correct", "both_wrong", "residual_correlation"]].round(5))
print("Декомпозиція рішень: тільки основний грошовий сценарій")
display(groups.loc[groups.scenario.eq("score_eligible")].round(4))
print("Рядки конкретних спорів збережені локально в disagreement_details.csv.")
print("Там є ID, обидві ймовірності, дії, won, сума, межа вигідності та фактичний внесок у різницю доходу.")

# %% [markdown]
# ## Використання розбіжностей
#
# Розбіжності показують, наскільки моделі доповнюють одна одну, чи залежить різниця
# Score від спорів поблизу межі рішення, у яких групах вона виникає та чи змінює
# результат фіксований ансамбль. Ансамбль оцінюється на тих самих рядках за Score.
#
# Для нового спору результат won невідомий. Вибір моделі, яка правильно передбачила
# відомий результат на validation, є витоком цілі. Модель або правило вибору між
# моделями (наприклад, за провайдером чи причиною) слід визначати на out-of-time
# прогнозах train і перевіряти на окремій вибірці; у цьому експерименті такого
# механізму немає.
#
# Ознака `near_boundary` означає, що хоча б один із двох прогнозів очікуваного чистого
# доходу відрізняється від нуля не більше ніж на 1 USD. Цей діагностичний поріг не
# впливає на рішення. Висновки щодо сегментів із малою кількістю спорів мають низьку
# точність; наведені асоціації не встановлюють причинності.

# %%
weekly = pd.read_csv(OUTPUT / "validation_weekly.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
view = summary.sort_values("score_usd")
axes[0].barh(view.model, view.score_usd, color=np.where(view.historical_row, "#7895b2", "#348c68"))
axes[0].axvline((summary.score_usd - summary.delta_always_usd).iloc[0], linestyle="--", color="gray",
                label="Always fight")
axes[0].set(xlabel="Validation Score, USD", title="Historical rows + train-selected new models")
axes[0].legend(fontsize=8)
for name in ["logistic", "logistic_train_tuned", "catboost", "catboost_calibrated", "lr_tuned_cat_50_50"]:
    group = weekly.loc[weekly.model.eq(name)]
    axes[1].plot(group.week, group.score_usd, "o-", label=name)
axes[1].set(xlabel="Validation week", ylabel="Score, USD", title="Same eligible disputes within each week")
axes[1].set_xticks([18, 19, 20, 21])
axes[1].legend(fontsize=7)
fig.tight_layout()
path = OUTPUT / "validation_extended_comparison.png"
fig.savefig(path, dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## Обмеження
#
# Результати отримано на повторно використаному validation, а не на test. Попередні
# значення відтворено, нові параметри визначено на train, доступ до файлів test
# заблоковано. Повторний запуск за тим самим протоколом і незалежний перерахунок Score
# за рядками перевіряють коректність реалізації, а не майбутній дохід. Дата завершення
# спорів невідома, тому доступність міток у момент прогнозу не підтверджена.
#
# Файли результатів: `validation_summary.csv` (зведена таблиця),
# `pairwise_disagreement.csv` (усі пари), `disagreement_details.csv` (детальні рядки
# для чотирьох пар), `decision_disagreement_groups.csv` (групи рішень),
# `disagreement_segments.csv` (розрізи за PSP, причиною та методом оплати). Ці файли
# не публікуються.
#
# Усереднення ймовірностей: [scikit-learn, soft voting](https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting).
