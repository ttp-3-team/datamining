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
# # CatBoost: додаткове порівняння на train
#
# У ноутбуці порівнюється модель CatBoost зі стратегіями ноутбука 05 на тих самих
# зовнішніх часових періодах. Validation і test не використовуються; попередні
# результати не перезаписуються. Експеримент виконано після перегляду результатів
# на train, тому він не є незалежною підтверджувальною перевіркою.
#
# Розглянуті алгоритми:
#
# - логістична регресія (ЛР) — регуляризована модель, у якій логарифм суми та
#   закодовані категорії мають адитивний внесок у логарифм шансів; взаємодії ознак
#   не додано;
# - HistGradientBoosting — бустинг дерев, що враховує нелінійності та взаємодії ознак;
# - CatBoost — бустинг дерев із власною обробкою категоріальних ознак та
#   Ordered boosting.
#
# Вхідні дані: `outputs/train_audit.csv` і результати ноутбука 05 у
# `train_only_outputs/`. Середовище встановлюється командою `uv sync`.

# %%
import matplotlib
import pandas as pd

from disputes.paths import PROJECT as PROJECT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

from disputes.catboost_study import SPECS, freeze, run

OUTPUT, protocol_sha256 = freeze(PROJECT_DIR)
print("Протокол зафіксовано до нових результатів:", protocol_sha256)
display(pd.DataFrame(SPECS))

# %% [markdown]
# ## Схема перевірки
#
# Зовнішні періоди: навчання на тижнях 0–5 → оцінювання на 6–9, 0–9 → 10–13,
# 0–13 → 14–17. Для кожної зовнішньої навчальної частини два внутрішні зрізи
# обирають параметри лише за її даними. Опорні значення очищення сум визначаються
# окремо в кожному зрізі.
#
# Кандидати CatBoost: глибина 4 або 6 × l2 3 або 10; 500 дерев, learning rate 0,05,
# seed 42, 4 потоки CPU, Ordered boosting, has_time=True. Рядки впорядковано за
# тижнем; порядок усередині тижня невідомий. Early stopping, агрегати цілі та
# балансування класів не застосовуються. Категоріальні пропуски кодуються як MISSING,
# числові — як NaN. Набір ознак збігається з набором ЛР.
#
# Параметри обираються за сумарним внутрішнім Score; за рівності — за pooled log loss,
# далі за назвою кандидата. Sigmoid-калібратор навчається лише на внутрішніх
# out-of-time прогнозах; вимоги до обсягу калібрування ті самі, що в ноутбуці 05.
# Ансамбль — середнє прогнозів CatBoost і налаштованої ЛР з вагами 0,5 і 0,5.
# Ваги та поріг правила рішення не підбираються.
#
# Для стратегій ноутбука 05 використовуються їхні збережені зовнішні прогнози;
# контрольні суми перевіряються, ID, суми, цілі та підмножини зіставляються.
#
# Правило вибору: стратегії, Score яких відстає від максимуму не більше ніж на
# 50 USD на 1 000 оцінених спорів, вважаються практично рівними; серед них обирається
# простіша. HistGradientBoosting і CatBoost мають однаковий рівень складності; за
# однакової складності враховуються найгірший період, розкид приросту та log loss.
# Калібровані варіанти з недостатнім обсягом калібрування виключаються з вибору.

# %%
summary, periods, calibration, selection = run(PROJECT_DIR)
columns = ["strategy", "score_usd", "delta_always_usd", "delta_constant_usd",
           "score_weeks_6-9", "score_weeks_10-13", "score_weeks_14-17",
           "log_loss", "brier", "roc_auc", "observed_score_usd",
           "observed_delta_always_usd", "evaluated_rows", "observed_rows", "selectable"]
display(summary[columns].round(5))
print("Внутрішньо вибрані параметри CatBoost:")
display(pd.read_csv(OUTPUT / "outer_choices.csv"))
print("Калібрування CatBoost:")
display(calibration.query("model == 'catboost'"))
display(periods.query("strategy in ['logistic', 'catboost', 'catboost_calibrated', 'lr_cat_50_50']").round(5))

# %%
selected = selection["selected_strategy"]
print("Вибір за незмінним правилом:", selected)
print("Формальний максимум Score:", selection["maximum_score_strategy"])
print("Практичний поріг близькості, USD:", selection["practical_margin_usd"])
print("Близькі стратегії:", selection["near_best_strategies"])
lr_score = summary.loc[summary.strategy.eq("logistic"), "score_usd"].iloc[0]
for name in ["catboost", "catboost_calibrated", "lr_cat_50_50"]:
    row = summary.loc[summary.strategy.eq(name)].iloc[0]
    print(f"{name}: Score={row.score_usd:.2f}; різниця з логістичною={row.score_usd-lr_score:+.2f} USD")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
names = ["logistic", "boosting", "catboost", "catboost_calibrated", "lr_cat_50_50"]
for name in names:
    group = periods.loc[periods.strategy.eq(name)]
    axes[0].plot(group.outer_period, group.delta_constant_usd/group.evaluated_rows, "o-", label=name)
axes[0].set(xlabel="Outer evaluation weeks", ylabel="Uplift vs train-rate policy, USD/dispute",
            title="Same nested temporal scheme")
axes[0].legend(fontsize=7)
view = summary.loc[summary.strategy.isin(names)].sort_values("score_usd")
axes[1].barh(view.strategy, view.score_usd)
axes[1].axvline(view.always_usd.iloc[0], color="gray", linestyle="--", label="Always fight")
axes[1].set(xlabel="Pooled Score, USD", title=f"Same {int(view.evaluated_rows.iloc[0]):,} eligible train disputes")
axes[1].legend()
fig.tight_layout()
path = OUTPUT / "catboost_train_comparison.png"
fig.savefig(path, dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## Інтерпретація та обмеження
#
# Score обчислюється на однакових спорах; невідомі суми для Score не заповнюються.
# Метрики ймовірностей обчислюються на всіх зовнішніх спорах. Сценарій observed
# виключає відновлені або виправлені суми в USD і є перевіркою чутливості.
# Якщо перевага складнішої стратегії не перевищує практичного порогу, обирається
# простіша стратегія.
#
# Експеримент розширює множину моделей після попереднього аналізу train; вкладена
# часова перевірка не усуває повторного використання даних, а три періоди не є
# доказом стійкості. Дата отримання результату спору невідома, тому доступність
# міток у момент прогнозу не підтверджена; параметр has_time=True цього не змінює.
# Validation і test у ноутбуці не відкриваються.
#
# Джерела: [категоріальні ознаки CatBoost](https://catboost.ai/docs/en/features/categorical-features),
# [пропуски](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing),
# [has_time](https://catboost.ai/docs/en/references/training-parameters/common).
