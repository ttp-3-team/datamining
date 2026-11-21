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
# # CatBoost: додаткове порівняння лише всередині train
#
# Мета — перевірити ще одну доречну модель на тих самих зовнішніх часових періодах,
# не використовуючи validation або test. Попередні результати не перезаписуються.
# Це продовження дослідження після перегляду train-результатів, а не нова незалежна
# підтверджувальна вибірка.
#
# Чому ці алгоритми:
#
# - Логістична регресія — проста регуляризована база для прогнозування ймовірності.
#   У нашому варіанті log-сума та закодовані категорії мають адитивний внесок
#   у логарифм шансів; складних взаємодій ми вручну не додавали.
# - HistGradientBoosting перевіряє нелінійності й взаємодії ознак за допомогою дерев.
#   Він був доступний у scikit-learn, коли перше завантаження CatBoost зависло.
# - CatBoost також є бустингом дерев, але має власну обробку категоріальних ознак
#   та Ordered boosting. Це доречний кандидат для наших продавців, провайдерів,
#   причин і валют, але підтримка категорій не гарантує більшого доходу.
#
# Середовище — `uv sync`.
# Для запуску потрібні модулі цього пакета, `outputs/train_audit.csv` і результати
# ноутбука 05 в `train_only_outputs/`. Дані й ID-прогнози залишаються локально.
# Запуск: Restart and Run All.

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
# ## Незмінна схема перевірки
#
# Зовнішні періоди: навчання 0–5 → оцінювання 6–9, 0–9 → 10–13,
# 0–13 → 14–17. Для кожної зовнішньої навчальної частини два внутрішні зрізи
# обирають параметри тільки за її минулими даними. Опора очищення сум також
# визначається заново в кожному зрізі.
#
# Чотири зафіксовані кандидати CatBoost: глибина 4/6 × l2 3/10; 500 дерев,
# learning rate 0.05, seed 42, CPU 4 потоки, Ordered boosting, has_time=True.
# Рядки впорядковані за тижнем; реальний порядок усередині тижня невідомий.
# Немає early stopping за зовнішнім періодом, ручних target-агрегатів або
# балансування класів. Категоріальні пропуски — MISSING, числові — NaN.
# Набір ознак такий самий, як у логістичної регресії.
#
# Параметри обираються за сумарним внутрішнім Score; рівність розв'язують pooled
# log loss і назва кандидата. Sigmoid-калібратор навчається тільки на внутрішніх
# out-of-time прогнозах. Вимоги обсягу калібрування ті самі, що в ноутбуці 05.
# Ансамбль — 50% звичайного CatBoost і 50% звичайної налаштованої логістичної
# регресії. Ваги й поріг бізнес-рішення не підбираються.
#
# Для попередніх стратегій використовуються їхні збережені train-only зовнішні
# прогнози. Контрольні суми перевіряються; ID, суми, цілі та підмножини зіставляються.
# Перенавчати їх за новими зовнішніми результатами не потрібно.
#
# Правило вибору незмінне: варіанти в межах 50 USD на 1 000 оцінених спорів від
# максимуму вважаються практично близькими; перевага простішій стратегії.
# HistGradientBoosting та CatBoost мають однаковий рівень складності. За однакової
# складності враховуються найгірший період, розкид приросту та log loss.
# Непридатні за обсягом калібровані варіанти виключаються з вибору.

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
# ## Як читати висновок
#
# Дохід оцінюється на однакових спорах; невідомі суми не заповнюються для Score.
# Метрики ймовірностей включають усі зовнішні спори. Observed-сценарій виключає
# відновлені/виправлені USD-суми й залишається перевіркою чутливості.
# Якщо складніша стратегія не перевищує практичний поріг, її невелика перевага
# не змінює вибір простішого рішення.
#
# Цей експеримент розширює простір моделей після попереднього аналізу train.
# Вкладена схема не скасовує повторного використання даних; три періоди не
# доводять стійкості. Немає дати отримання результату спору, тому доступність
# навчальних міток у реальному часі не підтверджена. has_time=True цього не виправляє.
# Validation і test тут не відкриваються. Параметри й правило мають бути зафіксовані
# до майбутнього фінального тесту. Оптимальність серед усіх можливих моделей не заявляється.
#
# Джерела: [категоріальні ознаки CatBoost](https://catboost.ai/docs/en/features/categorical-features),
# [пропуски](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing),
# [has_time](https://catboost.ai/docs/en/references/training-parameters/common).
