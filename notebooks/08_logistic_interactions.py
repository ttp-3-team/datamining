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
# # Взаємодії ознак у логістичній регресії (train)
#
# Перевіряється гіпотеза, що комбінації провайдера з валютою або з причиною спору
# містять інформацію понад окремі внески цих категорій. Validation і test не
# відкриваються; попередні результати не перезаписуються.
#
# Варіанти з однаковими базовими ознаками:
#
# 1. `baseline` — без взаємодій;
# 2. `psp_currency` — додано взаємодію провайдер × валюта;
# 3. `psp_reason` — додано взаємодію провайдер × категорія причини;
# 4. `both_interactions` — додано обидві взаємодії.
#
# Логістична регресія (ЛР) з фіксованим C = 0,1 наведена як діагностичний контроль.
# Це значення C обрано раніше на тих самих періодах train, тому воно не бере участі
# у виборі.
#
# Вхідні дані: `outputs/train_audit.csv` і результати ноутбука 05 у
# `train_only_outputs/`. Результати записуються в `interaction_outputs/`.

# %%
import matplotlib

from disputes.paths import PROJECT as PROJECT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

from disputes.interaction_study import VARIANTS, freeze, run

OUTPUT, protocol_hash = freeze(PROJECT_DIR)
print("Протокол зафіксовано до нових результатів:", protocol_hash)
print("Варіанти:", VARIANTS)

# %% [markdown]
# ## Кодування та схема порівняння
#
# Взаємодія є додатковою категоріальною ознакою, наприклад [провайдер, USD]; базові
# категорії залишаються в моделі. Пари кодуються як JSON-список, щоб роздільники в
# назвах категорій не об'єднували різні комбінації. Словник one-hot навчається лише
# на навчальній частині кожного зрізу. Комбінація, якої немає в навчальній частині,
# отримує нульовий блок one-hot для взаємодії; базові ознаки при цьому враховуються.
#
# Для кожного варіанта C ∈ {0,01; 0,1; 1; 10; 100} обирається окремо на внутрішніх
# часових зрізах: за найбільшим pooled внутрішнім Score, далі за меншим pooled log loss
# і меншим C. Калібрування, ваги спостережень, підбір порогу та ансамблі не
# застосовуються.
#
# Зовнішні періоди: 0–5 → 6–9, 0–9 → 10–13, 0–13 → 14–17; внутрішні зрізи ті самі,
# що в ноутбуці 05. Опорні значення очищення сум визначаються на навчальній частині
# кожного зрізу.
#
# Правило вибору: варіанти, Score яких відстає від максимуму не більше ніж на 50 USD
# на 1 000 оцінених спорів, вважаються практично рівними; серед них обирається варіант
# з меншою кількістю взаємодій. За однакової складності враховуються найгірший період
# за приростом відносно постійної стратегії на один спір, розкид і log loss. Правило
# є евристичним і не є статистичним тестом.

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
# ## Інтерпретація
#
# Основне порівняння проводиться з `baseline`, для якого C обирається лише за даними,
# що передують відповідному зовнішньому періоду. Прогнози `baseline` звіряються з
# результатами ноутбука 05. ЛР з фіксованим C = 0,1 є контролем і не замінює
# вкладеного `baseline`, оскільки це значення обрано на тих самих даних.
#
# Метрики ймовірностей обчислюються на всіх зовнішніх спорах, Score — на однакових
# спорах із сумою, придатною для Score. Сценарій observed виключає відновлені або
# виправлені суми в USD і є перевіркою чутливості, а не критерієм вибору.
# Коефіцієнти описують внески в логарифм шансів і не є оцінками причинного впливу
# провайдера, валюти чи причини спору.

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
display(fig)
plt.close(fig)

# %% [markdown]
# ## Обмеження та відтворення
#
# Ці періоди train аналізувалися раніше, зокрема під час формулювання гіпотез.
# Вкладена часова перевірка контролює підбір C, але не усуває повторного використання
# даних; три зовнішні періоди не є доказом стійкості. Дата отримання результату спору
# невідома, тому доступність міток у момент прогнозу не підтверджена. Validation і
# test у ноутбуці не відкриваються.
#
# Файли результатів: `protocol.json` (протокол), `strategy_summary.csv` (зведена
# таблиця), `outer_choices.csv` (обрані C і розмір моделей), `interaction_coverage.csv`
# (нові комбінації), `coefficients.csv` (коефіцієнти), `run_evidence.json` (контрольні
# суми та журнал доступу до вхідних CSV). Виводи та прогнози з ID не публікуються.
#
# Тести: `uv run pytest tests/test_interaction_study.py`. Виконання ноутбука:
# `uv run python scripts/check_notebooks.py 08_logistic_interactions`.
