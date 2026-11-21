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
# # Навчання з вагами за сумою та щотижнева стійкість моделей
#
# Нові розрахунки виконуються лише на train. Зведення результатів на validation
# будується із збережених агрегованих значень; нові прогнози на validation не
# обчислюються. Test не відкривається.
#
# Повний перерахунок: `uv run python -m disputes.weighted_weekly_study`, далі
# `uv run python scripts/build_method_review.py`. За замовчуванням ноутбук читає
# збережені результати після перевірки контрольних сум; для повторного навчання
# слід встановити `RECOMPUTE=True`.
#
# Усі методи та відносні оцінки зведено в `method_review_outputs/method_review.html`
# і `all_methods_summary.csv`. Режими оцінювання (вкладена часова перевірка на train,
# щотижнева перевірка, validation) мають різні множини спорів, тому абсолютні значення
# Score різних режимів не порівнюються між собою.

# %%
import json

import matplotlib

from disputes.paths import PROJECT as PROJECT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

from disputes.protocol import verified_csv
from disputes.weighted_weekly_study import freeze, run

RECOMPUTE=False
OUTPUT,protocol_hash=freeze(PROJECT_DIR)
if RECOMPUTE:
    run(PROJECT_DIR)
manifest=json.loads((OUTPUT/"run_evidence.json").read_text(encoding="utf-8"))
assert manifest["protocol_sha256"]==protocol_hash
def verified_table(name):
    return verified_csv(OUTPUT / f"{name}.csv", manifest["result_sha256"][name])
primary=verified_table("primary_summary")
weekly=verified_table("weekly_summary")
periods=verified_table("weekly_periods")
print("Зафіксований протокол:",protocol_hash)
print("Прочитано перевірені результати; RECOMPUTE =",RECOMPUTE)

# %% [markdown]
# ## 1. Навчання з вагами за сумою
#
# Порівнюються логістична регресія (ЛР) без ваг, з вагою, пропорційною сумі, і з
# вагою, пропорційною квадратному кореню із суми. Ваги обмежено 99-м перцентилем,
# обчисленим на навчальній частині зрізу, і нормовано до середнього значення 1.
# Спорам із невідомою сумою призначається вага, що відповідає медіанній відомій сумі
# train; ця вага використовується лише під час навчання й не замінює суму в Score.
# Ваги не залежать від won і передаються лише класифікатору; попередня обробка
# ознак навчається без ваг.
#
# Для кожної схеми ваг C обирається окремо на внутрішніх зрізах кожного зовнішнього
# періоду. Додатково перевіряється sigmoid-калібрування на внутрішніх out-of-time
# прогнозах; вимоги до обсягу калібрування ті самі, що в ноутбуці 05. Score
# обчислюється без ваг на тих самих спорах, що й для інших стратегій.

# %%
cols=["strategy","score_usd","delta_vs_logistic_usd","uplift_vs_always_percent",
      "uplift_vs_constant_percent","uplift_vs_logistic_percent","log_loss","brier","roc_auc",
      "observed_score_usd","evaluated_rows","observed_rows"]
display(primary[cols].round(5))
display(verified_table("primary_calibration"))
print("Вибір за попереднім практичним правилом:")
print((OUTPUT/"primary_selection.json").read_text())

# %% [markdown]
# ## 2. Щотижнева стійкість
#
# Кожен тиждень 10–17 прогнозується окремо моделями, навченими на попередніх тижнях.
# Для останнього навчального тижня t внутрішні зрізи такі: навчання до тижня t−6 →
# оцінювання t−5…t−3; навчання до t−3 → оцінювання t−2…t. Для кожного тижня
# заново обираються C, конфігурації HGB і CatBoost, параметри очищення, словники
# категорій, шкали, ваги та калібратори. Ансамблі мають фіксовані ваги 0,5 і 0,5;
# варіанти взаємодій ті самі, що в ноутбуці 08. ЛР з фіксованим C = 0,1 є
# діагностичним контролем і не бере участі у виборі.
#
# Усі 17 стратегій оцінюються на тих самих спорах кожного тижня. Наводяться сумарний
# Score, різниця на 1 000 спорів, кількість тижнів із перевагою, найгірший тиждень і
# розкид. Нормування на 1 000 спорів враховує різний обсяг тижнів, але не різний
# склад спорів.
#
# Тижні 10–17 уже використовувалися в попередніх дослідженнях; навчальні вікна
# перекриваються. Дата отримання результату спору невідома, тому припущення про
# доступність усіх попередніх міток на наступному тижні не підтверджене.

# %%
display(weekly[cols+["periods_above_always","periods_above_logistic",
                     "worst_delta_vs_logistic_per1000","sd_delta_vs_logistic_per1000"]].round(5))
display(periods.pivot(index="strategy",columns="outer_period",values="delta_vs_logistic_usd").round(2))
print("Щотижневий вибір за практичним правилом:")
print((OUTPUT/"weekly_selection.json").read_text())

fig,axes=plt.subplots(1,2,figsize=(14,6))
view=weekly.loc[~weekly.strategy.eq("constant_train_rate")].sort_values("delta_vs_logistic_usd")
axes[0].barh(view.strategy,view.delta_vs_logistic_usd)
axes[0].axvline(0,color="gray",linestyle="--")
axes[0].set(xlabel="Weekly pooled difference vs logistic, USD",title="Same weekly evaluation disputes")
for name in ["lr_weight_amount","lr_weight_amount_calibrated","lr_weight_sqrt","lr_weight_sqrt_calibrated","lr_cat_50_50"]:
    group=periods.loc[periods.strategy.eq(name)]
    axes[1].plot(group.outer_period,group.delta_vs_logistic_per1000,"o-",label=name)
axes[1].axhline(0,color="gray",linestyle="--")
axes[1].set(xlabel="Evaluation week",ylabel="Difference vs logistic, USD/1000",title="Week-to-week stability")
axes[1].legend(fontsize=7)
fig.tight_layout()
figure=OUTPUT/"weighted_weekly_comparison.png"
fig.savefig(figure,dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## Зведена таблиця методів
#
# Звіт `method_review_outputs/method_review.html` містить загальний огляд, окремі
# таблиці трьох режимів оцінювання, сітку C і попередні експерименти з вибором ознак
# для HGB. Порожнє значення означає, що метод у цьому режимі не оцінювався. Нові
# методи з вагами та взаємодіями на validation не оцінювалися.
#
# Відносні оцінки: (Score − Score базової стратегії) / Score базової стратегії × 100 %;
# індекс Score, де ЛР = 100; Score на 1 000 оцінених спорів. Кожен режим має власну
# базову стратегію та кількість рядків. Метрики ймовірностей обчислюються на всіх
# рядках із міткою, Score — на спорах із сумою, придатною для Score. Сценарій
# observed виключає відновлені або виправлені суми.
#
# Порядок методів за Score може відрізнятися від порядку за log loss або Brier.
# Практичний поріг рівності — 50 USD на 1 000 спорів; серед практично рівних
# обирається простіший варіант. Результати слід інтерпретувати з урахуванням
# повторного використання train і validation, невідомої дати доступності міток і
# відбору спорів у даних.
#
# Параметр sample_weight: [документація LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
