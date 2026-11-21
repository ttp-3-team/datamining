# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Друга оцінка: покращення, обрані на тижнях 0–21, і один замір test
#
# Перша оцінка (логістична регресія, навчена на тижнях 0–17) дала на test +$1 740 до «боротися завжди».
# Вона лишається незмінною. Тут — наперед зафіксований другий експеримент:
#
# 1. **Розробка лише на train+validation (тижні 0–21).** Вкладена часова перевірка на чотирьох зовнішніх
#    періодах (6–9, 10–13, 14–17, 18–21) п'яти рецептів: логістична регресія; + вага свіжих тижнів;
#    + історія мерчанта (згладжена частка виграшів лише з попередніх тижнів); + обидва; CatBoost.
# 2. **Вибір** — те саме правило: серед рецептів у межах $50 на 1 000 спорів від максимуму — простіший.
# 3. **Фіксація** рецепта, коду й залежностей, потім **один замір test** для кожного рецепта, кожен
#    перенавчений на тижнях 0–21. Головний результат — заздалегідь обраний рецепт, яким би він не був.
#
# Обмеження: агрегати test першої оцінки вже були відомі до цього протоколу; це зазначено у звіті.
# Запуск: `uv run python -m disputes.second_evaluation develop`, потім `freeze` і `test`.

# %%
import json

import matplotlib
import pandas as pd
from IPython.display import display

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disputes.paths import PROJECT
from disputes.second_evaluation import OUTPUT, TEST_OUTPUT

development, test_output = PROJECT / OUTPUT, PROJECT / TEST_OUTPUT
summary = pd.read_csv(development / "strategy_summary.csv")
selection = json.loads((development / "selection.json").read_text(encoding="utf-8"))
report = json.loads((test_output / "test_report.json").read_text(encoding="utf-8"))

# %% [markdown]
# ## 1. Розробка на тижнях 0–21 (test закритий)

# %%
columns = ["strategy", "score_usd", "delta_always_usd", "delta_constant_usd", "log_loss", "roc_auc",
           "worst_period_delta_constant_per_row", "complexity"]
display(summary[columns].round(4))
print("Обраний рецепт:", selection["selected_strategy"], "| максимум Score:", selection["maximum_score_strategy"],
      "| допуск, USD:", round(selection["practical_margin_usd"], 2), "| близькі:", selection["near_best_strategies"])
display(pd.read_csv(development / "cost_sensitivity.csv").query("strategy == @selection['selected_strategy']").round(2))

# %% [markdown]
# ## 2. Один замір test для кожного рецепта
#
# Усі рецепти показано, а не лише найкращий: `delta_vs_first_evaluation_usd` — різниця з моделлю першої
# оцінки на тих самих спорах. Інтервали — 95% bootstrap за спорами та за мерчантами.

# %%
comparison = pd.DataFrame(report["comparison"])
view = comparison[["strategy", "headline", "score_usd", "delta_vs_always_usd", "delta_vs_always_ci_disputes",
                   "delta_vs_always_ci_merchants", "delta_vs_first_evaluation_usd", "delta_vs_first_ci_disputes",
                   "roc_auc", "log_loss", "brier"]]
display(view.round(4))

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
ordered = comparison.sort_values("delta_vs_always_usd")
low = ordered.delta_vs_always_ci_disputes.str[0]
high = ordered.delta_vs_always_ci_disputes.str[1]
ax.barh(ordered.strategy, ordered.delta_vs_always_usd,
        xerr=[ordered.delta_vs_always_usd - low, high - ordered.delta_vs_always_usd],
        color=["#c8641e" if h else "#2b59c3" for h in ordered.headline])
ax.axvline(0, color="gray", linewidth=1)
ax.set(xlabel="Score − «боротися завжди», USD (95% bootstrap)", title="Test, тижні 22–25: усі рецепти")
fig.tight_layout()
fig.savefig(test_output / "second_evaluation.png", dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## 3. Звідки приріст і наскільки він стійкий

# %%
segments = pd.read_csv(test_output / "headline_segments.csv")
display(segments[["field", "segment", "rows", "eligible_rows", "win_rate", "fight_rate", "score_usd",
                  "always_usd", "delta_vs_always_usd"]].round(3))
display(pd.DataFrame(report["cost_sensitivity"]).round(2))

# %% [markdown]
# ## 4. Якість даних: обсяг за тижнями та категорії причин
#
# Спори без фінального результату не потрапили в датасет, тому останні тижні можуть бути неповними.

# %%
weekly = pd.read_csv(test_output / "weekly_volume.csv").dropna(subset=["dispute_week"])
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(weekly.dispute_week, weekly.rows, color="#9aa3b2")
ax.set(xlabel="Тиждень", ylabel="Спорів")
twin = ax.twinx()
twin.plot(weekly.dispute_week, weekly.win_rate, "o-", color="#2b59c3")
twin.set_ylabel("Частка won")
fig.tight_layout()
display(fig)
plt.close(fig)
display(pd.read_csv(test_output / "reason_categories.csv"))
