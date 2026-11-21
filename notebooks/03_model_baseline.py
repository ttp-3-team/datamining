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
# # Перша модель: ймовірність виграшу та рішення боротися
#
# Використовуємо готові результати очищення. Навчання — тижні 0–17,
# перевірка — 18–21. Test і unknown_time тут не читаються.
# Це початкова модель для порівняння, а не остаточний вибір.
#
# Запуск: `uv sync`, потім Restart and Run All (потрібен `outputs/` з ноутбука 02).
# Результати записуються окремо в `model_outputs`; очищені таблиці не змінюються.

# %%
import json
import sys

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disputes.economics import check_aligned, payoff, policy, profit_rows
from disputes.evaluation import calibration_bins, classification_metrics
from disputes.models import feature_columns, features, fit, predict
from disputes.paths import PROJECT as PROJECT_DIR
from disputes.preparation import calculate_baseline, load_table

OUTPUT = PROJECT_DIR / "model_outputs"
OUTPUT.mkdir(exist_ok=True)
print("scikit-learn:", sklearn.__version__)

# %% [markdown]
# ## 1. Дані й ознаки
#
# `won` — ціль, `dispute_id` — ключ. Аудит потрібний лише для оцінювання сум.
# Беремо категорії продавця, провайдера, способу оплати, бренду, причини та валюти.
# День тижня — категорія, а не числова відстань. Невідомі категорії мають значення
# `MISSING`; нові категорії на validation кодуються без помилки.
#
# Сума: `log1p(amount_usd)`, медіана train, прапорець пропуску й стандартизація.
# Після логарифма медіана обчислюється в перетвореному просторі. Це лише вхід моделі:
# для доходу використовуємо вихідну очищену USD-суму.
# `amount` поки виключено через різні валюти. Номер тижня виключено з першого
# варіанта, щоб не закладати лінійну екстраполяцію часу; його користь можна перевірити
# окремим експериментом. Правила очищення й доступність ознак до рішення залишаються припущеннями.

# %%
train = load_table(PROJECT_DIR / "outputs", "train_clean.csv")
train_audit = load_table(PROJECT_DIR / "outputs", "train_audit.csv")
validation = load_table(PROJECT_DIR / "outputs", "validation_clean.csv")
validation_audit = load_table(PROJECT_DIR / "outputs", "validation_audit.csv")
assert train.dispute_week.max() < validation.dispute_week.min()
assert set(train.dispute_id).isdisjoint(validation.dispute_id)
assert set(train.won.unique()) == {0, 1}

SPEC = {"name": "logistic", "kind": "logistic", "C": 1.0}
CATEGORICAL, NUMERIC = feature_columns(SPEC)
X_train, _ = features(train, train_audit, SPEC)
assert not {"won", "dispute_id", "amount"}.intersection(X_train)
display(pd.DataFrame({"rows": [len(train), len(validation)],
                      "win_rate": [train.won.mean(), validation.won.mean()]},
                     index=["train", "validation"]))

# %% [markdown]
# ## 2. Навчання
#
# Логістична регресія з регуляризацією, `C=1`. Параметри за validation не підбираємо.
# Немає балансування класів: потрібні ймовірності для природної частоти виграшів.
# Медіана, масштаб і словник one-hot навчаються тільки на train в одному Pipeline.
# Постійна ймовірність, рівна частці виграшів train, — контроль якості прогнозу.

# %%
model = fit(train, train_audit, SPEC)
p = predict(model, validation, validation_audit, SPEC)
p_constant = np.full(len(validation), train.won.mean())
print("Ітерацій навчання:", model.named_steps["classifier"].n_iter_.tolist())
print("Ознак після перетворення:", model.named_steps["preprocess"].transform(X_train).shape[1])

# %% [markdown]
# ## 3. Якість ймовірностей
#
# ROC AUC оцінює ранжування, log loss і Brier — якість імовірностей (менше — краще).
# Accuracy і матриця помилок при 0.5 — додаткові показники, а не критерій доходу.
# Average precision показано для `won=1`, який тут є більшістю.
# Калібрування перевіряємо за групами прогнозів; на validation його не донавчаємо.

# %%
metrics = pd.DataFrame({name: classification_metrics(validation.won, probability)
                        for name, probability in {"constant_train_rate": p_constant,
                                                  "logistic_regression": p}.items()}).T
display(metrics[["roc_auc", "average_precision_won", "log_loss", "brier", "accuracy"]].astype(float).round(4))
counts = metrics.loc["logistic_regression"]
display(pd.DataFrame([[counts.true_lost, counts.false_won], [counts.false_lost, counts.true_won]],
                     index=["actual_lost", "actual_won"], columns=["predicted_lost", "predicted_won"]))
calibration = calibration_bins(validation.won, p)
display(calibration)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.plot(calibration.mean_probability, calibration.win_rate, "o-")
ax.set(xlabel="Mean predicted win probability", ylabel="Observed win rate",
       title="Validation calibration", xlim=(0, 1), ylim=(0, 1))
fig.tight_layout()
fig.savefig(OUTPUT / "validation_calibration.png", dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## 4. Дохід від рішення
#
# Боремося, якщо `p × 0.2 × amount_usd > 6`. Це поріг очікуваного доходу,
# виведений із умови задачі, а не підібраний за validation. Фактичний Score:
# сума `0.2 × amount_usd × won − 6` для вибраних спорів.
#
# Порівнюємо чотири стратегії на однакових рядках: ніколи, завжди,
# рішення за постійною train-ймовірністю та рішення за моделлю.
# Два сценарії: усі придатні суми й суми без відновлення/корекції масштабу.
# Між сценаріями склад рядків різний; їхні суми не вимірюють ефект виправлень.
# За відсутньої суми дія — `review_amount`; це не автоматична відмова і не нульовий дохід.

# %%
def economic_table(data, audit, p, p_constant):
    """Чотири стратегії на однакових рядках для обох сценаріїв сум."""
    check_aligned(data, audit)
    decisions = {"never_fight": np.zeros(len(data), dtype=bool), "always_fight": np.ones(len(data), dtype=bool),
                 "constant_probability": policy(data, audit, p_constant) == "fight",
                 "logistic_regression": policy(data, audit, p) == "fight"}
    result = []
    for scenario, flag in [("eligible", "score_eligible"), ("observed_only", "score_observed_only")]:
        mask = audit[flag].to_numpy(dtype=bool)
        profit = payoff(data)[mask]
        for strategy, fight in decisions.items():
            score = float(profit[fight[mask]].sum())
            result.append({"scenario": scenario, "strategy": strategy, "evaluated_rows": int(mask.sum()),
                           "total_rows": len(data), "coverage": float(mask.mean()),
                           "fight_rows": int(fight[mask].sum()), "score_usd": score,
                           "delta_vs_always_usd": score - float(profit.sum())})
    return pd.DataFrame(result)

economics = economic_table(validation, validation_audit, p, p_constant)
for scenario, observed in [("eligible", False), ("observed_only", True)]:
    baseline = calculate_baseline(validation, validation_audit, observed_only=observed)
    row = economics.loc[economics.scenario.eq(scenario) & economics.strategy.eq("always_fight")].iloc[0]
    assert np.isclose(row.score_usd, baseline["score_usd"])
    assert row.evaluated_rows == baseline["rows"]
display(economics.round(3))

# %% [markdown]
# ### Контрольний приклад грошового розрахунку
#
# Перевіряємо вибір великої суми з високою ймовірністю, відмову за низької
# ймовірності або малої суми, виключення невідомої суми й відповідність ID.

# %%
example = pd.DataFrame({"dispute_id": ["a", "b", "c", "d"],
                        "amount_usd": [100., 100., 20., np.nan], "won": [1, 1, 1, 0]})
example_audit = pd.DataFrame({"dispute_id": example.dispute_id,
                              "score_eligible": [True, True, True, False],
                              "score_observed_only": [True, False, True, False]})
example_constant = np.full(4, train.won.mean())
check = economic_table(example, example_audit, np.array([.9, .1, .99, .8]), example_constant)
chosen = check.query("scenario == 'eligible' and strategy == 'logistic_regression'").iloc[0]
assert chosen.score_usd == 14 and chosen.fight_rows == 1 and chosen.evaluated_rows == 3
assert check.query("scenario == 'eligible' and strategy == 'always_fight'").score_usd.iloc[0] == 26
assert check.query("strategy == 'never_fight'").score_usd.eq(0).all()
assert check.query("scenario == 'observed_only'").evaluated_rows.eq(2).all()
try:
    economic_table(example, example_audit.iloc[::-1].reset_index(drop=True), np.ones(4), example_constant)
except ValueError:
    pass
else:
    raise AssertionError("Переставлений аудит має бути відхилено")
print("Контроль формули, підмножин та відповідності ID: OK")

# %% [markdown]
# ## 5. Збереження результатів
#
# Зберігаємо ймовірності, дії, метрики, таблицю доходу й конфігурацію.
# Для `review_amount` фактичний дохід лишається невідомим, не нулем.
# Ймовірності є для всіх validation-спорів, зокрема з пропущеною сумою.
# Модель можна відтворити повторним запуском; pickle-файл не потрібний.

# %%
eligible = validation_audit.score_eligible.to_numpy(dtype=bool)
predictions = validation[["dispute_id", "won", "amount_usd"]].assign(
    p_won=p, score_eligible=eligible, score_observed_only=validation_audit.score_observed_only.to_numpy(),
    action=policy(validation, validation_audit, p), realized_profit_usd=profit_rows(validation, validation_audit, p))
assert predictions.loc[~eligible, "realized_profit_usd"].isna().all()
assert np.isclose(predictions.realized_profit_usd.sum(),
                  economics.query("scenario == 'eligible' and strategy == 'logistic_regression'").score_usd.iloc[0])
predictions.to_csv(OUTPUT / "validation_predictions.csv", index=False)
metrics.to_csv(OUTPUT / "validation_probability_metrics.csv", index_label="model")
economics.to_csv(OUTPUT / "validation_economics.csv", index=False)
calibration.to_csv(OUTPUT / "validation_calibration.csv", index=False)
source_manifest = json.loads((PROJECT_DIR / "outputs" / "manifest.json").read_text(encoding="utf-8"))
config = {
    "model": "LogisticRegression", "spec": SPEC, "seed": 42,
    "categorical": CATEGORICAL, "numeric": NUMERIC,
    "amount_transform": "log1p, train median with missing indicator, StandardScaler",
    "decision": "p * 0.2 * amount_usd > 6; missing amount -> review_amount",
    "test_evaluated": False, "validation_tuned": False,
    "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    "input_sha256": {name: source_manifest["files"][name]["sha256"] for name in
                     ["train_clean.csv", "train_audit.csv", "validation_clean.csv", "validation_audit.csv"]},
}
(OUTPUT / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
selected = economics.query("scenario == 'eligible' and strategy == 'logistic_regression'").iloc[0]
constant = economics.query("scenario == 'eligible' and strategy == 'constant_probability'").iloc[0]
summary = (f"Validation: {len(validation)} disputes; monetary evaluation: {int(selected.evaluated_rows)}.\n"
           f"Logistic policy Score: {selected.score_usd:.2f} USD.\n"
           f"Delta vs always fight: {selected.delta_vs_always_usd:+.2f} USD.\n"
           f"Delta vs constant-probability policy: {selected.score_usd - constant.score_usd:+.2f} USD.\n"
           f"Amount review required: {int((~eligible).sum())} disputes.\n"
           "Test has not been evaluated. No validation tuning was performed.\n")
print(summary)
(OUTPUT / "summary.txt").write_text(summary, encoding="utf-8")
print("Результати:", OUTPUT)

# %% [markdown]
# ## Висновки та наступний експеримент
#
# Дивись фактичні числа в попередній комірці. Перевага над «завжди» може бути
# наслідком відмови від малих сум, тому важливо також порівняння з постійною
# ймовірністю. Одна validation-вибірка не доводить стійкість покращення.
#
# Далі: порівняти сильнішу модель на тих самих рядках, перевірити додавання
# `amount` разом із валютою та прапорців якості. Після вибору зафіксувати процес
# і один раз оцінити test. Поточний ноутбук не виконує цих наступних експериментів.
#
# Обмеження: правила відновлення сум є припущеннями; Score охоплює лише частину
# спорів; test переглядали при первинному дослідженні; немає дати отримання результату;
# вибірка містить лише історично оскаржені спори з відомим результатом.
#
# Методи: [Pipeline і ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html),
# [логістична регресія](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
# [оцінка калібрування](https://scikit-learn.org/stable/modules/calibration.html).
