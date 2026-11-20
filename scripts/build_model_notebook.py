"""Зібрати читабельний ноутбук першої моделі (без запуску навчання)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# Перша модель: ймовірність виграшу та рішення боротися

Використовуємо готові результати очищення. Навчання — тижні 0–17,
перевірка — 18–21. Test і unknown_time тут не читаються.
Це початкова модель для порівняння, а не остаточний вибір.

Запуск: встановити `requirements_model.txt`, покласти ноутбук поряд із
`preparation.py` та папкою `outputs`, виконати Restart and Run All.
У Colab за потреби змінити тільки `PROJECT_DIR` нижче.
Результати записуються окремо в `model_outputs`; очищені таблиці не змінюються.''')
code('''from pathlib import Path
import sys
PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / "preparation.py").exists():
    raise FileNotFoundError("Задайте PROJECT_DIR: папку з preparation.py та outputs")
sys.path.insert(0, str(PROJECT_DIR))

import json
import hashlib
import warnings
import base64
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             brier_score_loss, accuracy_score, confusion_matrix)
from preparation import load_table, calculate_baseline

OUTPUT = PROJECT_DIR / "model_outputs"
OUTPUT.mkdir(exist_ok=True)
print("scikit-learn:", sklearn.__version__)
''')
md('''## 1. Дані й ознаки

`won` — ціль, `dispute_id` — ключ. Аудит потрібний лише для оцінювання сум.
Беремо категорії продавця, провайдера, способу оплати, бренду, причини та валюти.
День тижня — категорія, а не числова відстань. Невідомі категорії мають значення
`MISSING`; нові категорії на validation кодуються без помилки.

Сума: `log1p(amount_usd)`, медіана train, прапорець пропуску й стандартизація.
Після логарифма медіана обчислюється в перетвореному просторі. Це лише вхід моделі:
для доходу використовуємо вихідну очищену USD-суму.
`amount` поки виключено через різні валюти. Номер тижня виключено з першого
варіанта, щоб не закладати лінійну екстраполяцію часу; його користь можна перевірити
окремим експериментом. Правила очищення й доступність ознак до рішення залишаються припущеннями.''')
code('''train = load_table(PROJECT_DIR / "outputs", "train_clean.csv")
validation = load_table(PROJECT_DIR / "outputs", "validation_clean.csv")
validation_audit = load_table(PROJECT_DIR / "outputs", "validation_audit.csv")
assert train.dispute_week.max() < validation.dispute_week.min()
assert set(train.dispute_id).isdisjoint(validation.dispute_id)
assert validation.dispute_id.equals(validation_audit.dispute_id)
assert set(train.won.unique()) == {0, 1}

CATEGORICAL = ["merchant_id", "psp", "payment_method", "card_brand",
               "reason_category", "reason_code", "currency", "dispute_weekday"]
NUMERIC = ["amount_usd"]
FEATURES = CATEGORICAL + NUMERIC
assert not {"won", "dispute_id", "amount"}.intersection(FEATURES)

def model_input(data):
    result = data.loc[:, FEATURES].copy()
    for col in CATEGORICAL:
        result[col] = result[col].astype("string").fillna("MISSING").astype(str)
    result["amount_usd"] = result.amount_usd.astype(float)
    return result

X_train, X_validation = model_input(train), model_input(validation)
y_train, y_validation = train.won.astype(int), validation.won.astype(int)
display(pd.DataFrame({"rows": [len(train), len(validation)],
                      "win_rate": [y_train.mean(), y_validation.mean()]},
                     index=["train", "validation"]))
''')
md('''## 2. Навчання

Логістична регресія з регуляризацією, `C=1`. Параметри за validation не підбираємо.
Немає балансування класів: потрібні ймовірності для природної частоти виграшів.
Медіана, масштаб і словник one-hot навчаються тільки на train в одному Pipeline.
Постійна ймовірність, рівна частці виграшів train, — контроль якості прогнозу.''')
code('''numeric_pipeline = Pipeline([
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])
preprocess = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("amount", numeric_pipeline, NUMERIC),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
])
with warnings.catch_warnings():
    warnings.simplefilter("error", ConvergenceWarning)
    model.fit(X_train, y_train)
p = model.predict_proba(X_validation)[:, list(model.classes_).index(1)]
assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
p_constant = np.full(len(validation), y_train.mean())
print("Ітерацій навчання:", model.named_steps["classifier"].n_iter_.tolist())
print("Ознак після перетворення:", model.named_steps["preprocess"].transform(X_train).shape[1])
''')
md('''## 3. Якість ймовірностей

ROC AUC оцінює ранжування, log loss і Brier — якість імовірностей (менше — краще).
Accuracy і матриця помилок при 0.5 — додаткові показники, а не критерій доходу.
Average precision показано для `won=1`, який тут є більшістю.
Калібрування перевіряємо за групами прогнозів; на validation його не донавчаємо.''')
code('''def probability_metrics(probability):
    return {"roc_auc": roc_auc_score(y_validation, probability),
            "average_precision_won": average_precision_score(y_validation, probability),
            "log_loss": log_loss(y_validation, probability, labels=[0, 1]),
            "brier": brier_score_loss(y_validation, probability),
            "accuracy_at_0.5": accuracy_score(y_validation, probability >= .5)}

metrics = pd.DataFrame({"constant_train_rate": probability_metrics(p_constant),
                        "logistic_regression": probability_metrics(p)}).T
display(metrics.round(4))
display(pd.DataFrame(confusion_matrix(y_validation, p >= .5, labels=[0, 1]),
                     index=["actual_lost", "actual_won"],
                     columns=["predicted_lost", "predicted_won"]))
calibration_rows = pd.DataFrame({"p": p, "won": y_validation.to_numpy()})
calibration_rows["bin"] = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
calibration = calibration_rows.groupby("bin", observed=True).agg(
    rows=("won", "size"), predicted=("p", "mean"), observed=("won", "mean"))
display(calibration)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.plot(calibration.predicted, calibration.observed, "o-")
ax.set(xlabel="Mean predicted win probability", ylabel="Observed win rate",
       title="Validation calibration", xlim=(0, 1), ylim=(0, 1))
fig.tight_layout()
plot_path = OUTPUT / "validation_calibration.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
display(HTML('<img alt="Validation calibration" width="600" src="data:image/png;base64,'
             + base64.b64encode(plot_path.read_bytes()).decode() + '">'))
''')
md('''## 4. Дохід від рішення

Боремося, якщо `p × 0.2 × amount_usd > 6`. Це поріг очікуваного доходу,
виведений із умови задачі, а не підібраний за validation. Фактичний Score:
сума `0.2 × amount_usd × won − 6` для вибраних спорів.

Порівнюємо чотири стратегії на однакових рядках: ніколи, завжди,
рішення за постійною train-ймовірністю та рішення за моделлю.
Два сценарії: усі придатні суми й суми без відновлення/корекції масштабу.
Між сценаріями склад рядків різний; їхні суми не вимірюють ефект виправлень.
За відсутньої суми дія — `review_amount`; це не автоматична відмова і не нульовий дохід.''')
code('''def economic_table(data, audit, probabilities):
    if not data.dispute_id.equals(audit.dispute_id):
        raise ValueError("Дані й аудит не вирівняні")
    result = []
    for scenario, flag in [("eligible", "score_eligible"),
                           ("observed_only", "score_observed_only")]:
        mask = audit[flag].to_numpy(dtype=bool)
        amount = data.loc[mask, "amount_usd"].to_numpy(dtype=float)
        won = data.loc[mask, "won"].to_numpy(dtype=int)
        assert np.isfinite(amount).all() and (amount > 0).all()
        profit = .2 * amount * won - 6
        decisions = {
            "never_fight": np.zeros(len(amount), dtype=bool),
            "always_fight": np.ones(len(amount), dtype=bool),
            "constant_probability": .2 * amount * y_train.mean() > 6,
            "logistic_regression": .2 * amount * probabilities[mask] > 6,
        }
        for strategy, fight in decisions.items():
            score = float(profit[fight].sum())
            result.append({"scenario": scenario, "strategy": strategy,
                           "evaluated_rows": len(amount), "total_rows": len(data),
                           "coverage": float(mask.mean()), "fight_rows": int(fight.sum()),
                           "score_usd": score,
                           "delta_vs_always_usd": score - float(profit.sum())})
    return pd.DataFrame(result)

economics = economic_table(validation, validation_audit, p)
for scenario, observed in [("eligible", False), ("observed_only", True)]:
    baseline = calculate_baseline(validation, validation_audit, observed_only=observed)
    row = economics.query("scenario == @scenario and strategy == 'always_fight'").iloc[0]
    assert np.isclose(row.score_usd, baseline["score_usd"])
    assert row.evaluated_rows == baseline["rows"]
display(economics.round(3))
''')
md('''### Контрольний приклад грошового розрахунку

Перевіряємо вибір великої суми з високою ймовірністю, відмову за низької
ймовірності або малої суми, виключення невідомої суми й відповідність ID.''')
code('''example = pd.DataFrame({"dispute_id": ["a", "b", "c", "d"],
                        "amount_usd": [100., 100., 20., np.nan], "won": [1, 1, 1, 0]})
example_audit = pd.DataFrame({"dispute_id": example.dispute_id,
                              "score_eligible": [True, True, True, False],
                              "score_observed_only": [True, False, True, False]})
check = economic_table(example, example_audit, np.array([.9, .1, .99, .8]))
chosen = check.query("scenario == 'eligible' and strategy == 'logistic_regression'").iloc[0]
assert chosen.score_usd == 14 and chosen.fight_rows == 1 and chosen.evaluated_rows == 3
assert check.query("scenario == 'eligible' and strategy == 'always_fight'").score_usd.iloc[0] == 26
assert check.query("strategy == 'never_fight'").score_usd.eq(0).all()
assert check.query("scenario == 'observed_only'").evaluated_rows.eq(2).all()
try:
    economic_table(example, example_audit.iloc[::-1].reset_index(drop=True), np.ones(4))
except ValueError:
    pass
else:
    raise AssertionError("Переставлений аудит має бути відхилено")
print("Контроль формули, підмножин та відповідності ID: OK")
''')
md('''## 5. Збереження результатів

Зберігаємо ймовірності, дії, метрики, таблицю доходу й конфігурацію.
Для `review_amount` фактичний дохід лишається невідомим, не нулем.
Ймовірності є для всіх validation-спорів, зокрема з пропущеною сумою.
Модель можна відтворити повторним запуском; pickle-файл не потрібний.''')
code('''eligible = validation_audit.score_eligible.to_numpy(dtype=bool)
amount = validation.amount_usd.to_numpy(dtype=float)
fight = eligible & (.2 * p * amount > 6)
predictions = validation[["dispute_id", "won", "amount_usd"]].copy()
predictions["p_won"] = p
predictions["score_eligible"] = eligible
predictions["score_observed_only"] = validation_audit.score_observed_only.to_numpy()
predictions["action"] = np.where(~eligible, "review_amount", np.where(fight, "fight", "skip"))
predictions["realized_profit_usd"] = np.where(
    eligible, np.where(fight, .2 * amount * y_validation.to_numpy() - 6, 0), np.nan)
assert predictions.loc[~eligible, "realized_profit_usd"].isna().all()
assert np.isclose(predictions.realized_profit_usd.sum(),
                  economics.query("scenario == 'eligible' and strategy == 'logistic_regression'").score_usd.iloc[0])
predictions.to_csv(OUTPUT / "validation_predictions.csv", index=False)
metrics.to_csv(OUTPUT / "validation_probability_metrics.csv", index_label="model")
economics.to_csv(OUTPUT / "validation_economics.csv", index=False)
calibration.to_csv(OUTPUT / "validation_calibration.csv")
source_manifest = json.loads((PROJECT_DIR / "outputs" / "manifest.json").read_text(encoding="utf-8"))
config = {
    "model": "LogisticRegression", "C": 1.0, "seed": 42,
    "categorical": CATEGORICAL, "numeric": NUMERIC,
    "amount_transform": "log1p, train median with missing indicator, StandardScaler",
    "decision": "p * 0.2 * amount_usd > 6; missing amount -> review_amount",
    "test_evaluated": False, "validation_tuned": False,
    "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    "input_sha256": {name: source_manifest["files"][name]["sha256"] for name in
                     ["train_clean.csv", "validation_clean.csv", "validation_audit.csv"]},
}
(OUTPUT / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
selected = economics.query("scenario == 'eligible' and strategy == 'logistic_regression'").iloc[0]
constant = economics.query("scenario == 'eligible' and strategy == 'constant_probability'").iloc[0]
summary = (f"Validation: {len(validation)} disputes; monetary evaluation: {int(selected.evaluated_rows)}.\\n"
           f"Logistic policy Score: {selected.score_usd:.2f} USD.\\n"
           f"Delta vs always fight: {selected.delta_vs_always_usd:+.2f} USD.\\n"
           f"Delta vs constant-probability policy: {selected.score_usd - constant.score_usd:+.2f} USD.\\n"
           f"Amount review required: {int((~eligible).sum())} disputes.\\n"
           "Test has not been evaluated. No validation tuning was performed.\\n")
print(summary)
(OUTPUT / "summary.txt").write_text(summary, encoding="utf-8")
print("Результати:", OUTPUT)
''')
md('''## Висновки та наступний експеримент

Дивись фактичні числа в попередній комірці. Перевага над «завжди» може бути
наслідком відмови від малих сум, тому важливо також порівняння з постійною
ймовірністю. Одна validation-вибірка не доводить стійкість покращення.

Далі: порівняти сильнішу модель на тих самих рядках, перевірити додавання
`amount` разом із валютою та прапорців якості. Після вибору зафіксувати процес
і один раз оцінити test. Поточний ноутбук не виконує цих наступних експериментів.

Обмеження: правила відновлення сум є припущеннями; Score охоплює лише частину
спорів; test переглядали при первинному дослідженні; немає дати отримання результату;
вибірка містить лише історично оскаржені спори з відомим результатом.

Методи: [Pipeline і ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html),
[логістична регресія](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[оцінка калібрування](https://scikit-learn.org/stable/modules/calibration.html).
''')
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})
nbf.validate(notebook)
nbf.write(notebook, ROOT / "03_model_baseline.ipynb")
print("03_model_baseline.ipynb created")
