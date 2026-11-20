"""Зібрати ноутбук дослідження HistGradientBoosting і перевірок рішення."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# Порівняння моделей і перевірка бізнес-рішення

Мета — підвищити чистий дохід та обґрунтувати рішення відповідно до вимог хакатону.
Це дослідження до фінальної перевірки. **Test не завантажується й не оцінюється.**

Вимоги: 15 балів за захист і презентацію, 8 — peer evaluation, 2 — self evaluation.
Журі оцінює аналітику даних, бізнес-інтерпретацію та пояснення рішень.
Детальних числових ваг усередині 15 балів немає. Вищий Score сам по собі не гарантує оцінки.
Для здачі ще потрібні результати test і презентація на 10 хвилин.

Дані залишаються локально. Не публікувати їх, прогнози з ID чи ноутбуки з рядками даних.
AI використано для допомоги з кодом і перевірками; на захисті це слід задекларувати.

Встановити `requirements_comparison.txt`. Запустити всі комірки в чистій сесії.
`model_comparison.py`, `preparation.py` та `outputs/` мають бути поруч.
Усі нові виходи — в `comparison_outputs/`.''')
code('''from pathlib import Path
import sys
PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / "model_comparison.py").exists():
    raise FileNotFoundError("Задайте PROJECT_DIR: папку з model_comparison.py")
sys.path.insert(0, str(PROJECT_DIR))
import json
import base64
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.metrics import confusion_matrix, log_loss
from model_comparison import (
    CANDIDATES, FOLDS, load_inputs, cross_validate, fit_finalists, evaluate,
    policy, profit_rows, week_diagnostics, paired_block_intervals,
    permutation_diagnostics, segment_diagnostics, export_run, features,
)

OUTPUT = PROJECT_DIR / "comparison_outputs"
OUTPUT.mkdir(exist_ok=True)
train, train_audit, validation, validation_audit, normalized, source_audit = load_inputs(PROJECT_DIR)
print("Train:", len(train), "Validation:", len(validation))
print("Test і unknown_time не завантажувалися")
# Протокол записується до виконання пошуку; його не змінюємо після перегляду результатів.
plan = {
    "candidates": CANDIDATES, "folds": FOLDS, "iterations": 400, "learning_rate": .05,
    "cv_selection": "pooled Score, mean log loss as tie breaker",
    "validation_finalists": ["logistic", "boosting", "boosting_calibrated", "blend_50_50"],
    "decision": "p * 0.2 * amount_usd > 6", "test_evaluated": False,
}
(OUTPUT / "experiment_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
display(pd.DataFrame(CANDIDATES))
''')
md('''## 1. Часовий підбір усередині train

Три послідовні зрізи: train 0–5 → оцінювання 6–9; 0–9 → 10–13;
0–13 → 14–17. Для кожного зрізу **заново визначаємо опору очищення сум тільки
на ранній частині**, відновлюючи початкові train-значення з аудиту. Використання
готових очищених сум із опорою всіх тижнів 0–17 тут давало б доступ до майбутньої статистики.

Модельні перетворення також навчаються лише на ранній частині.
HistGradientBoosting: 400 ітерацій, learning rate 0.05, без early stopping. Рядки впорядковано
за тижнем; кожен фолд оцінюється лише на пізніших тижнях.
Категорії — рядки з MISSING, числові NaN підтримуються моделлю. Категорії кодуються OrdinalEncoder, але модель отримує явний список категоріальних
ознак: номери кодів не є числовою шкалою. Нові категорії переходять у NaN.
Словник категорій навчається тільки на train. Немає балансування й target-агрегатів. Early stopping не використовує validation.

Порівнюємо глибини 4/6 і внесок суми в оригінальній валюті, прапорців якості,
номера тижня та ID продавця. За незмінної глибини це окремі перевірки внеску ознак,
а не повний перебір їхніх комбінацій. Переможець HistGradientBoosting — найбільший сумарний
Score на однакових пізніших фолдах; за рівності — менший середній log loss.
Ці CV-оцінки беруть участь у виборі, тому не є незалежним фінальним результатом.

Часовий поділ не гарантує доступності міток: дата завершення спору відсутня.
Це обмеження не усувається самим часовим поділом.''')
code('''cv_scores, cv_ranking, best_spec, oof, folds = cross_validate(normalized, source_audit)
display(folds)
display(cv_ranking.round(4))
print("Обрана конфігурація HistGradientBoosting:", best_spec)
display(cv_scores.query("scenario == 'eligible'").pivot(index="candidate", columns="fold", values="score_usd").round(2))
''')
md('''## 2. Фіксовані фіналісти на validation

Навчаємо обраний HistGradientBoosting і ту саму логістичну регресію на всіх train-тижнях 0–17.
Додатково перевіряємо sigmoid-калібрування HistGradientBoosting: калібратор навчається тільки
на out-of-time прогнозах усередині train. Це приблизна корекція; вона може не
переноситися на модель, навчену на більшій вибірці.
Ансамбль — рівно 50% логістичної регресії і 50% HistGradientBoosting; ваги не підбираються.

Validation використовується для вибору серед цих чотирьох фіксованих варіантів
за Score (log loss розв'язує рівність). Тому наведені нижче числа — оцінки для
вибору, не незалежне підтвердження якості. Поріг бізнес-рішення незмінний:
`p × 0.2 × amount_usd > 6`. Поріг 0.5 застосовується лише для звичних метрик класифікації.''')
code('''logistic, boosting, calibrator, probabilities, metrics, economics, selected = fit_finalists(
    train, train_audit, validation, validation_audit, best_spec, oof)
display(metrics.round(4))
display(economics.round(2))
print("Вибраний на validation варіант:", selected)
display(pd.DataFrame(confusion_matrix(validation.won.astype(int), probabilities[selected] >= .5),
                     index=["actual_lost", "actual_won"], columns=["predicted_lost", "predicted_won"]))
''')
md('''## 3. Друга перевірка: базова модель, стійкість і невизначеність

Перевіряємо відтворення попереднього baseline, відповідність рядкового Score
та таблиці, незалежність рішень від `won`, покриття грошовим оцінюванням.
Дохід невідомих сум не замінюється медіаною чи нулем.

Показуємо дохід за кожним тижнем і парні block-bootstrap інтервали різниці доходу:
перевибір тижнів або продавців, 2 000 повторів. Лише чотири validation-тижні —
надто мало для надійного висновку про часову невизначеність. Перевибір продавців
не враховує спільні часові шоки. Інтервали описові, умовні на навчених моделях
і не враховують оптимізм від вибору переможця на цих даних.''')
code('''baseline_path = PROJECT_DIR / "model_outputs" / "validation_predictions.csv"
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
''')
md('''## 4. Пояснення закономірностей і помилок

Permutation importance: наскільки гіршим стає log loss HistGradientBoosting при перемішуванні
однієї ознаки на validation (5 повторів). Це пояснення асоціацій моделі, не
причинний вплив. Пов'язані ознаки можуть підміняти одна одну. Після перегляду
цих діагностик нові моделі в цьому експерименті не підбираємо.

Розрізи вибраної стратегії: спосіб оплати, причина спору, сума та відомі/нові
продавці. Результат на малих групах нестійкий; відсутність нових продавців
означає, що узагальнення на них тут не перевірено.''')
code('''importance = permutation_diagnostics(boosting, validation, validation_audit, best_spec)
segments = segment_diagnostics(validation, validation_audit, p_selected, train)
display(importance.round(4))
display(segments.round(3))
calibration_frames = []
for name, p in probabilities.items():
    frame = pd.DataFrame({"p": p, "won": validation.won.to_numpy()})
    frame["bin"] = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    grouped = frame.groupby("bin", observed=True).agg(rows=("won", "size"),
                  predicted=("p", "mean"), observed=("won", "mean")).reset_index()
    grouped["model"] = name
    calibration_frames.append(grouped)
calibration = pd.concat(calibration_frames, ignore_index=True)
display(calibration.query("model == @selected"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name in ["logistic", "boosting", "boosting_calibrated", "blend_50_50"]:
    group = calibration.query("model == @name")
    axes[0].plot(group.predicted, group.observed, "o-", label=name, markersize=4)
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
plt.close(fig)
display(HTML('<img alt="Validation model comparison" width="1000" src="data:image/png;base64,'
             + base64.b64encode(figure.read_bytes()).decode() + '">'))
''')
md('''## 5. Експорт і зафіксований вибір

Зберігаємо повний журнал кандидатів і всі результати, включно з невдалими.
Після збереження перевіряємо повторне завантаження моделей та тотожність прогнозів.
Рядки validation з ID та моделі призначені для локального відтворення.
Не завантажувати датасет, його фрагменти чи ID-прогнози до сторонніх сервісів.

Рекомендація бізнесу в межах припущень задачі: оцінювати ймовірність виграшу,
порівнювати очікуваний виграш із 6 USD; відсутню суму спочатку уточнювати.
Перед реальним впровадженням перевірити актуальну вартість дії, доступність усіх
ознак до рішення й зсув складу спорів. Це історична симуляція на відібраних
оспорених випадках, а не доведений причинний ефект для всіх спорів.''')
code('''output, config = export_run(
    PROJECT_DIR, best_spec, (logistic, boosting, calibrator), probabilities, metrics, economics,
    selected, cv_scores, cv_ranking, folds, oof, weekly, intervals, importance, segments,
    validation, validation_audit)
calibration.to_csv(output / "validation_calibration.csv", index=False)
row = economics.query("scenario == 'eligible' and model == @selected").iloc[0]
lr_row = economics.query("scenario == 'eligible' and model == 'logistic'").iloc[0]
summary = (f"Selected on validation: {selected}\\n"
           f"HistGradientBoosting candidate selected by temporal CV: {best_spec['name']}\\n"
           f"Validation Score: {row.score_usd:.2f} USD on {int(row.rows)} disputes\\n"
           f"Delta vs always: {row.delta_vs_always_usd:+.2f} USD\\n"
           f"Delta vs logistic: {row.score_usd - lr_row.score_usd:+.2f} USD\\n"
           "Validation used for selection; test has not been evaluated.\\n")
(output / "summary.txt").write_text(summary, encoding="utf-8")
print(summary)
print("Повторне завантаження моделей і контроль Score: OK")
print("Лідер внутрішнього часового CV:", cv_ranking.index[0])
logistic_intervals = intervals.query("reference == 'logistic'")
if ((logistic_intervals.p025_usd <= 0) & (logistic_intervals.p975_usd >= 0)).any():
    print("Описові інтервали різниці з логістичною регресією охоплюють 0: стійку перевагу не встановлено.")
print("Формальний вибір за validation Score не замінює фінальної перевірки на test.")
''')
md('''## Що ще потрібно до здачі

1. Зафіксувати модель, її калібрування/ваги та правило дії до оцінювання test.
   Якщо після вибору моделі перенавчати її на train+validation, наперед визначити
   спосіб калібрування; не використовувати test для його налаштування.
2. Один раз оцінити test: класифікаційні метрики, Score проти «завжди» на тих самих
   рядках, покриття та сценарій без виправлених/відновлених сум. Не підбирати далі за test.
3. Побудувати 10-хвилинну презентацію: задача → закономірності → очищення й поділ
   → порівняння → правило дії → test → рекомендації й обмеження. Задекларувати AI
   та вміти пояснити кожне рішення. Peer/self evaluation мають відображати реальний внесок.

Не робимо висновку про оцінку викладача або майбутній дохід лише з validation.

Джерела методів: [HistGradientBoosting: категорії та пропуски](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting),
[калібрування](https://scikit-learn.org/stable/modules/calibration.html),
[permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})
nbf.validate(nb)
nbf.write(nb, root / "04_model_comparison.ipynb")
print("04_model_comparison.ipynb created")
