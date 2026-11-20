"""Зібрати доповнення validation і пояснення розбіжностей моделей."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# Validation: попередні стратегії, CatBoost і розбіжності рішень

У ноутбуці 04 validation вже використовували для оцінювання й вибору.
У 05–06 діяло обмеження «лише train», тому CatBoost на validation ще не оцінювали.
Цей експеримент за запитом додає нові стратегії до єдиної validation-таблиці.
Test не відкривається. Validation залишається повторно використаною вибіркою.

Параметри нових моделей визначаються тільки за train 0–17; потім моделі
перевіряються на validation 18–21. Старі validation-прогнози не перераховуються
й не замінюються: їхні хеші та Score перевіряються, рядки додаються до нової
об'єднаної таблиці. Старі результати залишаються відтворюваними.

Залежності — requirements_catboost.txt. Запуск: Restart and Run All.
Потрібні модулі пакета, outputs/, train_only_outputs/ і comparison_outputs/.
Усі нові результати — validation_extension_outputs/. Датасет та ID-прогнози
залишаються локально; не публікувати виводи ноутбука чи CSV з окремими спорами.''')
code('''from pathlib import Path
import sys
PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / "validation_extension.py").exists():
    raise FileNotFoundError("Задайте PROJECT_DIR: папку з validation_extension.py")
sys.path.insert(0, str(PROJECT_DIR))
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from validation_extension import freeze, run

OUTPUT, protocol_hash = freeze(PROJECT_DIR)
print("Протокол до нових validation-прогнозів:", protocol_hash)
''')
md('''## Що визначається на train

- C логістичної регресії: найбільший сумарний Score на трьох train-часових
  зрізах із перевіреної C-сітки; log loss і менший C розв'язують рівність.
- CatBoost: чотири вже визначені depth/l2-комбінації на тих самих трьох
  train-зрізах. Вибір за сумарним Score, потім pooled log loss.
- Обидва sigmoid-калібратори: лише out-of-time train-прогнози обраних
  параметрів; калібратор не бачить validation-міток.
- Ансамбль: рівно 50% некаліброваної train-tuned логістичної і 50% CatBoost.
  Ваги та економічний поріг не підбираються за validation.

Параметри та коефіцієнти калібрування записуються в train_parameter_lock.json
до першого відкриття validation в розрахунковому процесі. Старий рядок logistic
має C=1; новий logistic_train_tuned може мати інший C. Це окремі рядки.
Поріг дії всюди p × 0.2 × amount_usd > 6. За невідомої суми — review_amount.

Цей додатковий перегляд validation не перетворює його на незалежний фінальний
тест. Усі нові розрізи — діагностика; за ними не навчаємо правила вибору моделі.''')
code('''summary, pairs, groups, locked = run(PROJECT_DIR)
print("Параметри, визначені лише на train:")
print("Логістична:", locked["logistic_spec"])
print("CatBoost:", locked["catboost_spec"])
columns = ["model", "historical_row", "score_usd", "delta_always_usd", "delta_constant_usd",
           "uplift_vs_always_percent", "log_loss", "brier", "roc_auc",
           "observed_score_usd", "evaluated_rows", "observed_evaluated_rows",
           "score_week_18", "score_week_19", "score_week_20", "score_week_21"]
display(summary[columns].round(5))
print("historical_row=True: збережений рядок попередньої validation-таблиці")
''')
md('''## Що означає «моделі відповідають по-різному»

Є три різні питання:

1. Різні ймовірності: наприклад, 0.7 і 0.8. Обидві моделі можуть радити
   ту саму дію, тому різниця ймовірностей не обов'язково змінює дохід.
2. Різні класи при порозі 0.5: це допомагає зрозуміти класифікаційні помилки,
   але поріг 0.5 не є нашим правилом вигідності.
3. Різні бізнес-дії при p × 0.2 × сума > 6: саме ці спори пояснюють різницю Score.
   У групах, де дії однакові, різниця фактичного доходу має дорівнювати нулю.

Для кожної пари показуємо кількість різних класів і дій, кореляцію ймовірностей
та залишків won−p, спільні/різні помилки, внесок різних рішень у Score.
Класифікаційні підрахунки включають усі validation-спори; грошові — лише
спори з придатними сумами. Є окремий observed-сценарій.
Кореляція ймовірностей сама по собі не вимірює користь для бізнесу.''')
code('''main_pairs = pairs.loc[
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
''')
md('''## Чи можна використати ці розбіжності

Так: вони показують, чи доповнюють моделі одна одну, чи рішення залежать від
кількох прикордонних спорів, у яких групах є проблеми й чи допомагає фіксований
ансамбль. Вже обраний ансамбль оцінюється на тих самих рядках; його користь
перевіряється фактичним Score, а не самою наявністю розбіжностей.

Для конкретного нового спору ми не знаємо won. Вибір «брати ту модель,
яка тут вгадала» за відомим validation-результатом створює витік.
Окрему модель для вибору між моделями або правила за провайдером/причиною
потрібно визначати на out-of-time train-прогнозах і перевіряти зовні.
У цьому експерименті такого маршрутизатора немає.

near_boundary означає, що хоча б один прогноз очікуваного чистого доходу
перебуває в межах 1 USD від нуля. Це зафіксований діагностичний поріг,
який не впливає на рішення моделі. Сегменти з малою кількістю спорів
мають слабку доказову силу; показані асоціації не доводять причинність.''')
code('''weekly = pd.read_csv(OUTPUT / "validation_weekly.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
view = summary.sort_values("score_usd")
axes[0].barh(view.model, view.score_usd, color=np.where(view.historical_row, "#7895b2", "#348c68"))
axes[0].axvline(5324., linestyle="--", color="gray", label="Always fight")
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
plt.close(fig)
display(HTML('<img alt="Extended validation comparison" width="1150" src="data:image/png;base64,'
             + base64.b64encode(path.read_bytes()).decode() + '">'))
''')
md('''## Висновок має враховувати межі перевірки

Звіт показує повторно використаний validation, а не остаточний test.
Старі числа відтворено, нові параметри визначено на train, тестові файли
заблоковані guard-ом. Повторення за тим самим протоколом і незалежний
рядковий розрахунок Score перевіряють реалізацію, а не майбутню прибутковість.
Дати завершення спорів немає; реальна часова доступність міток не підтверджена.

Повна таблиця: validation_summary.csv. Усі пари: pairwise_disagreement.csv.
Чотири детальні пари: disagreement_details.csv; групи рішень:
decision_disagreement_groups.csv; розрізи PSP/причини/методу:
disagreement_segments.csv. Ці файли не призначені для публікації.

Метод усереднення ймовірностей: [scikit-learn, soft voting](https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting).
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})
nbf.validate(nb)
nbf.write(nb, root / "07_validation_and_disagreements.ipynb")
print("07_validation_and_disagreements.ipynb created")
