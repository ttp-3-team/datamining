"""Зібрати відтворюваний звіт із перевірених результатів тривалого backtest."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
md('''# Зважене навчання й щотижнева стійкість моделей

Нові розрахунки виконуються лише всередині train. Зведення старого validation
будується окремо з уже збережених агрегованих чисел; нових validation-прогнозів
немає. Test не відкривається.

Повний перерахунок: `python weighted_weekly_study.py`.
Після нього: `python build_method_review.py`.
Залежності — requirements_catboost.txt. Цей notebook за замовчуванням читає
результати після перевірки хешів. Для повторення повного навчання встановити
RECOMPUTE=True. Це може тривати кілька хвилин.

Усі методи й відносні оцінки зведені в method_review_outputs/method_review.html
та all_methods_summary.csv. Режими train/weekly/validation розділені:
їхні абсолютні суми доходу не можна ранжувати разом.''')
code('''from pathlib import Path
import sys
PROJECT_DIR=Path.cwd()
if not (PROJECT_DIR/"weighted_weekly_study.py").exists():
    raise FileNotFoundError("Задайте PROJECT_DIR")
sys.path.insert(0,str(PROJECT_DIR))
import json, hashlib, base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from weighted_weekly_study import freeze, run
RECOMPUTE=False
OUTPUT,protocol_hash=freeze(PROJECT_DIR)
if RECOMPUTE:
    run(PROJECT_DIR)
manifest=json.loads((OUTPUT/"run_evidence.json").read_text(encoding="utf-8"))
assert manifest["protocol_sha256"]==protocol_hash
def verified_table(name):
    path=OUTPUT/f"{name}.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest["result_sha256"][name]
    return pd.read_csv(path)
primary=verified_table("primary_summary")
weekly=verified_table("weekly_summary")
periods=verified_table("weekly_periods")
print("Зафіксований протокол:",protocol_hash)
print("Прочитано перевірені результати; RECOMPUTE =",RECOMPUTE)
''')
md('''## 2. Навчання з вагами за сумою

Порівнюємо звичайну логістичну регресію, вагу за сумою та вагу за √суми.
Ваги обмежені p99, який визначений тільки на навчальній частині зрізу,
і нормовані до середнього 1. Відсутня сума отримує вагу медіанного відомого
train-спору — це тільки важливість прикладу, не відновлення грошей для Score.
Ваги не залежать від won. Вони передаються лише класифікатору;
навчання preprocessing залишається незваженим.

Для кожної схеми C підбирається окремо всередині зовнішнього train.
Додатково перевіряється sigmoid-калібрування на внутрішніх out-of-time
прогнозах. Вимоги обсягу калібрування залишаються попередніми.
Для Score використовується звичайна грошова формула та ті самі спори,
без навчальних ваг. Вищий зважений training objective сам по собі не
вважається покращенням бізнес-результату.''')
code('''cols=["strategy","score_usd","delta_vs_logistic_usd","uplift_vs_always_percent",
      "uplift_vs_constant_percent","uplift_vs_logistic_percent","log_loss","brier","roc_auc",
      "observed_score_usd","evaluated_rows","observed_rows"]
display(primary[cols].round(5))
display(verified_table("primary_calibration"))
print("Вибір за попереднім практичним правилом:")
print((OUTPUT/"primary_selection.json").read_text())
''')
md('''## 3. Щотижнева стійкість

Прогнозуємо окремо кожен тиждень 10–17, навчаючись на попередніх тижнях.
Для cutoff t внутрішні зрізи: train до t−6 → оцінювання t−5…t−3;
train до t−3 → оцінювання t−2…t. Усі вони завершуються до зовнішнього тижня.
Кожного разу заново обираються C, конфігурації HGB/CatBoost, параметри
очищення, словники категорій, шкали, ваги й калібратори. Ансамблі мають
незмінні ваги 50/50; interaction-варіанти ті самі, що в notebook 08.
Фіксований C=0.1 є додатковою діагностикою, не кандидатом для незалежного вибору.

Усі 17 стратегій оцінюються на тих самих спорах кожного тижня. Показуємо
сумарний Score, приріст на 1000 спорів, кількість тижнів із перевагою,
найгірший тиждень та розкид. Порівняння на 1000 спорів враховує різний
обсяг тижнів, але не усуває відмінностей складу спорів.

Це повторно використані тижні 10–17, а не нова вибірка. Навчальні вікна
перекриваються. Дата отримання результату спору відсутня: припущення, що
всі попередні мітки доступні на наступному тижні, не підтверджене.''')
code('''display(weekly[cols+["periods_above_always","periods_above_logistic",
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
plt.close(fig)
display(HTML('<img alt="Weighted and weekly comparison" width="1150" src="data:image/png;base64,'+base64.b64encode(figure.read_bytes()).decode()+'">'))
''')
md('''## Єдина таблиця всіх методів і висновки

Звіт method_review_outputs/method_review.html містить загальний огляд,
окремі таблиці трьох режимів, C-сітку та ранні feature-ablation спроби HGB.
Порожня оцінка означає «не перевіряли», а не нуль. Нові зважені методи
й взаємодії не отримують вигаданих validation-оцінок.

Відносні оцінки: (Score−baseline)/baseline ×100%; score index ЛР=100;
Score на 1000 оцінених спорів. У кожному режимі свій baseline й кількість рядків.
Метрики ймовірностей рахуються на всіх мічених рядках, Score — на придатних сумах.
Observed-сценарій виключає відновлені/виправлені суми.

Вибір за доходом не означає, що log loss або Brier мають той самий рейтинг.
Практичний поріг незмінний: 50 USD на 1000 спорів; за близьких результатів
перевага простішому варіанту. Статистично переконливий переможець не вимагається.
Результати треба читати з урахуванням повторного використання train/validation,
невідомої доступності міток у часі та історичного відбору спорів.

Ваги sample_weight: [документація LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
nbf.validate(nb)
nbf.write(nb,root/"09_weighted_weekly_study.ipynb")
print("09_weighted_weekly_study.ipynb created")
