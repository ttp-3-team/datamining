"""Зібрати пояснювальний ноутбук вкладеної train-only перевірки."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# Перевірка стратегій лише всередині train

Цей експеримент доповнює ноутбуки 03–04. Він не відкриває validation або test.
Вхід — **тільки `outputs/train_audit.csv`**: із початкових полів аудиту
відновлюємо train до обробки сум. Правила очищення сум навчаються заново
в кожному часовому зрізі, включно з внутрішніми.

Завдання: перевірити регуляризацію логістичної регресії; порівняти звичайні й
калібровані логістичну регресію та бустинг, а також фіксований ансамбль 50/50
за однаковою вкладеною часовою схемою. Це дослідження для вибору стратегії,
а не фінальна незалежна перевірка: ці train-періоди вже аналізували раніше.

Запуск: встановити `requirements_comparison.txt`, відкрити ноутбук поряд із
`train_only_study.py`, `model_comparison.py`, `preparation.py` і `outputs/`.
За потреби змінити PROJECT_DIR. Виконати Restart and Run All.
Дані, прогнози з ID та виводи ноутбука залишаються локальними.''')
code('''from pathlib import Path
import sys
PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / "train_only_study.py").exists():
    raise FileNotFoundError("Задайте PROJECT_DIR: папку з train_only_study.py")
sys.path.insert(0, str(PROJECT_DIR))
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from train_only_study import PROTOCOL, freeze_protocol, run

# Ця операція передує читанню даних і розрахункам. Наявний протокол
# перевіряється на збіг; змінений протокол не перезаписується автоматично.
OUTPUT, protocol_hash = freeze_protocol(PROJECT_DIR)
print("SHA-256 зафіксованого протоколу:", protocol_hash)
display(pd.DataFrame({"outer_train": ["0–5", "0–9", "0–13"],
                      "outer_evaluation": ["6–9", "10–13", "14–17"],
                      "inner_train_to_evaluation": ["0–1 → 2–3; 0–3 → 4–5",
                          "0–3 → 4–6; 0–6 → 7–9", "0–5 → 6–9; 0–9 → 10–13"]}))
print("Практична близькість: 50 USD на 1 000 оцінених спорів")
''')
md('''## Правило вибору, зафіксоване до нових результатів

Основний критерій — сумарний Score на доступних USD-сумах. Практично близькими
вважаємо стратегії в межах **0.05 USD × кількість оцінених спорів** від найбільшого
доходу. Це відкрите робоче припущення про достатню користь від ускладнення,
а не поріг із завдання викладача й не статистичний критерій.

Серед близьких варіантів обираємо простіший: постійна train-ймовірність →
логістична регресія → калібрована логістична → бустинг → калібрований бустинг →
ансамбль. За однакової складності спочатку кращий найгірший приріст до постійної
стратегії на один спір, потім менший розкид такого приросту за періодами,
менший pooled log loss і назва стратегії для детермінованого розв'язання рівності.
Складність — зафіксований порядок підтримки моделей, не універсальна числова міра.

Результат без відновлених/виправлених USD-сум є перевіркою чутливості,
не додатковим способом післяфактум змінити правило вибору. Немає вимоги
обов'язково знайти статистично переконливого переможця.''')
md('''## Вкладені параметри та калібрування

Для кожного зовнішнього train-періоду внутрішні два зрізи порівнюють
`C=0.01, 0.1, 1, 10, 100` і шість уже визначених конфігурацій бустингу.
Обираємо найбільший внутрішній сумарний Score; за рівності — менший pooled
log loss, менший C або назву конфігурації. Весь preprocessing навчається
заново. Параметри для зовнішнього прогнозу не визначаються за його мітками.

Для кожної обраної моделі збираємо її out-of-time прогнози на внутрішніх
пізніших підперіодах. Саме вони та відповідні внутрішні мітки навчають
sigmoid-калібратор. Базова модель, що дала кожен такий прогноз, бачила
лише ранніші тижні. Мітки зовнішнього оцінюваного періоду калібратору недоступні.
Внутрішні прогнози беруть участь і у виборі параметрів, і в калібруванні;
тому їхню якість не видаємо за незалежну оцінку — для цього служить зовнішній зріз.

Калібрування дозволене за >=1 000 прогнозів, >=50 прикладів кожного класу
і >=2 внутрішніх періодів. Це евристика достатності обсягу, не гарантія
надійності. Якщо даних замало, використовуємо некалібровані прогнози,
явно позначаємо fallback і виключаємо такий калібрований варіант із вибору.

Ансамбль має незмінні ваги: 50% звичайної налаштованої логістичної регресії
і 50% звичайного налаштованого бустингу. Після збору прогнозів для калібрування
базові моделі перенавчаються на всьому зовнішньому train. Відповідність
калібратора моделям на більшому train не гарантована — її оцінюємо зовні.

Окрема C-сітка на всіх трьох зовнішніх періодах — діагностика чутливості.
Її глобальний переможець не підміняє вкладену стратегію, яка обирає C лише
за минулими даними. Поріг дії скрізь однаковий: `p × 0.2 × amount_usd > 6`.''')
code('''summary, periods, C_grid, calibration, selection = run(PROJECT_DIR)
display(C_grid.round(5))
print("C: результати кожного періоду")
display(pd.read_csv(OUTPUT / "C_grid_periods.csv").round(5))
print("Параметри, вибрані всередині кожної зовнішньої навчальної частини")
display(pd.read_csv(OUTPUT / "outer_choices.csv"))
print("Достатність даних калібрування")
display(calibration)
''')
md('''## Єдина таблиця стратегій

Кожен рядок означає повний алгоритм: внутрішній вибір параметрів → за потреби
калібрування → навчання базової моделі → прогноз майбутнього періоду.
Класифікаційні метрики рахуються на всіх зовнішніх спорах, включно з невідомою
сумою. Score — лише на однакових придатних рядках; їх кількість указана окремо.

Pooled log loss і Brier рахуються за об'єднаними зовнішніми прогнозами,
а не незваженим середнім фолдів. Pooled ROC AUC додатково містить ранжування
між періодами; ROC AUC кожного періоду також доступний окремо.
`observed_*` виключає відновлені та виправлені USD-суми. Підмножини між
основним і observed-сценаріями відрізняються; моделі всередині сценарію — ні.
Невідомі суми не перетворюються на нулі для оцінювання доходу.''')
code('''columns = ["strategy", "score_usd", "delta_always_usd", "delta_constant_usd",
           "score_weeks_6-9", "score_weeks_10-13", "score_weeks_14-17",
           "log_loss", "brier", "roc_auc", "observed_score_usd",
           "observed_delta_always_usd", "observed_delta_constant_usd",
           "evaluated_rows", "observed_rows", "classification_rows", "selectable"]
display(summary[columns].round(5))
display(periods.round(5))
print("Вибір за протоколом:", selection["selected_strategy"])
print("Формальний максимум Score:", selection["maximum_score_strategy"])
print("Поріг практичної близькості, USD:", selection["practical_margin_usd"])
print("Відставання обраного варіанта від максимуму, USD:", selection["selected_gap_to_max_usd"])
print("Близькі варіанти:", selection["near_best_strategies"])
if not calibration.adequate_by_protocol.all():
    print("УВАГА: є періоди з недостатнім обсягом калібрування — див. status.")
else:
    print("Усі калібратори пройшли евристичне правило обсягу; це не гарантує точного калібрування.")
''')
code('''fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for c, group in pd.read_csv(OUTPUT / "C_grid_periods.csv").groupby("C"):
    axes[0].plot(group.outer_period, group.delta_constant_usd / group.evaluated_rows,
                 "o-", label=f"C={c:g}")
axes[0].set(title="Fixed C sensitivity", xlabel="Outer evaluation weeks",
            ylabel="Uplift vs train-rate policy, USD/dispute")
axes[0].legend(fontsize=8)
for name, group in periods.groupby("strategy", sort=False):
    if name != "constant_train_rate":
        axes[1].plot(group.outer_period, group.delta_constant_usd / group.evaluated_rows,
                     "o-", label=name)
axes[1].set(title="Nested strategy comparison", xlabel="Outer evaluation weeks",
            ylabel="Uplift vs train-rate policy, USD/dispute")
axes[1].legend(fontsize=7)
fig.tight_layout()
figure = OUTPUT / "train_only_comparison.png"
fig.savefig(figure, dpi=150)
plt.close(fig)
display(HTML('<img alt="Train-only nested temporal comparison" width="1100" src="data:image/png;base64,'
             + base64.b64encode(figure.read_bytes()).decode() + '">'))
''')
md('''## Межі висновку та наступне використання

- Протокол вибирає стратегію, а не доводить наявність статистично переконливого
  переможця. Три зовнішні періоди — мало для сильних тверджень про стійкість.
- Ці train-дані вже використовували для EDA та попереднього порівняння.
  Вкладена схема виправляє локальний підбір параметрів/калібрування, але не
  скасовує попереднього знайомства з даними та вибору простору кандидатів.
- Раніше validation використовували для вибору; цей експеримент не повертає
  йому незалежності. Тут його файли не відкривалися. Test також не відкривався.
- Дата отримання результату спору відсутня: не можна перевірити, чи навчальні
  мітки реально були доступні на момент наступного прогнозу. Це суттєве
  невирішене обмеження всіх часових перевірок у цій роботі.
- Вибірка містить лише історично оскаржені спори з відомим результатом.
  Відновлення сум і покриття Score також обмежують узагальнення.

Не переналаштовувати моделі за test. Перед наступним зовнішнім оцінюванням
треба зафіксувати обраний алгоритм, спосіб навчання на всьому доступному train,
калібрування й правило дії. У цьому ноутбуці немає оцінювання validation/test
або підгонки практичного порога після перегляду результатів.

`protocol.json` містить зафіксоване правило, `selection.json` — вибір і параметри
кожного зовнішнього зрізу, `run_evidence.json` — хеші коду/результатів і журнал
фактично відкритих вхідних CSV. Python audit hook блокує відкриття інших
таблиць `outputs/`, а також dataset/validation/test/unknown_time CSV.

Метод калібрування та вимога відокремлення прогнозованих прикладів від
навчання базової моделі: [документація scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})
nbf.validate(nb)
nbf.write(nb, root / "05_train_only_nested_study.ipynb")
print("05_train_only_nested_study.ipynb created")
