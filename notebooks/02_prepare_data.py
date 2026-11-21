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
# # Підготовка й експорт даних
#
# Послідовність етапів: початковий CSV → уніфікація → часовий поділ → правила сум → перевірки → експорт. Моделі не навчаються. Обґрунтування правил наведено в `01_data_review.ipynb`.
#
# Підготовка відтворюється повністю з початкового CSV. Експортовані CSV не використовуються як початковий набір даних, оскільки це знищило б інформацію про походження виправлень.

# %%

import pandas as pd
from IPython.display import display

from disputes.paths import DATASET, OUTPUTS
from disputes.preparation import (
    calculate_baseline,
    export_results,
    fit_scale_reference,
    load_table,
    normalize_source,
    prepare_amounts,
    quality_summary,
    split_by_time,
)

if not DATASET.exists():
    raise FileNotFoundError("Покладіть dataset.csv у корінь репозиторію")
OUTPUT = OUTPUTS
raw = pd.read_csv(DATASET)


# %% [markdown]
# ## 1. Уніфікація
#
# Видаляються лише повні дублікати; формати категорій і типи колонок уніфікуються. Початкові значення зберігаються окремо. Повторні ID з різним вмістом або некоректне значення цілі зупиняють обробку.

# %%
normalized, source_audit = normalize_source(raw)
print(f"Початково: {len(raw):,}; після видалення повних дублікатів: {len(normalized):,}")

# %% [markdown]
# ## 2. Часовий поділ
#
# Статистичні правила визначаються лише на train і без повторного налаштування застосовуються до validation, test і рядків без відомого тижня. Цей етап належить до підготовки даних, а не до навчання моделі.

# %%
parts = split_by_time(normalized)
reference = fit_scale_reference(parts["train"])
display(reference[["pairs", "median", "lower", "upper"]])

# %% [markdown]
# ## 3. Обробка сум
#
# Застосовуються правило масштабу, вилучення непридатних значень і відновлення пар сум у USD. Невизначені значення замінюються пропусками; їхні початкові значення зберігаються в аудиті. Правила однакові для всіх частин. Test лише обробляється, Score на ньому не обчислюється.

# %%
clean, audits = {}, {}
for name, part in parts.items():
    clean[name], audits[name] = prepare_amounts(
        part, source_audit.loc[part.index], reference
    )
display(quality_summary(clean, audits))

# %% [markdown]
# ## 4. Перевірки цілісності
#
# Перевіряється, що після дедуплікації жоден спір не втрачено, частини не перетинаються, а для невизначених сум відсутня сума, придатна для Score.

# %%
all_ids = pd.concat([part.dispute_id for part in clean.values()])
assert all_ids.is_unique
assert len(all_ids) == len(normalized)
assert set(all_ids) == set(normalized.dispute_id)
for name, part in clean.items():
    audit = audits[name]
    assert part.dispute_id.equals(audit.dispute_id)
    assert part.loc[audit.amount_usd_needs_review, "amount_usd"].isna().all()
    assert part.loc[audit.score_eligible, "amount_usd"].gt(0).all()
print("Цілісність рядків, зв'язок з аудитом і доступність сум: OK")

# %% [markdown]
# ## 5. Експорт
#
# Основні таблиці містять 13 колонок. Аудит зберігається окремо з ключем `dispute_id`. Manifest містить типи колонок і контрольні суми. Після зміни правил експорт формується повторно цілком.

# %%
manifest = export_results(clean, audits, reference, DATASET, OUTPUT)
print(f"Результати: {OUTPUT.resolve()}")
print(f"Версія підготовки: {manifest['preparation_version']}")

# %% [markdown]
# ## 6. Перевірка читання
#
# CSV не зберігає типи, тому таблиці читаються через `load_table`: коди залишаються текстом, nullable-типи та прапорці відновлюються.

# %%
for name in clean:
    loaded = load_table(OUTPUT, f"{name}_clean.csv")
    loaded_audit = load_table(OUTPUT, f"{name}_audit.csv")
    pd.testing.assert_frame_equal(loaded, clean[name].reset_index(drop=True))
    pd.testing.assert_frame_equal(loaded_audit, audits[name].reset_index(drop=True))
print("Усі таблиці й аудит після експорту відновлено без зміни значень і типів.")

# %% [markdown]
# ## 7. Контроль сум без моделі
#
# Блок перевіряє обчислення Score базової стратегії «боротися завжди»: 0,2 × amount_usd × won − 6 для кожного спору. Модель не навчається, рішення не обирається.
#
# `score_eligible` позначає наявність суми, придатної для Score. `score_observed_only` додатково виключає відновлені й виправлені суми. Підсумки обчислюються лише на відповідних рядках; оскільки підмножини різні, різниця двох підсумків не є ефектом виправлення. Test не оцінюється.

# %%
baseline_checks = pd.DataFrame({
    "придатні_суми": calculate_baseline(clean["validation"], audits["validation"]),
    "без_відновлених_і_виправлених": calculate_baseline(
        clean["validation"], audits["validation"], observed_only=True
    ),
}).T
display(baseline_checks)
