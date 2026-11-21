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
# # Підготовка даних і передача команді
#
# Послідовний процес: початковий CSV → уніфікація → часові частини → правила сум → перевірки → експорт. Моделі не навчаються. Обґрунтування рішень — у `01_data_review.ipynb`.
#
# Повний запуск від початкового CSV можна повторювати. Готові CSV не подавати назад як початковий датасет: це стирало б походження виправлень.

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
# Видаляємо тільки повні дублікати, виправляємо формат категорій і типи. Початкові значення зберігаємо окремо. Повторні ID з різним вмістом або некоректний target зупиняють процес.

# %%
normalized, source_audit = normalize_source(raw)
print(f"Початково: {len(raw):,}; після видалення повних дублікатів: {len(normalized):,}")

# %% [markdown]
# ## 2. Часові частини
#
# Правила, що спираються на статистику, визначаємо лише за ранніми тижнями (train). Потім без переналаштування застосовуємо їх до validation, test і рядків без відомого тижня. Це частина підготовки даних, а не навчання моделі.

# %%
parts = split_by_time(normalized)
reference = fit_scale_reference(parts["train"])
display(reference[["pairs", "median", "lower", "upper"]])

# %% [markdown]
# ## 3. Обробка сум
#
# Застосовуємо правило масштабу, вилучення непридатних значень і відновлення USD-пар. Невизначені значення стають пропусками; їхні початкові суми залишаються в аудиті. У всіх частинах однакові правила. Test тут лише обробляється, його дохід не оцінюємо.

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
# Очищення не повинно губити спори після дедуплікації, змішувати частини або залишати суму для Score там, де її вважаємо невизначеною.

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
# Основні таблиці містять 13 колонок. Аудит — окремо, з ключем `dispute_id`. Manifest описує типи та контрольні суми. Після зміни правил команда повинна отримати нову версію всього експорту, а не окремий виправлений CSV.

# %%
manifest = export_results(clean, audits, reference, DATASET, OUTPUT)
print(f"Результати: {OUTPUT.resolve()}")
print(f"Версія підготовки: {manifest['preparation_version']}")

# %% [markdown]
# ## 6. Перевірка читання
#
# CSV не зберігає типи. Читаємо через `load_table`, щоб коди не стали числами, а nullable-типи та прапорці відновилися правильно.

# %%
for name in clean:
    loaded = load_table(OUTPUT, f"{name}_clean.csv")
    loaded_audit = load_table(OUTPUT, f"{name}_audit.csv")
    pd.testing.assert_frame_equal(loaded, clean[name].reset_index(drop=True))
    pd.testing.assert_frame_equal(loaded_audit, audits[name].reset_index(drop=True))
print("Усі таблиці й аудит після експорту відновлено без зміни значень і типів.")

# %% [markdown]
# ## 7. Контроль грошових сум — без моделі
#
# Цей блок лише перевіряє, як команда надалі рахуватиме baseline «боротися завжди»: `0.2 * amount_usd * won - 6`. Він не навчає модель і не вибирає рішення.
#
# `score_eligible` означає наявність придатної USD-суми. `score_observed_only` додатково виключає відновлені й виправлені суми. Дохід стосується лише зазначених рядків. Різниця двох підсумків не є ефектом корекції: підмножини різні. Test не оцінюємо.

# %%
baseline_checks = pd.DataFrame({
    "придатні_суми": calculate_baseline(clean["validation"], audits["validation"]),
    "без_відновлених_і_виправлених": calculate_baseline(
        clean["validation"], audits["validation"], observed_only=True
    ),
}).T
display(baseline_checks)
