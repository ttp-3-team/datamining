#!/usr/bin/env bash
# Повне відтворення всіх результатів із dataset.csv у корені репозиторію.
#
#   ./scripts/reproduce.sh            # усе, включно з фінальним test
#   ./scripts/reproduce.sh --no-test  # без відкриття test
#
# Порядок важливий: кожен етап читає перевірені результати попередніх.
# Дослідження з перевіркою запускаються двічі (модуль, потім ноутбук), і
# verify_*.py звіряє контрольні суми обох запусків.
set -euo pipefail
cd "$(dirname "$0")/.."

run() { echo; echo ">>> $*"; uv run "$@"; }
notebook() { run python scripts/check_notebooks.py "$1"; }
first_run() { cp "$1/run_evidence.json" "$1/first_run_evidence.json"; }

[[ -f dataset.csv ]] || { echo "Покладіть dataset.csv у корінь репозиторію" >&2; exit 1; }

run pytest -q

# 1. Дані: огляд і підготовка (outputs/)
notebook 01_data_review
notebook 02_prepare_data

# 2. Перші моделі на validation (model_outputs/, comparison_outputs/)
notebook 03_model_baseline
notebook 04_model_comparison

# 3. Train-only вибір стратегії (train_only_outputs/)
run python -m disputes.train_only_study
notebook 05_train_only_nested_study

# 4. CatBoost і взаємодії на train
run python -m disputes.catboost_study
first_run catboost_outputs
notebook 06_catboost_train_only
run python scripts/verify_catboost_results.py

run python -m disputes.interaction_study
first_run interaction_outputs
notebook 08_logistic_interactions
run python scripts/verify_interaction_results.py

# 5. Повторно використана validation
run python -m disputes.validation_extension
first_run validation_extension_outputs
notebook 07_validation_and_disagreements
run python scripts/verify_validation_extension.py

# 6. Зважене навчання й щотижневий backtest
run python -m disputes.weighted_weekly_study
notebook 09_weighted_weekly_study
run python scripts/verify_weighted_weekly.py

# 7. Загальне зведення методів
run python scripts/build_method_review.py
run python scripts/verify_method_review.py

# 8. Фінальний test: зафіксувати вибір, потім одна оцінка
if [[ "${1:-}" != "--no-test" ]]; then
  run python -m disputes.final_test_evaluation --freeze-only
  run python -m disputes.final_test_evaluation
fi
echo; echo "Готово."
