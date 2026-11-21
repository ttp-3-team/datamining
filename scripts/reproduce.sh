#!/usr/bin/env bash
# Reproduce all results from dataset.csv in the repository root.
#
#   ./scripts/reproduce.sh            # everything, including both test evaluations
#   ./scripts/reproduce.sh --no-test  # without opening test
#
# Order matters: each stage reads verified results of the previous ones.
# Verified studies run twice (module, then notebook); verify_*.py compares the checksums of both runs.
set -euo pipefail
cd "$(dirname "$0")/.."

run() { echo; echo ">>> $*"; uv run "$@"; }
notebook() { run python scripts/check_notebooks.py "$1"; }
first_run() { cp "$1/run_evidence.json" "$1/first_run_evidence.json"; }

[[ -f dataset.csv ]] || { echo "Put dataset.csv in the repository root" >&2; exit 1; }

run pytest -q

# 1. Data review and preparation (outputs/)
notebook 01_data_review
notebook 02_prepare_data

# 2. First models on validation (model_outputs/, comparison_outputs/)
notebook 03_model_baseline
notebook 04_model_comparison

# 3. Train-only strategy selection (train_only_outputs/)
run python -m disputes.train_only_study
notebook 05_train_only_nested_study

# 4. CatBoost and interactions on train
run python -m disputes.catboost_study
first_run catboost_outputs
notebook 06_catboost_train_only
run python scripts/verify_catboost_results.py

run python -m disputes.interaction_study
first_run interaction_outputs
notebook 08_logistic_interactions
run python scripts/verify_interaction_results.py

# 5. Reused validation
run python -m disputes.validation_extension
first_run validation_extension_outputs
notebook 07_validation_and_disagreements
run python scripts/verify_validation_extension.py

# 6. Weighted training and weekly backtest
run python -m disputes.weighted_weekly_study
notebook 09_weighted_weekly_study
run python scripts/verify_weighted_weekly.py

# 7. Method review
run python scripts/build_method_review.py
run python scripts/verify_method_review.py

# 8. First test evaluation: freeze the choice, then evaluate once
if [[ "${1:-}" != "--no-test" ]]; then
  run python -m disputes.final_test_evaluation --freeze-only
  run python -m disputes.final_test_evaluation
fi

# 9. Second evaluation: recipes chosen on weeks 0–21, then a single test evaluation
run python -m disputes.second_evaluation develop
if [[ "${1:-}" != "--no-test" ]]; then
  run python -m disputes.second_evaluation freeze
  run python -m disputes.second_evaluation test
  notebook 10_second_evaluation
fi
echo; echo "Done."
