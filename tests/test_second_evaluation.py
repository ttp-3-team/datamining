"""Нові ознаки другої оцінки не бачать майбутнього."""
import unittest

import numpy as np
import pandas as pd

from disputes import second_evaluation as study
from disputes.backtest import time_window
from disputes.models import merchant_history
from synthetic import synthetic_source


class MerchantHistoryChecks(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame({"merchant_id": ["A", "A", "A", "B", None], "dispute_week": [0, 1, 2, 1, 1],
                                     "won": [1, 0, 1, 0, 1]})

    def test_training_rows_use_only_earlier_weeks(self):
        result = merchant_history(self.history, self.history, prior=1e-12)
        # Перший спір мерчанта A не має історії → загальна частка; другий бачить лише тиждень 0.
        self.assertEqual(result.merchant_hist_log_n.tolist()[:3], [0., np.log1p(1), np.log1p(2)])
        self.assertAlmostEqual(result.merchant_hist_rate.iloc[1], 1.)
        self.assertAlmostEqual(result.merchant_hist_rate.iloc[2], .5)

    def test_same_week_and_future_labels_do_not_leak(self):
        changed = self.history.copy()
        changed.loc[changed.dispute_week.ge(1), "won"] = 1 - changed.loc[changed.dispute_week.ge(1), "won"]
        before, after = merchant_history(self.history, self.history), merchant_history(changed, changed)
        np.testing.assert_allclose(before.merchant_hist_log_n, after.merchant_hist_log_n)
        # Рядки тижнів 0–1 бачать лише тиждень 0, тож зміна міток від тижня 1 їх не зачіпає (крім загальної частки).
        self.assertEqual(before.merchant_hist_log_n.iloc[1], after.merchant_hist_log_n.iloc[1])

    def test_evaluation_rows_see_all_history_and_missing_merchant_gets_prior(self):
        rows = pd.DataFrame({"merchant_id": ["A", "C", None], "dispute_week": [5, 5, 5], "won": [0, 0, 0]})
        result = merchant_history(self.history, rows, prior=20.)
        overall = self.history.won.mean()
        self.assertAlmostEqual(result.merchant_hist_rate.iloc[0], (2 + 20 * overall) / (3 + 20))
        self.assertAlmostEqual(result.merchant_hist_rate.iloc[1], overall)
        self.assertAlmostEqual(result.merchant_hist_rate.iloc[2], overall)
        np.testing.assert_array_equal(result.won, rows.won)


class RecipeChecks(unittest.TestCase):
    def test_recency_weights_favour_recent_weeks_and_average_one(self):
        weights = study.recency_weights(pd.DataFrame({"dispute_week": [0, 4, 8, 8]}), half_life=4.)
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertAlmostEqual(weights[2] / weights[1], 2.)
        self.assertAlmostEqual(weights[1] / weights[0], 2.)

    def test_evaluation_labels_do_not_change_any_recipe_prediction(self):
        data, audit = synthetic_source()
        changed = data.copy()
        changed.loc[changed.dispute_week.gt(5), "won"] = 1 - changed.loc[changed.dispute_week.gt(5), "won"]
        for specs in [study.RECIPES["lr_merchant_history_recency"][:1], study.RECIPES["lr_recency"][:1]]:
            spec = specs[0]
            a = study.recipe_predict(time_window(data, audit, 5, 9), spec)
            b = study.recipe_predict(time_window(changed, audit, 5, 9), spec)
            np.testing.assert_allclose(a, b, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
