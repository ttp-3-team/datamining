import unittest
import numpy as np
import pandas as pd
import weighted_weekly_study as study
from test_train_only_study import synthetic_source


class WeightAndTimeChecks(unittest.TestCase):
    def test_weights_are_finite_normalized_and_target_independent(self):
        data = pd.DataFrame({"amount_usd": [10., 30., 100., 10000., np.nan], "won": [0, 0, 1, 1, 0]})
        changed = data.assign(won=1-data.won)
        for mode in study.WEIGHT_MODES:
            w, stats = study.sample_weights(data, mode)
            other, _ = study.sample_weights(changed, mode)
            np.testing.assert_allclose(w, other)
            self.assertTrue(np.isfinite(w).all() and (w > 0).all())
            self.assertAlmostEqual(w.mean(), 1.)
            self.assertLessEqual(stats["weight_effective_n"], len(data)+1e-9)

    def test_missing_weight_is_median_case_and_cap_is_train_only(self):
        data = pd.DataFrame({"amount_usd": [10., 20., 30., np.nan]})
        weights, stats = study.sample_weights(data, "amount_p99")
        self.assertAlmostEqual(weights[1], weights[3])
        self.assertEqual(stats["median_amount"], 20.)
        self.assertAlmostEqual(stats["cap_amount"], 29.8)
        sqrt_weights, _ = study.sample_weights(data, "sqrt_amount_p99")
        self.assertLess(sqrt_weights.max()/sqrt_weights.min(), weights.max()/weights.min())

    def test_weekly_inner_periods_end_before_evaluation(self):
        self.assertEqual(study.WEEKLY_WEEKS, list(range(10,18)))
        for week in study.WEEKLY_WEEKS:
            cut = week-1
            previous_end = -1
            for train_end, eval_end in study.WEEKLY_INNER[cut]:
                self.assertLess(train_end, eval_end)
                self.assertLessEqual(eval_end, cut)
                self.assertGreater(train_end+1, previous_end)
                previous_end = eval_end

    def test_future_labels_do_not_change_weighted_model(self):
        data, audit = synthetic_source()
        changed = data.copy()
        changed.loc[changed.dispute_week.gt(5), "won"] = 1-changed.loc[changed.dispute_week.gt(5), "won"]
        spec = study.lr_specs("amount_p99")[1]
        a, b = study.Engine(data,audit), study.Engine(changed,audit)
        pa, pb = a.predict(5,9,spec), b.predict(5,9,spec)
        np.testing.assert_allclose(pa,pb,atol=1e-12)
        self.assertEqual(a.fit_log,b.fit_log)
        self.assertEqual(len(a.models),1)
        a.predict(5,8,spec)
        self.assertEqual(len(a.models),1)
        a.predict(3,5,spec)
        self.assertEqual(len(a.models),2)

    def test_evaluation_amounts_do_not_change_fit_weights_or_coefficients(self):
        data, audit = synthetic_source()
        changed = data.copy()
        changed.loc[changed.dispute_week.gt(5), "amount_usd"] = 9000.
        spec = study.lr_specs("amount_p99")[1]
        a,b = study.Engine(data,audit),study.Engine(changed,audit)
        a.predict(5,9,spec); b.predict(5,9,spec)
        self.assertEqual(a.fit_log,b.fit_log)
        ma, mb = next(iter(a.models.values())),next(iter(b.models.values()))
        np.testing.assert_allclose(ma.named_steps["classifier"].coef_,mb.named_steps["classifier"].coef_)


if __name__ == "__main__":
    unittest.main()
