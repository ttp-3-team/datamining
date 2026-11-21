import unittest
from unittest.mock import patch
import numpy as np

from disputes import catboost_study as study
from synthetic import synthetic_source
from disputes.model_comparison import features, predict


class CatBoostChecks(unittest.TestCase):
    def test_fit_predict_with_missing_numeric_and_unseen_categories(self):
        data, audit = synthetic_source()
        a, aa, b, ab = study.previous.prepare_window(data, audit, 5, 9)
        model = study.fit_cat(a, aa, study.SPECS[0])
        b = b.copy()
        b.loc[b.index[0], "merchant_id"] = "UNSEEN_MERCHANT"
        b.loc[b.index[1], "amount_usd"] = np.nan
        p = predict(model, b, ab, study.SPECS[0])
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        changed = b.copy()
        changed.won = 1 - changed.won
        np.testing.assert_allclose(p, predict(model, changed, ab, study.SPECS[0]), atol=1e-12)
        self.assertTrue(model.get_params()["has_time"])
        self.assertEqual(model.get_params()["boosting_type"], "Ordered")

    def test_inner_search_only_trains_on_earlier_periods(self):
        data, audit = synthetic_source()
        training = data.loc[data.dispute_week.le(5)]
        cutoffs = []
        def fake_fit(data, audit, spec):
            cutoff = int(data.dispute_week.max())
            cutoffs.append(cutoff)
            return cutoff
        def fake_predict(cutoff, data, audit, spec):
            self.assertLess(cutoff, int(data.dispute_week.min()))
            self.assertLessEqual(int(data.dispute_week.max()), 5)
            return np.full(len(data), .7)
        with (patch.object(study, "fit_cat", fake_fit), patch.object(study, "predict", fake_predict),
              patch.object(study, "SPECS", study.SPECS[:1])):
            _, _, oof = study.search(training, audit.loc[training.index], study.previous.INNER_FOLDS[5])
        self.assertEqual(cutoffs, [1, 3])
        self.assertTrue((oof.base_train_max_week < oof.dispute_week).all())
        self.assertTrue(oof.dispute_id.is_unique)

    def test_catboost_features_have_no_target_or_identifier(self):
        data, audit = synthetic_source()
        x, cats = features(data, audit, study.SPECS[0])
        self.assertNotIn("won", x)
        self.assertNotIn("dispute_id", x)
        self.assertIn("merchant_id", cats)
        self.assertIn("reason_code", cats)
        self.assertEqual(x.reason_code.iloc[0], "10.4")


if __name__ == "__main__":
    unittest.main()
