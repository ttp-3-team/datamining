"""Незалежні перевірки розрахунків і часових меж модельного експерименту."""
import unittest
import numpy as np
import pandas as pd
from model_comparison import (CANDIDATES, features, policy, profit_rows,
                              temporal_fold, logits, fit_model, predict)
from preparation import SOURCE_COLUMNS, normalize_source


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "dispute_id": ["a", "b", "c", "d"], "amount_usd": [100., 100., 20., np.nan],
            "won": [1, 0, 1, 0], "merchant_id": ["M1", "M2", "M1", None],
            "psp": ["P"] * 4, "payment_method": ["card"] * 4,
            "card_brand": ["visa"] * 4, "reason_category": ["fraud"] * 4,
            "reason_code": ["10.4"] * 4, "currency": ["USD"] * 4,
            "dispute_week": [0, 1, 2, 3], "dispute_weekday": [0, 1, None, 3],
            "amount": [100., 100., 20., np.nan],
        })
        self.audit = pd.DataFrame({"dispute_id": self.data.dispute_id,
                                   "score_eligible": [True, True, True, False],
                                   "score_observed_only": [True, True, True, False]})

    def test_hand_calculated_actions_and_profit(self):
        p = np.array([.9, .1, .99, .8])
        self.assertEqual(policy(self.data, self.audit, p).tolist(), ["fight", "skip", "skip", "review_amount"])
        profit = profit_rows(self.data, self.audit, p)
        np.testing.assert_allclose(profit[:3], [14., 0., 0.])
        self.assertTrue(np.isnan(profit[3]))
        # Боремося й програємо: вартість сплачується навіть за won=0.
        self.assertEqual(profit_rows(self.data, self.audit, np.ones(4))[1], -6.)

    def test_equality_at_break_even_is_skip(self):
        p = np.array([.3, .3, 1., .5])
        self.assertEqual(policy(self.data, self.audit, p)[0], "skip")

    def test_decisions_and_features_ignore_target(self):
        changed = self.data.copy()
        changed.won = 1 - changed.won
        p = np.array([.8, .2, .9, .1])
        np.testing.assert_array_equal(policy(self.data, self.audit, p), policy(changed, self.audit, p))
        pd.testing.assert_frame_equal(features(self.data, self.audit, CANDIDATES[0])[0],
                                      features(changed, self.audit, CANDIDATES[0])[0])

    def test_reject_misaligned_audit(self):
        with self.assertRaises(ValueError):
            policy(self.data, self.audit.iloc[::-1].reset_index(drop=True), np.ones(4))

    def test_reject_bad_probabilities_and_amounts(self):
        for p in [np.array([1.1, .2, .3, .4]), np.array([np.nan] * 4), np.ones(3)]:
            with self.assertRaises(ValueError):
                policy(self.data, self.audit, p)
        invalid = self.data.copy()
        invalid.loc[0, "amount_usd"] = np.nan
        with self.assertRaises(ValueError):
            policy(invalid, self.audit, np.ones(4))

    def test_numeric_codes_are_categorical_and_missing_is_explicit(self):
        x, cats = features(self.data, self.audit, CANDIDATES[0])
        self.assertIn("reason_code", cats)
        self.assertEqual(x.reason_code.iloc[0], "10.4")
        self.assertEqual(x.merchant_id.iloc[3], "MISSING")
        self.assertTrue(pd.isna(x.amount_usd.iloc[3]))

    def test_fold_reference_and_training_ignore_future_values(self):
        raw = pd.concat([self.data] * 20, ignore_index=True)[SOURCE_COLUMNS]
        raw.dispute_id = [str(i) for i in range(len(raw))]
        normalized, audit = normalize_source(raw)
        a, aa, _, _, reference = temporal_fold(normalized, audit, 1, 3)
        changed = normalized.copy()
        changed.loc[changed.dispute_week.gt(1), ["amount_usd", "amount", "won"]] = [9000., 40., 0]
        b, ba, _, _, changed_reference = temporal_fold(changed, audit, 1, 3)
        pd.testing.assert_frame_equal(reference, changed_reference)
        pd.testing.assert_frame_equal(a, b)
        model_a = fit_model(a, aa, CANDIDATES[0])
        model_b = fit_model(b, ba, CANDIDATES[0])
        np.testing.assert_allclose(predict(model_a, a, aa, CANDIDATES[0]),
                                   predict(model_b, a, aa, CANDIDATES[0]), atol=1e-12)

    def test_logit_clipping_is_finite(self):
        self.assertTrue(np.isfinite(logits([0., 1., .5])).all())

    def test_boosting_new_categories_are_not_learned_from_validation(self):
        spec = CANDIDATES[1]
        model = fit_model(self.data, self.audit, spec)
        encoder = model.named_steps["preprocess"].named_transformers_["categories"]
        before = [values.copy() for values in encoder.categories_]
        changed = self.data.copy()
        changed.loc[0, "merchant_id"] = "NEW_MERCHANT"
        p = predict(model, changed, self.audit, spec)
        self.assertTrue(np.isfinite(p).all())
        for old, current in zip(before, encoder.categories_):
            np.testing.assert_array_equal(old, current)
        mask = model.named_steps["classifier"].is_categorical_
        self.assertEqual(mask.tolist(), [True] * 8 + [False])

    def test_all_feature_variants_ignore_target(self):
        audit = self.audit.copy()
        from model_comparison import QUALITY
        for col in QUALITY:
            audit[col] = False
        changed = self.data.copy()
        changed.won = 1 - changed.won
        for spec in CANDIDATES:
            pd.testing.assert_frame_equal(features(self.data, audit, spec)[0],
                                          features(changed, audit, spec)[0])

    def test_group_without_scorable_amount_has_unknown_income(self):
        from model_comparison import segment_diagnostics
        segments = segment_diagnostics(self.data, self.audit, np.array([.9, .1, .99, .8]), self.data)
        row = segments.query("field == 'amount_band' and group == 'MISSING'").iloc[0]
        self.assertEqual(row.evaluated_rows, 0)
        self.assertTrue(pd.isna(row.score_usd))
        self.assertTrue(pd.isna(row.delta_vs_always_usd))


if __name__ == "__main__":
    unittest.main()
