import unittest

import numpy as np
import pandas as pd

from disputes.validation_extension import pair_details, pair_summary


class DisagreementChecks(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"dispute_id": ["a", "b", "c", "d"], "dispute_week": [18]*4,
            "won": [1, 0, 1, 0], "amount_usd": [100., 100., 20., np.nan],
            "psp": ["P"]*4, "reason_category": ["fraud"]*4, "payment_method": ["card"]*4})
        self.audit = pd.DataFrame({"dispute_id": self.data.dispute_id,
            "score_eligible": [True, True, True, False], "score_observed_only": [True, False, True, False]})
        self.a = np.array([.8, .2, .9, .9])
        self.b = np.array([.2, .8, .1, .1])

    def test_different_classes_are_not_always_different_actions(self):
        details = pair_details(self.data, self.audit, self.a, self.b, "a", "b")
        summary = pair_summary(details, "score_eligible")
        self.assertEqual(summary["class_disagreements_at_0_5"], 4)
        self.assertEqual(summary["action_disagreements"], 2)
        self.assertEqual(summary["delta_b_minus_a_usd"], -20.)
        self.assertEqual(summary["only_a_correct"], 3)
        self.assertEqual(summary["only_b_correct"], 1)
        self.assertTrue(pd.isna(details.profit_a.iloc[3]))
        self.assertEqual(details.decision_group.iloc[2], "both_skip")

    def test_scenario_masks_change_comparison_without_imputing_income(self):
        details = pair_details(self.data, self.audit, self.a, self.b, "a", "b")
        summary = pair_summary(details, "score_observed_only")
        self.assertEqual(summary["evaluated_rows"], 2)
        self.assertEqual(summary["action_disagreements"], 1)
        self.assertEqual(summary["delta_b_minus_a_usd"], -14.)

    def test_probability_changes_can_leave_all_actions_and_profits_equal(self):
        b = np.array([.7, .25, .8, .8])
        details = pair_details(self.data, self.audit, self.a, b, "a", "b")
        summary = pair_summary(details, "score_eligible")
        self.assertGreater(summary["mean_abs_probability_gap"], 0)
        self.assertEqual(summary["action_disagreements"], 0)
        self.assertEqual(summary["delta_b_minus_a_usd"], 0)

    def test_decisions_never_select_model_using_actual_outcome(self):
        original = pair_details(self.data, self.audit, self.a, self.b, "a", "b")
        changed = self.data.copy()
        changed.won = 1 - changed.won
        flipped = pair_details(changed, self.audit, self.a, self.b, "a", "b")
        pd.testing.assert_series_equal(original.action_a, flipped.action_a)
        pd.testing.assert_series_equal(original.action_b, flipped.action_b)
        pd.testing.assert_series_equal(original.action_disagreement, flipped.action_disagreement)


if __name__ == "__main__":
    unittest.main()
