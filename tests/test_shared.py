"""Спільні частини: контроль доступу до даних, метрики та групові підсумки."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes.evaluation import calibration_bins, classification_metrics, segment_scores


class AccessGuardChecks(unittest.TestCase):
    def test_guard_blocks_logs_and_is_removed_after_block(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = Path(folder) / "outputs"
            inputs.mkdir()
            for name in ["train_audit.csv", "validation_clean.csv", "test_clean.csv"]:
                (inputs / name).write_text("a\n1\n", encoding="utf-8")
            with proto.train_only_access(folder) as opened:
                pd.read_csv(inputs / "train_audit.csv")
                for name in ["validation_clean.csv", "test_clean.csv"]:
                    with self.assertRaises(PermissionError):
                        pd.read_csv(inputs / name)
            self.assertEqual(opened, ["train_audit.csv"])
            # Після виходу з блоку обмеження не діє — хуки не накопичуються між дослідженнями.
            self.assertEqual(len(pd.read_csv(inputs / "test_clean.csv")), 1)


class MetricChecks(unittest.TestCase):
    def test_confusion_counts_agree_with_rates(self):
        y = np.array([1, 1, 1, 0, 0])
        p = np.array([.9, .8, .2, .7, .1])
        m = classification_metrics(y, p)
        self.assertEqual((m["true_won"], m["false_lost"], m["false_won"], m["true_lost"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["precision_won"], 2 / 3)
        self.assertAlmostEqual(m["recall_lost"], 1 / 2)
        self.assertAlmostEqual(m["accuracy"], 3 / 5)

    def test_segment_without_amounts_has_unknown_income(self):
        data = pd.DataFrame({"dispute_id": list("abc"), "won": [1, 0, 1], "amount_usd": [100., 100., np.nan]})
        audit = pd.DataFrame({"dispute_id": data.dispute_id, "score_eligible": [True, True, False]})
        table = segment_scores(data, audit, np.array([.9, .1, .9]), ["x", "x", "y"]).set_index("segment")
        self.assertEqual(table.loc["x", "score_usd"], 14.)
        self.assertEqual(table.loc["x", "delta_vs_always_usd"], 14. - (14. - 6.))
        self.assertTrue(np.isnan(table.loc["y", "score_usd"]))
        self.assertTrue(np.isnan(table.loc["y", "fight_rate"]))

    def test_calibration_bins_are_ordered(self):
        bins = calibration_bins([0, 1, 1, 0], [.05, .95, .55, .15])
        self.assertEqual(bins.rows.sum(), 4)
        self.assertTrue(bins.bin.iloc[0].startswith("(-0.001"))


if __name__ == "__main__":
    unittest.main()
