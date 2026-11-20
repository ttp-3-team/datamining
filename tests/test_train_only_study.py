"""Перевірки вкладеної схеми та фіксованого правила вибору."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_only_study as study
from preparation import SOURCE_COLUMNS, normalize_source


def synthetic_source():
    rows = []
    for week in range(18):
        for j in range(16):
            rows.append([f"{week}_{j}", f"M{j % 3}", "P", "card", "visa", "fraud", "10.4",
                         "USD", 40. + j, 40. + j, week, j % 7, int(j % 3 != 0)])
    return normalize_source(pd.DataFrame(rows, columns=SOURCE_COLUMNS))


class NestedChecks(unittest.TestCase):
    def test_protocol_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            output, h1 = study.freeze_protocol(folder)
            _, h2 = study.freeze_protocol(folder)
            self.assertEqual(h1, h2)
            altered = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
            altered["random_seed"] = 0
            (output / "protocol.json").write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                study.freeze_protocol(folder)

    def test_loader_requests_only_train_audit(self):
        data, audit = synthetic_source()
        calls = []
        def loader(root, filename):
            calls.append(filename)
            if filename != "train_audit.csv":
                raise AssertionError("Forbidden table")
            return audit
        with patch.object(study, "load_table", loader):
            normalized, _, access = study.load_train_only(Path("unused"))
        self.assertEqual(calls, ["train_audit.csv"])
        self.assertEqual(access, calls)
        self.assertEqual(normalized.dispute_id.tolist(), data.dispute_id.tolist())

    def test_all_inner_eval_periods_are_inside_outer_train(self):
        for cutoff, end in study.OUTER_FOLDS:
            previous_eval_end = -1
            for a, b in study.INNER_FOLDS[cutoff]:
                self.assertLess(a, b)
                self.assertLessEqual(b, cutoff)
                self.assertGreater(a + 1, previous_eval_end)
                previous_eval_end = b
            self.assertLess(cutoff, end)

    def test_C_grid_reaches_actual_estimator(self):
        data, audit = synthetic_source()
        a, aa, _, _ = study.prepare_window(data, audit, 5, 9)
        for spec in study.LR_SPECS:
            fitted = study.fit(a, aa, spec)
            self.assertEqual(fitted.named_steps["classifier"].C, spec["C"])

    def test_calibration_uses_only_out_of_time_training_labels(self):
        n = 1200
        oof = pd.DataFrame({"dispute_id": [str(i) for i in range(n)],
                            "won": np.tile([0, 1], n // 2), "p": np.tile([.3, .7], n // 2),
                            "inner_period": ["2-3"] * 600 + ["4-5"] * 600,
                            "base_train_max_week": [1] * 600 + [3] * 600,
                            "dispute_week": [2] * 600 + [4] * 600})
        calibrator, status = study.train_calibrator(oof, 5)
        self.assertIsNotNone(calibrator)
        self.assertTrue(status["adequate_by_protocol"])
        with self.assertRaises(ValueError):
            study.train_calibrator(oof, 3)
        bad = oof.copy()
        bad.loc[0, "base_train_max_week"] = 2
        with self.assertRaises(ValueError):
            study.train_calibrator(bad, 5)
        small_calibrator, small_status = study.train_calibrator(oof.iloc[:100], 5)
        self.assertIsNone(small_calibrator)
        self.assertFalse(small_status["adequate_by_protocol"])

    def test_practical_tie_prefers_simpler_but_not_large_income_loss(self):
        summary = pd.DataFrame([
            {"strategy": "logistic", "score_usd": 1000., "complexity": 1},
            {"strategy": "boosting_calibrated", "score_usd": 1020., "complexity": 4},
        ]).assign(selectable=True, evaluated_rows=1000, worst_period_delta_constant_per_row=.1,
                  period_delta_constant_sd=.01, log_loss=.3)
        self.assertEqual(study.select_from_summary(summary)["selected_strategy"], "logistic")
        summary.loc[1, "score_usd"] = 1060.
        self.assertEqual(study.select_from_summary(summary)["selected_strategy"], "boosting_calibrated")
        summary.loc[1, "selectable"] = False
        self.assertEqual(study.select_from_summary(summary)["selected_strategy"], "logistic")

    def test_same_complexity_uses_period_stability(self):
        summary = pd.DataFrame([
            {"strategy": "a", "score_usd": 1020., "worst_period_delta_constant_per_row": .05},
            {"strategy": "b", "score_usd": 1000., "worst_period_delta_constant_per_row": .1},
        ]).assign(selectable=True, evaluated_rows=1000, complexity=1, period_delta_constant_sd=.02, log_loss=.3)
        self.assertEqual(study.select_from_summary(summary)["selected_strategy"], "b")

    def test_outer_eval_targets_do_not_change_choices_or_predictions(self):
        data, audit = synthetic_source()
        changed = data.copy()
        mask = changed.dispute_week.between(6, 9)
        changed.loc[mask, "won"] = 1 - changed.loc[mask, "won"]
        # Маленький фіксований набір прискорює структурну перевірку вкладеності.
        with (patch.object(study, "LR_SPECS", study.LR_SPECS[:2]),
              patch.object(study, "GB_SPECS", study.GB_SPECS[:1]),
              patch.dict(study.PROTOCOL["calibration"], {"minimum_rows": 40, "minimum_each_class": 10})):
            result = study.run_outer(data, audit, 5, 9, progress=lambda *a, **k: None)
            altered = study.run_outer(changed, audit, 5, 9, progress=lambda *a, **k: None)
        # Зменшений поріг стосується лише синтетичного тесту: він перевіряє
        # також реально навчені калібратори, а не лише fallback.
        self.assertTrue(result[2].adequate_by_protocol.all())
        self.assertEqual(result[3], altered[3])
        pd.testing.assert_frame_equal(result[2], altered[2])
        np.testing.assert_allclose(result[0].p, altered[0].p, atol=1e-12)
        np.testing.assert_allclose(result[1].p, altered[1].p, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
