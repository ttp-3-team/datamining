import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

import interaction_study as study
from test_train_only_study import synthetic_source


class InteractionChecks(unittest.TestCase):
    def test_pairs_keep_base_features_and_ignore_target(self):
        data, audit = synthetic_source()
        changed = data.copy()
        changed.won = 1 - changed.won
        for variant in study.VARIANTS:
            original, cats = study.make_features(data, audit, variant)
            altered, _ = study.make_features(changed, audit, variant)
            pd.testing.assert_frame_equal(original, altered)
            self.assertNotIn("won", original)
            self.assertNotIn("dispute_id", original)
            self.assertIn("psp", cats)
            self.assertIn("currency", cats)
            self.assertEqual(len(cats), 8 + len(study.VARIANTS[variant]))

    def test_pair_encoding_has_no_separator_collisions(self):
        data, audit = synthetic_source()
        data.loc[0, ["psp", "currency"]] = ["P|X", "Y"]
        data.loc[1, ["psp", "currency"]] = ["P", "X|Y"]
        data.loc[2, "currency"] = pd.NA
        x, _ = study.make_features(data, audit, "psp_currency")
        self.assertNotEqual(x.interaction_psp_currency.iloc[0], x.interaction_psp_currency.iloc[1])
        self.assertEqual(json.loads(x.interaction_psp_currency.iloc[2])[1], "MISSING")

    def test_unseen_pair_does_not_change_train_encoder(self):
        data, audit = synthetic_source()
        a, aa, b, ab = study.previous.prepare_window(data, audit, 5, 9)
        mask = np.arange(len(a)) % 2 == 0
        a.loc[mask, ["psp", "currency"]] = ["Q", "EUR"]
        model = study.fit(a, aa, "psp_currency", .1)
        encoder = model.named_steps["preprocess"].named_transformers_["categories"]
        categories = [v.copy() for v in encoder.categories_]
        b.loc[b.index[0], ["psp", "currency"]] = ["P", "EUR"]
        p = study.predict(model, b, ab, "psp_currency")
        self.assertTrue(np.isfinite(p).all())
        x, _ = study.make_features(b, ab, "psp_currency")
        self.assertNotIn(x.interaction_psp_currency.iloc[0], encoder.categories_[-1])
        for before, after in zip(categories, encoder.categories_):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(model.named_steps["classifier"].C, .1)

    def test_outer_labels_do_not_change_inner_parameters_or_predictions(self):
        data, audit = synthetic_source()
        changed = data.copy()
        mask = changed.dispute_week.between(6, 9)
        changed.loc[mask, "won"] = 1 - changed.loc[mask, "won"]
        with (patch.object(study, "VARIANTS", {"baseline": [], "psp_currency": ["psp_currency"]}),
              patch.object(study.previous, "C_VALUES", [.1, 1.])):
            original = study.run_outer(data, audit, 5, 9)
            altered = study.run_outer(changed, audit, 5, 9)
        pd.testing.assert_frame_equal(original[1], altered[1])
        np.testing.assert_allclose(original[0].p, altered[0].p, atol=1e-12)

    def test_freeze_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            output, before = study.freeze(folder)
            _, after = study.freeze(folder)
            self.assertEqual(before, after)
            value = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
            value["version"] = "changed"
            (output / "protocol.json").write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                study.freeze(folder)


if __name__ == "__main__":
    unittest.main()
