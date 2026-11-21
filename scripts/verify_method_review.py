"""Перевірити зведення методів: джерела, контрольні суми, повноту й усі відносні оцінки."""
import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes.paths import PROJECT
from disputes.verification import write_result

folder = PROJECT / "method_review_outputs"
evidence = proto.read_json(folder / "report_evidence.json")
for source in evidence["sources"]:
    path = PROJECT / source["path"]
    assert path.parent != PROJECT / "outputs" and not path.name.startswith("test_")
    if path.parent == PROJECT / "validation_extension_outputs":
        assert path.name == "validation_summary.csv"
    assert proto.sha256(path) == source["sha256"]
assert all(proto.sha256(folder / name) == digest for name, digest in evidence["output_sha256"].items())

all_methods = pd.read_csv(folder / "all_methods_summary.csv")
assert not all_methods.duplicated(["evaluation", "method_id"]).any()
counts = {}
for regime, g in all_methods.groupby("evaluation"):
    assert g.evaluated_rows.nunique() == g.always_usd.nunique() == g.constant_usd.nunique() == 1
    lr = g.loc[g.method_id.eq("logistic")].iloc[0]
    np.testing.assert_allclose(g.uplift_vs_always_percent, 100 * (g.score_usd - g.always_usd) / g.always_usd)
    np.testing.assert_allclose(g.uplift_vs_constant_percent, 100 * (g.score_usd - g.constant_usd) / g.constant_usd)
    np.testing.assert_allclose(g.uplift_vs_logistic_percent, 100 * (g.score_usd - lr.score_usd) / lr.score_usd)
    np.testing.assert_allclose(g.score_index_logistic_100, 100 * g.score_usd / lr.score_usd)
    np.testing.assert_allclose(g.score_per1000_usd, 1000 * g.score_usd / g.evaluated_rows)
    np.testing.assert_allclose(g.observed_uplift_vs_logistic_percent,
                               100 * (g.observed_score_usd - lr.observed_score_usd) / lr.observed_score_usd)
    counts[regime] = {"methods": len(g), "evaluated_rows": int(g.evaluated_rows.iloc[0])}
assert counts == {"nested_train_3": {"methods": 17, "evaluated_rows": 5141},
                  "reused_validation": {"methods": 10, "evaluated_rows": 1666},
                  "weekly_train_8": {"methods": 17, "evaluated_rows": 3918}}, counts
assert len(pd.read_csv(folder / "C_grid_diagnostics.csv")) == 5
assert len(pd.read_csv(folder / "HGB_feature_diagnostics.csv")) == 6
overview = pd.read_csv(folder / "all_methods_overview.csv")
assert len(overview) == 19
html = (folder / "method_review.html").read_text(encoding="utf-8")
assert '<html lang="uk">' in html and "Щотижневий train-backtest" in html and "Висновок" in html

write_result(folder, {"regimes": counts, "overview_methods": len(overview), "fixed_C_diagnostics": 5,
                      "HGB_feature_diagnostics": 6, "all_relative_scores_verified": True,
                      "all_source_hashes_verified": True, "new_validation_predictions": False, "test_opened": False})
