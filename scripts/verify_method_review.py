"""Перевірка повноти зведення та всіх відносних оцінок."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
folder=root/"method_review_outputs"
evidence=json.loads((folder/"report_evidence.json").read_text(encoding="utf-8"))
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for source in evidence["sources"]:
    p=Path(source["path"])
    assert not (p.parent==root/"outputs")
    assert not p.name.startswith("test_")
    if p.parent==root/"validation_extension_outputs":
        assert p.name=="validation_summary.csv"
    assert digest(p)==source["sha256"]
for name,expected in evidence["output_sha256"].items():
    assert digest(folder/name)==expected
all_methods=pd.read_csv(folder/"all_methods_summary.csv")
assert not all_methods.duplicated(["evaluation","method_id"]).any()
counts={}
for regime,g in all_methods.groupby("evaluation"):
    assert g.evaluated_rows.nunique()==1
    assert g.always_usd.nunique()==1
    assert g.constant_usd.nunique()==1
    lr=g.loc[g.method_id.eq("logistic")].iloc[0]
    np.testing.assert_allclose(g.uplift_vs_always_percent,100*(g.score_usd-g.always_usd)/g.always_usd)
    np.testing.assert_allclose(g.uplift_vs_constant_percent,100*(g.score_usd-g.constant_usd)/g.constant_usd)
    np.testing.assert_allclose(g.uplift_vs_logistic_percent,100*(g.score_usd-lr.score_usd)/lr.score_usd)
    np.testing.assert_allclose(g.score_index_logistic_100,100*g.score_usd/lr.score_usd)
    np.testing.assert_allclose(g.score_per1000_usd,1000*g.score_usd/g.evaluated_rows)
    np.testing.assert_allclose(g.observed_uplift_vs_logistic_percent,100*(g.observed_score_usd-lr.observed_score_usd)/lr.observed_score_usd)
    counts[regime]={"methods":len(g),"evaluated_rows":int(g.evaluated_rows.iloc[0])}
assert counts=={"nested_train_3":{"methods":17,"evaluated_rows":5141},
               "reused_validation":{"methods":10,"evaluated_rows":1666},
               "weekly_train_8":{"methods":17,"evaluated_rows":3918}}
assert len(pd.read_csv(folder/"C_grid_diagnostics.csv"))==5
assert len(pd.read_csv(folder/"HGB_feature_diagnostics.csv"))==6
overview=pd.read_csv(folder/"all_methods_overview.csv")
assert len(overview)==19
html=(folder/"method_review.html").read_text(encoding="utf-8")
assert '<html lang="uk">' in html and 'Щотижневий train-backtest' in html and 'Висновок' in html
result={"regimes":counts,"overview_methods":len(overview),"fixed_C_diagnostics":5,"HGB_feature_diagnostics":6,
        "all_relative_scores_verified":True,"all_source_hashes_verified":True,
        "new_validation_predictions":False,"test_opened":False}
(folder/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
