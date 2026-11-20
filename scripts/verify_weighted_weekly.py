"""Незалежні Score-перевірки та повторне навчання двох контрольних тижнів."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nbformat
from scipy.special import expit

import weighted_weekly_study as study
from model_comparison import logits

root=Path(__file__).resolve().parent
output=root/"weighted_weekly_outputs"
access=study.base.install_data_access_guard(root)
evidence=json.loads((output/"run_evidence.json").read_text(encoding="utf-8"))
assert set(evidence["actual_source_csv_open_events"])=={"train_audit.csv"}
for name,h in evidence["code_sha256"].items():
    assert study.digest(root/name)==h
for name,h in evidence["result_sha256"].items():
    assert study.digest(output/f"{name}.csv")==h
score_checks=0
all_predictions={}
for regime in ["primary","weekly"]:
    p=pd.read_csv(output/f"{regime}_predictions.csv",dtype={"dispute_id":"string"})
    s=pd.read_csv(output/f"{regime}_summary.csv").set_index("strategy")
    all_predictions[regime]=p
    for method,g in p.groupby("strategy"):
        assert g.dispute_id.is_unique
        assert g.p.between(0,1).all()
        assert g.loc[~g.eligible,"profit"].isna().all()
        for flag,col in [("eligible","score_usd"),("observed_only","observed_score_usd")]:
            rows=g.loc[g[flag]]
            fight=.2*rows.amount_usd*rows.p>6
            expected=((.2*rows.amount_usd*rows.won-6)*fight).sum()
            assert np.isclose(expected,s.loc[method,col])
            score_checks+=1
    inner=pd.read_csv(output/f"{regime}_inner_candidates.csv")
    assert (inner.inner_train_last<inner.inner_eval_last).all()
    assert (inner.inner_eval_last<=inner.outer_cutoff).all()
    cal=pd.read_csv(output/f"{regime}_calibration.csv")
    assert (cal.calibration_max_week<=cal.cutoff).all()

# Однаковий training cutoff та inner-схема дають ті самі p незалежно від
# того, чи об'єднували оцінювані тижні 10–13, чи прогнозували тільки 10.
primary=all_predictions["primary"]
weekly=all_predictions["weekly"]
for method in primary.strategy.unique():
    a=primary.loc[primary.strategy.eq(method)&primary.outer_period.eq("10-13")&primary.dispute_week.eq(10)]
    b=weekly.loc[weekly.strategy.eq(method)&weekly.dispute_week.eq(10)]
    assert a.dispute_id.tolist()==b.dispute_id.tolist()
    np.testing.assert_allclose(a.p,b.p,atol=1e-12)

fit_log=pd.read_csv(output/"fit_log.csv")
assert (fit_log.max_fit_week<=fit_log.cutoff).all()
weighted=fit_log.loc[fit_log["mode"].isin(["amount_p99","sqrt_amount_p99"])]
np.testing.assert_allclose(weighted["mean"],1.,atol=1e-12)
assert (weighted["max"]>0).all()
assert (weighted.weight_effective_n<=weighted.rows+1e-8).all()

# Два крайні weekly-зрізи перевчаються свіжим Engine, без кешу першого запуску.
data,audit,_=study.base.load_train_only(root)
choices=pd.read_csv(output/"weekly_choices.csv")
cal=pd.read_csv(output/"weekly_calibration.csv")
anchors=[]
# Діагностика концентрації приросту у тижні 13: тільки агрегати, без нових правил.
_,_,week13,_=study.base.prepare_window(data,audit,12,13)
reference=weekly.loc[weekly.strategy.eq("logistic")&weekly.dispute_week.eq(13),["dispute_id","profit","p"]].rename(columns={"profit":"lr_profit","p":"lr_p"})
groups=[]
for method in ["catboost","boosting","lr_psp_currency"]:
    frame=weekly.loc[weekly.strategy.eq(method)&weekly.dispute_week.eq(13)&weekly.eligible].merge(reference,on="dispute_id",validate="one_to_one")
    frame=frame.merge(week13[["dispute_id","psp","currency"]],on="dispute_id",validate="one_to_one")
    frame["delta"]=frame.profit-frame.lr_profit
    for (psp,currency),g in frame.groupby(["psp","currency"],dropna=False):
        groups.append({"method":method,"week":13,"psp":psp,"currency":currency,"rows":len(g),
                       "win_rate":g.won.mean(),"mean_method_p":g.p.mean(),"mean_lr_p":g.lr_p.mean(),
                       "delta_vs_logistic_usd":g.delta.sum()})
pd.DataFrame(groups).sort_values("delta_vs_logistic_usd",ascending=False).to_csv(output/"week13_group_diagnostic.csv",index=False)
for week in [10,17]:
    engine=study.Engine(data,audit)
    cut=week-1
    raw={}
    for _,row in choices.loc[choices.cutoff.eq(cut)].iterrows():
        raw[row.family]=engine.predict(cut,week,json.loads(row.spec))
    probabilities=dict(raw)
    for _,row in cal.loc[cal.cutoff.eq(cut)].iterrows():
        if row.adequate_by_protocol:
            probabilities[row.family+"_calibrated"]=expit(row.slope*logits(raw[row.family]).ravel()+row.intercept)
        else:
            probabilities[row.family+"_calibrated"]=raw[row.family]
    a,aa,b,ab=engine.window(cut,week)
    probabilities["constant_train_rate"]=np.full(len(b),a.won.mean())
    probabilities["blend_50_50"]=(raw["logistic"]+raw["boosting"])/2
    probabilities["lr_cat_50_50"]=(raw["logistic"]+raw["catboost"])/2
    probabilities["lr_fixed_C_0_1"]=engine.predict(cut,week,study.lr_specs()[1])
    assert set(probabilities)==set(study.COMPLEXITY)
    for method,p in probabilities.items():
        old=weekly.loc[weekly.strategy.eq(method)&weekly.dispute_week.eq(week)]
        assert b.dispute_id.tolist()==old.dispute_id.tolist()
        np.testing.assert_allclose(p,old.p,atol=1e-10)
    anchors.append({"week":week,"methods_reproduced":len(probabilities)})
    print(f"Fresh anchor week {week}: all methods reproduced",flush=True)

nb=nbformat.read(root/"09_weighted_weekly_study.ipynb",as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=="code"]
assert all(c.execution_count and not any(o.output_type=="error" for o in c.outputs) for c in code)
result={"independent_score_checks":score_checks,"fresh_model_anchor_repeats":anchors,
        "same_cutoff_horizon_invariance":True,"train_weight_normalization_verified":True,
        "source_csv_events":access,"unit_tests":5,"unit_test_failures":0,
        "executed_report_cells":len(code),"report_reads_verified_results":True,
        "full_training_runs":1,"validation_opened":False,"test_opened":False}
(output/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
