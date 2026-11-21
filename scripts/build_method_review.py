"""Об'єднати збережені оцінки методів; нових прогнозів validation/test немає."""
from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
from disputes.paths import PROJECT
from disputes.protocol import sha256

ROOT = PROJECT
OUTPUT = ROOT/"method_review_outputs"
LABELS = {
 "constant_train_rate": "Постійна train-ймовірність", "logistic": "Логістична, внутрішній вибір C",
 "logistic_calibrated": "Логістична + калібрування", "boosting": "HistGradientBoosting",
 "boosting_calibrated": "HGB + калібрування", "catboost": "CatBoost",
 "catboost_calibrated": "CatBoost + калібрування", "blend_50_50": "Логістична + HGB 50/50",
 "lr_cat_50_50": "Логістична + CatBoost 50/50", "lr_weight_amount": "ЛР, вага за сумою (p99)",
 "lr_weight_amount_calibrated": "ЛР, вага за сумою + калібрування",
 "lr_weight_sqrt": "ЛР, вага √суми (p99)", "lr_weight_sqrt_calibrated": "ЛР, вага √суми + калібрування",
 "lr_psp_currency": "ЛР + провайдер × валюта", "lr_psp_reason": "ЛР + провайдер × причина",
 "lr_both_interactions": "ЛР + обидві взаємодії", "lr_fixed_C_0_1": "ЛР, фіксований C=0.1 (діагностика)",
 "lr_fixed_C_1": "ЛР, фіксований C=1", "historical_lr_hgb_50_50": "ЛР C=1 + HGB (історичний ансамбль)",
}


def digest(p):
    return sha256(Path(p))


def read_verified(folder, filename, manifest=None):
    path = ROOT/folder/filename
    if manifest is not None:
        key = filename.removesuffix(".csv")
        expected = manifest["result_sha256"][key]
        if digest(path) != expected:
            raise ValueError(f"Змінився {folder}/{filename}")
    return pd.read_csv(path)


def relative(frame, baseline_id="logistic"):
    frame = frame.copy()
    ref = frame.loc[frame.method_id.eq(baseline_id), "score_usd"].iloc[0]
    frame["delta_vs_logistic_usd"] = frame.score_usd-ref
    frame["uplift_vs_logistic_percent"] = 100*(frame.score_usd-ref)/ref
    frame["uplift_vs_always_percent"] = 100*(frame.score_usd-frame.always_usd)/frame.always_usd
    frame["uplift_vs_constant_percent"] = 100*(frame.score_usd-frame.constant_usd)/frame.constant_usd
    frame["score_per1000_usd"] = 1000*frame.score_usd/frame.evaluated_rows
    frame["score_index_logistic_100"] = 100*frame.score_usd/ref
    if "observed_score_usd" in frame:
        obsref=frame.loc[frame.method_id.eq(baseline_id),"observed_score_usd"].iloc[0]
        frame["observed_delta_vs_logistic_usd"]=frame.observed_score_usd-obsref
        frame["observed_uplift_vs_logistic_percent"]=100*(frame.observed_score_usd-obsref)/obsref
    return frame


def build():
    import sys
    import os
    sys.path.insert(0, str(ROOT / "src"))
    # Звіту дозволені тільки збережені оцінки validation, не його початкові рядки.
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix!='.csv':
            return
        if p.parent==(ROOT/'outputs').resolve() or p.name=='dataset.csv' or p.name.startswith('test_'):
            raise PermissionError('Зведення не читає початкові train/validation/test CSV')
        if p.parent==(ROOT/'validation_extension_outputs').resolve() and p.name!='validation_summary.csv':
            raise PermissionError('Дозволені тільки агреговані validation-оцінки')
    sys.addaudithook(guard)
    OUTPUT.mkdir(exist_ok=True)
    sources = []
    def load(folder, filename, manifest=None):
        path=ROOT/folder/filename
        result=read_verified(folder,filename,manifest)
        sources.append({"path": str(path), "sha256": digest(path)})
        return result
    manifests = {folder: json.loads((ROOT/folder/"run_evidence.json").read_text(encoding="utf-8"))
                 for folder in ["catboost_outputs", "interaction_outputs", "weighted_weekly_outputs", "validation_extension_outputs"]}
    core = load("catboost_outputs","strategy_summary.csv",manifests["catboost_outputs"])
    core = core.rename(columns={"strategy":"method_id"})
    interactions = load("interaction_outputs","strategy_summary.csv",manifests["interaction_outputs"])
    interactions = interactions.loc[~interactions.strategy.eq("baseline")].copy()
    interactions["method_id"] = interactions.strategy.map({"psp_currency":"lr_psp_currency", "psp_reason":"lr_psp_reason", "both_interactions":"lr_both_interactions", "fixed_C_0_1":"lr_fixed_C_0_1"})
    weighted = load("weighted_weekly_outputs","primary_summary.csv",manifests["weighted_weekly_outputs"])
    weighted = weighted.loc[weighted.strategy.str.startswith("lr_weight")].rename(columns={"strategy":"method_id"})
    nested = pd.concat([core,interactions,weighted],ignore_index=True)
    if nested.method_id.duplicated().any():
        raise ValueError("Дублікати методів у головній train-таблиці")
    nested["evaluation"]="nested_train_3"
    nested=relative(nested)
    weekly=load("weighted_weekly_outputs","weekly_summary.csv",manifests["weighted_weekly_outputs"]).rename(columns={"strategy":"method_id"})
    weekly["evaluation"]="weekly_train_8"
    weekly=relative(weekly)
    # Читаються тільки вже обчислені агреговані validation-метрики, не початкові дані/прогнози.
    validation=load("validation_extension_outputs","validation_summary.csv",manifests["validation_extension_outputs"])
    mapping={"logistic":"lr_fixed_C_1", "logistic_train_tuned":"logistic", "logistic_train_tuned_calibrated":"logistic_calibrated",
             "lr_tuned_cat_50_50":"lr_cat_50_50", "blend_50_50":"historical_lr_hgb_50_50"}
    validation["method_id"]=validation.model.map(lambda name:mapping.get(name,name))
    validation["evaluation"]="reused_validation"
    validation["always_usd"]=validation.score_usd-validation.delta_always_usd
    validation["constant_usd"]=validation.score_usd-validation.delta_constant_usd
    validation["classification_rows"]=validation.rows
    validation["observed_rows"]=validation.observed_evaluated_rows
    validation=relative(validation)
    all_methods=pd.concat([nested,weekly,validation],ignore_index=True)
    all_methods["method"]=all_methods.method_id.map(LABELS).fillna(all_methods.method_id)
    all_methods["evidence_status"]=np.where(all_methods.evaluation.eq("reused_validation"),
          "already reused validation; historical aggregates only", "reused train; no independent final test")
    cols=["evaluation","method_id","method","classification_rows","evaluated_rows","score_usd","always_usd","constant_usd",
          "delta_vs_logistic_usd","uplift_vs_always_percent","uplift_vs_constant_percent","uplift_vs_logistic_percent",
          "score_index_logistic_100","score_per1000_usd","log_loss","brier","roc_auc","observed_rows","observed_score_usd",
          "observed_delta_vs_logistic_usd","observed_uplift_vs_logistic_percent",
          "periods","periods_above_always","periods_above_logistic","worst_delta_vs_logistic_per1000","evidence_status"]
    for col in cols:
        if col not in all_methods:
            all_methods[col]=np.nan
    all_methods[cols].to_csv(OUTPUT/"all_methods_summary.csv",index=False)
    predictions=load("weighted_weekly_outputs","weekly_predictions.csv",manifests["weighted_weekly_outputs"])
    ref=predictions.loc[predictions.strategy.eq("logistic"),["dispute_id","profit"]].rename(columns={"profit":"lr_profit"})
    concentration=[]
    for method,part in predictions.groupby("strategy"):
        g=part.merge(ref,on="dispute_id",validate="one_to_one")
        g["delta"]=g.profit-g.lr_profit
        positive=g.loc[g.delta.gt(0)].sort_values("delta",ascending=False)
        week_delta=g.groupby("dispute_week").delta.sum()
        net=float(g.delta.sum())
        best_week=int(week_delta.idxmax())
        best_gain=float(week_delta.max())
        concentration.append({"method_id":method,"method":LABELS.get(method,method),"net_delta_usd":net,
            "positive_cases":len(positive),"negative_cases":int(g.delta.lt(0).sum()),
            "largest_positive_case_usd":float(positive.delta.iloc[0]) if len(positive) else 0.,
            "top3_positive_cases_usd":float(positive.delta.head(3).sum()),"best_week":best_week,
            "best_week_delta_usd":best_gain,"best_week_share_net_percent":100*best_gain/net if net>0 else np.nan,
            "net_without_best_week_usd":net-best_gain})
    concentration=pd.DataFrame(concentration)
    concentration.to_csv(OUTPUT/"weekly_gain_concentration.csv",index=False)
    for name, frame in [("nested_train",nested),("weekly_train",weekly),("validation",validation)]:
        frame.to_csv(OUTPUT/f"{name}_table.csv",index=False)

    # Параметри та ранні feature-ablation спроби — окремі діагностики, не нові незалежні моделі.
    cgrid=load("train_only_outputs","C_grid_summary.csv")
    cgrid["method_id"]=cgrid.C.map(lambda c:f"lr_fixed_C_{c:g}")
    cgrid["method"]=cgrid.C.map(lambda c:f"ЛР, C={c:g}")
    cgrid["evaluation"]="fixed_C_train_diagnostic"
    cgrid=relative(cgrid, "lr_fixed_C_1")
    cgrid.to_csv(OUTPUT/"C_grid_diagnostics.csv",index=False)
    trials=load("comparison_outputs","cv_scores.csv")
    trial_rows=[]
    for candidate,g in trials.groupby("candidate"):
        if not candidate.startswith("hgb_"):
            continue
        eligible=g.loc[g.scenario.eq("eligible")]
        observed=g.loc[g.scenario.eq("observed_only")]
        oof=load("comparison_outputs",f"oof_{candidate}.csv")
        from disputes.train_only_study import probability_metrics
        metrics=probability_metrics(oof.won,oof.p)
        trial_rows.append({"candidate":candidate,"score_usd":eligible.score_usd.sum(),
                          "evaluated_rows":int(eligible.rows.sum()),"observed_score_usd":observed.score_usd.sum(),
                          "always_usd":(eligible.score_usd-eligible.delta_vs_always_usd).sum(), **metrics,
                          "status":"fixed candidate on reused train folds; feature-search diagnostic"})
    hgb_diagnostics=pd.DataFrame(trial_rows)
    hgb_diagnostics["uplift_vs_always_percent"]=100*(hgb_diagnostics.score_usd-hgb_diagnostics.always_usd)/hgb_diagnostics.always_usd
    hgb_diagnostics.to_csv(OUTPUT/"HGB_feature_diagnostics.csv",index=False)

    # Огляд: порожні клітинки означають, що метод у відповідному режимі не оцінювали.
    overview=[]
    for method in dict.fromkeys(all_methods.method_id):
        item={"Метод":LABELS.get(method,method), "method_id":method}
        for key,prefix in [("nested_train_3","Train: 3 зрізи"),("weekly_train_8","Train: 8 тижнів"),("reused_validation","Validation")]:
            row=all_methods.loc[all_methods.method_id.eq(method)&all_methods.evaluation.eq(key)]
            if len(row):
                row=row.iloc[0]
                item[prefix+": Score, USD"]=row.score_usd
                item[prefix+": до завжди, %"]=row.uplift_vs_always_percent
                item[prefix+": до ЛР, %"]=row.uplift_vs_logistic_percent
                if key=="weekly_train_8":
                    item["Тижнів краще ЛР / 8"]=row.periods_above_logistic
        overview.append(item)
    overview=pd.DataFrame(overview)
    overview.to_csv(OUTPUT/"all_methods_overview.csv",index=False)
    weights=all_methods.loc[all_methods.method_id.str.startswith("lr_weight")]
    conclusions={"primary_weighted_best":weights.loc[weights.evaluation.eq("nested_train_3")].sort_values("score_usd",ascending=False).iloc[0][["method_id","score_usd","delta_vs_logistic_usd","uplift_vs_logistic_percent"]].to_dict(),
                 "weekly_best":weekly.sort_values("score_usd",ascending=False).iloc[0][["method_id","score_usd","delta_vs_logistic_usd","periods_above_logistic"]].to_dict(),
                 "primary_selection":json.loads((ROOT/"weighted_weekly_outputs"/"primary_selection.json").read_text()),
                 "weekly_selection":json.loads((ROOT/"weighted_weekly_outputs"/"weekly_selection.json").read_text()),
                 "no_test_evaluation":True}
    (OUTPUT/"conclusions.json").write_text(json.dumps(conclusions,indent=2,ensure_ascii=False),encoding="utf-8")
    best_weight=conclusions["primary_weighted_best"]
    weekly_best=conclusions["weekly_best"]
    chosen_primary=conclusions["primary_selection"]["selected_strategy"]
    chosen_weekly=conclusions["weekly_selection"]["selected_strategy"]
    if chosen_primary==chosen_weekly=="logistic":
        recommendation="За незмінним практичним правилом обидві нові схеми залишають звичайну логістичну регресію. Додаткові ваги не дали достатнього стабільного приросту для ускладнення рішення."
    elif chosen_primary==chosen_weekly:
        recommendation=f"Обидві схеми за практичним правилом обрали {LABELS.get(chosen_primary,chosen_primary)}. Це кандидат для фіксації рецепта й подальшої зовнішньої перевірки; майбутню перевагу не доведено."
    else:
        recommendation="Вибір за трьома зрізами й щотижневою схемою різниться. Узгодженого переможця не отримано; великий сумарний Score одного режиму не є достатньою підставою замінити просту базову модель."
    conclusions["recommendation"]=recommendation
    conclusions["weekly_concentration"]=concentration.loc[concentration.method_id.isin(["catboost","boosting","lr_psp_currency"])].to_dict(orient="records")
    (OUTPUT/"conclusions.json").write_text(json.dumps(conclusions,indent=2,ensure_ascii=False),encoding="utf-8")

    def table(frame,columns):
        existing=[c for c in columns if c in frame]
        names={"method":"Метод","score_usd":"Score, USD","evaluated_rows":"Спорів для Score",
               "uplift_vs_always_percent":"До «завжди», %","uplift_vs_constant_percent":"До постійної p, %",
               "uplift_vs_logistic_percent":"До ЛР, %","score_per1000_usd":"Score / 1000, USD",
               "observed_score_usd":"Без змінених USD-сум","periods_above_always":"Тижнів > «завжди»",
               "observed_uplift_vs_logistic_percent":"Без змінених сум: до ЛР, %",
               "periods_above_logistic":"Тижнів > ЛР","worst_delta_vs_logistic_per1000":"Найгірший Δ до ЛР / 1000",
               "sd_delta_vs_logistic_per1000":"Розкид Δ до ЛР / 1000"}
        return frame[existing].rename(columns=names).to_html(index=False,na_rep="—",float_format=lambda x:f"{x:,.3f}",border=0,escape=True)
    metric_cols=["method","score_usd","evaluated_rows","uplift_vs_always_percent","uplift_vs_constant_percent","uplift_vs_logistic_percent","score_per1000_usd","log_loss","brier","roc_auc","observed_score_usd","observed_uplift_vs_logistic_percent"]
    for f in [nested,weekly,validation]:
        f["method"]=f.method_id.map(LABELS).fillna(f.method_id)
    source_note="<ul>"+"".join(f"<li>{html.escape(str(Path(s['path']).relative_to(ROOT)))} — {s['sha256'][:16]}…</li>" for s in sources)+"</ul>"
    group_path=ROOT/"weighted_weekly_outputs"/"week13_group_diagnostic.csv"
    group_html=""
    if group_path.exists():
        group_diagnostic=load("weighted_weekly_outputs","week13_group_diagnostic.csv")
        group_html=table(group_diagnostic.loc[group_diagnostic.delta_vs_logistic_usd.abs().gt(1)],
             ["method","psp","currency","rows","win_rate","mean_method_p","mean_lr_p","delta_vs_logistic_usd"])
    source_note="<ul>"+"".join(f"<li>{html.escape(str(Path(s['path']).relative_to(ROOT)))} — {s['sha256'][:16]}…</li>" for s in sources)+"</ul>"
    import base64
    figure=OUTPUT/"weekly_delta_by_strategy.png"
    image_html='<section><h2>Графік щотижневої стійкості</h2><img style="max-width:100%;height:auto" alt="Різниця доходу методів і щотижнева стійкість" src="data:image/png;base64,'+base64.b64encode(figure.read_bytes()).decode()+'"></section>' if figure.exists() else ''
    body=f'''<!doctype html><html lang="uk"><meta charset="utf-8"><title>Порівняння методів спорів</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;margin:35px;color:#183044;background:#f7f9fc}}h1,h2{{color:#163a51}}section{{background:white;padding:22px;margin:20px 0;border-radius:12px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;width:100%}}th,td{{padding:8px 12px;border-bottom:1px solid #dde4ec;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eaf0f7}}.note{{background:#fff4d7;padding:15px;border-radius:8px}}small{{color:#536779}}a{{color:#216490}}</style>
<h1>Усі перевірені методи: дохід, відносні оцінки й стійкість</h1>
<p>Нові експерименти: зважене навчання та 8 щотижневих зрізів тільки train. Test не відкривався.</p>
<p class="note">Порівнюйте моделі всередині одного режиму. Суми train і validation стосуються різних спорів; більший абсолютний Score між режимами не означає кращої моделі. Щотижневі зрізи повторно використовують тижні 10–17: це більше часових перевірок, але не нові незалежні дані.</p>
<section><h2>Висновок</h2><p>{html.escape(recommendation)}</p><p>Найбільший дохід серед нових зважених варіантів на 3 зрізах: {html.escape(LABELS.get(best_weight['method_id'],best_weight['method_id']))}, {best_weight['score_usd']:,.2f} USD; різниця з логістичною {best_weight['delta_vs_logistic_usd']:+,.2f} USD ({best_weight['uplift_vs_logistic_percent']:+.3f}%).</p><p>Номінальний максимум на щотижневій схемі: {html.escape(LABELS.get(weekly_best['method_id'],weekly_best['method_id']))}, {weekly_best['score_usd']:,.2f} USD; різниця з ЛР {weekly_best['delta_vs_logistic_usd']:+,.2f} USD. Номінальний максимум і вибір за практичним порогом можуть різнитися.</p></section>
<section><h2>Загальний огляд</h2><p>Score — USD. Відносні оцінки — відсотки приросту до «боротися завжди» та до логістичної регресії в тому самому режимі. «—» = не оцінювали.</p><div class="scroll">{table(overview,[c for c in overview if c!='method_id'])}</div></section>
<section><h2>Вкладена train-перевірка: 3 зовнішні періоди</h2><p>6–9, 10–13, 14–17; {int(nested.evaluated_rows.iloc[0])} придатних спорів. Для основних сімей параметри визначені всередині минулої частини. Фіксований C=0.1 — діагностика після попереднього підбору.</p><div class="scroll">{table(nested.sort_values('score_usd',ascending=False),metric_cols)}</div></section>
<section><h2>Щотижневий train-backtest: 8 періодів</h2><p>Кожен тиждень 10–17 прогнозується за попередніми тижнями; гіперпараметри й калібратори визначаються всередині відповідної навчальної частини. {int(weekly.evaluated_rows.iloc[0])} придатних спорів.</p><div class="scroll">{table(weekly.sort_values('score_usd',ascending=False),metric_cols+['periods_above_always','periods_above_logistic','worst_delta_vs_logistic_per1000','sd_delta_vs_logistic_per1000'])}</div></section>
<section><h2>Чи не створив приріст один тиждень або кілька великих спорів?</h2><p>Показано декомпозицію різниці з ЛР, без зміни рішень чи виключення рядків з основної оцінки. Частка одного тижня від чистого приросту може бути великою навіть тоді, коли цей тиждень містить багато звичайних спорів. Діагностика «без найкращого тижня» не є новим критерієм підбору.</p><div class="scroll">{table(concentration.loc[concentration.method_id.isin(['catboost','boosting','lr_psp_currency','lr_weight_amount','lr_weight_sqrt'])],['method','net_delta_usd','positive_cases','negative_cases','largest_positive_case_usd','top3_positive_cases_usd','best_week','best_week_delta_usd','best_week_share_net_percent','net_without_best_week_usd'])}</div><p>CatBoost: 261.60 із 320.00 USD приросту припадає на тиждень 13 (81.75%). PSP×currency: 212.00 із 224.80 USD (94.31%). Найбільший окремий позитивний внесок цих методів — 17.20 USD: це концентрація в часовому сегменті, а не один надвеликий платіж.</p><div class="scroll">{group_html}</div><p>Групові закономірності є описовими: за цими відомими результатами не створювали нових правил маршрутизації моделей.</p></section>
<section><h2>Раніше використаний validation</h2><p>Лише вже збережені агреговані результати; нових validation-прогнозів немає. Ці дані раніше використовували для вибору. Історичний ансамбль ЛР C=1 + HGB відокремлено від ансамблів із налаштованою ЛР.</p><div class="scroll">{table(validation.sort_values('score_usd',ascending=False),metric_cols)}</div></section>
<section><h2>Діагностика C та ознак HGB</h2><p>Це спроби на вже знайомих train-періодах. Вони не є незалежним підтвердженням вибору глобального параметра.</p><div class="scroll">{table(cgrid,['method','score_usd','uplift_vs_always_percent','log_loss','brier','roc_auc','observed_score_usd'])}</div><div class="scroll">{table(hgb_diagnostics,['candidate','score_usd','uplift_vs_always_percent','log_loss','brier','roc_auc','observed_score_usd'])}</div></section>
<section><h2>Що саме перевірено</h2><p>Вага за сумою та √суми: cap p99 тільки з train, пропуски мають вагу медіанного train-спору, середня вага 1. Ваги впливають лише на навчання класифікатора. Score рахується за початковою грошовою формулою; у Score суми не імпутуються. Калібратори використовують тільки внутрішні out-of-time прогнози.</p><p>Log loss і Brier — менше краще; ROC AUC — більше краще. Вони оцінені на всіх мічених спорах. Score та observed Score мають окремі кількості рядків. Поріг практичної користі лишається 50 USD на 1000 оцінених спорів; статистичний переможець не вимагається.</p><p class="note">Дата отримання результату спору відсутня. Щотижневе оновлення припускає, що попередні результати вже доступні; перевірити це неможливо. Спільні навчальні вікна й повторний аналіз train обмежують висновки про майбутню стійкість.</p></section>
{image_html}
<section><h2>Джерела чисел</h2><small>{source_note}</small></section>
<script>document.querySelectorAll('table').forEach(table=>{{table.querySelectorAll('thead th').forEach((th,col)=>{{th.style.cursor='pointer';th.title='Натисніть, щоб сортувати всередині цієї таблиці';th.onclick=()=>{{const body=table.tBodies[0];const rows=[...body.rows];const dir=th.dataset.direction==='desc'?'asc':'desc';th.dataset.direction=dir;rows.sort((a,b)=>{{const av=a.cells[col].textContent.trim(),bv=b.cells[col].textContent.trim();if(av==='—')return bv==='—'?0:1;if(bv==='—')return -1;const an=Number(av.replaceAll(',','')),bn=Number(bv.replaceAll(',',''));const c=Number.isNaN(an)||Number.isNaN(bn)?av.localeCompare(bv,'uk'):an-bn;return dir==='asc'?c:-c;}});rows.forEach(row=>body.appendChild(row));}};}});}});</script></html>'''
    (OUTPUT/"method_review.html").write_text(body,encoding="utf-8")
    (OUTPUT/"report_evidence.json").write_text(json.dumps({"sources":sources,"test_opened":False,"new_validation_predictions":False,
         "output_sha256":{p.name:digest(p) for p in OUTPUT.glob('*') if p.name!='report_evidence.json'}},indent=2),encoding="utf-8")
    print(OUTPUT/"method_review.html")
    return all_methods,overview,conclusions


if __name__=="__main__":
    build()
