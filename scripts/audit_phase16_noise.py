"""Reproduce noise-region diagnostics with own certified predictions, without fitting."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import FEATURES, arr_sha256, file_sha256, save_json
from scripts.phase16_common import bank, noise_ceiling, effective_flip_rate, auc_se, error_pairs, SCOPE
from src.features.s6e10 import load_original, original_leak_audit
from src.features.view import RAW21


def wilson(k, n):
    if not n: return None
    z=1.959963984540054; ph=k/n; den=1+z*z/n
    c=(ph+z*z/(2*n))/den; d=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return [float(c-d),float(c+d)]


def region(y, p, m):
    n=int(m.sum()); k=int(y[m].sum())
    result={'rows':n,'satisfied':k,'positive_rate':k/n if n else None,
            'positive_rate_wilson95':wilson(k,n)}
    if 0<k<n:
        se=auc_se(y[m],p[m]); auc=float(roc_auc_score(y[m],p[m]))
        result.update(auc=auc,conditional_auc_se=se,normal_auc95=[auc-1.96*se,auc+1.96*se])
    return result


def main():
    start=time.monotonic();out=Path('reports/phase16_noise_reproduction_20261009.json')
    if out.exists():raise FileExistsError('Preserve prior audit')
    tr,te,y,folds,vectors,tests,proof=bank()
    names=json.loads((FEATURES/'static_full.json').read_text());ix=names.index('teach_lgbm')
    with np.load(FEATURES/'static_full.npz') as cache:
        raw=cache['tr'];assert raw.shape==(len(y),len(names));q=raw[:,ix].copy();del raw
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    og=load_original();overlap=original_leak_audit(og,__import__('pandas').concat([tr[RAW21],te[RAW21]]),RAW21)
    lo,hi=q<.001,q>=.999;sure=lo|hi;discord=(lo&(y==1))|(hi&(y==0));regions={}
    for name,p in vectors.items():
        errors,pairs=error_pairs(y,p);outside,outpairs=error_pairs(y,p,~discord)
        assert abs(1-errors/pairs-roc_auc_score(y,p))<1e-14
        flat=p.copy();flat[lo]=p[lo].mean();flat[hi]=p[hi].mean()
        regions[name]={'full_auc':float(roc_auc_score(y,p)),'full_conditional_auc_se':auc_se(y,p),
            'near_certain_original_low':region(y,p,lo),'near_certain_original_high':region(y,p,hi),
            'other':region(y,p,~sure),'errors':errors,'total_pairs':pairs,
            'pairs_involving_discordance_fraction':1-outpairs/pairs,
            'errors_involving_discordance':errors-outside,
            'error_fraction_involving_discordance':(errors-outside)/errors,
            'flattened_auc':float(roc_auc_score(y,flat)),
            'flattening_loss':float(roc_auc_score(y,p)-roc_auc_score(y,flat)),
            'perfold_region_auc':[{ 'fold':k,'low':region(y,p,lo&(folds==k)),
                                   'high':region(y,p,hi&(folds==k))} for k in range(5)]}
    groups=[]
    for col in ['Class','Type of Travel','Customer Type','Gender']:
        for value in sorted(tr[col].unique()):
            m=(tr[col]==value).to_numpy()
            groups.append({'column':col,'value':str(value),'low':region(y,vectors['candidate'],lo&m),
                           'high':region(y,vectors['candidate'],hi&m)})
    qobs=float(y.mean());scope=json.loads(SCOPE.read_text())
    gain=regions['candidate']['full_auc']-regions['aux10']['full_auc']
    ds=(regions['aux10']['errors_involving_discordance']-regions['candidate']['errors_involving_discordance'])/regions['candidate']['total_pairs']
    result={'utc':datetime.now(timezone.utc).isoformat(),'scope_sha256':file_sha256(SCOPE),
            'notebook_source_sha256':file_sha256('research/raw/phase16_noise/s6e10-the-gap-is-3-8-label-noise-not-features.ipynb'),
            'bank':proof,'teacher_probability_sha256':arr_sha256(q),'teacher_cache_sha256':file_sha256(FEATURES/'static_full.npz'),
            'teacher_builder_sha256':file_sha256('src/features/s6e10.py'),'teacher_view_sha256':file_sha256('src/features/view.py'),
            'teacher_recipe':scope['noise_audit']['existing_source_teacher'],
            'teacher_limitations':'Existing original-only5-model recipe differs from author single63leaf600round config; source-domain confidence is a proxy, not latent truth. Original OOF/flip-curve fits are not stored and not reproduced.',
            'original_data_sha256':file_sha256('data/original/arseniyshutko__binary-aviation-satisfaction-129k/data.csv'),
            'original_positive_rate':float(og.satisfaction.mean()),'original_overlap_audit':overlap,
            'observed_positive_rate':qobs,'original_teacher_auc_on_competition':float(roc_auc_score(y,q)),
            'near_certain_rows':int(sure.sum()),'discordant_rows':int(discord.sum()),'discordant_row_fraction':float(discord.mean()),
            'models':regions,'segments':groups,'symmetric_flip_sensitivity':[
                {'assumed_flip_rate':e,'conditional_perfect_clean_ranker_auc_ceiling':noise_ceiling(qobs,e)}
                for e in scope['noise_audit']['symmetric_flip_rates']],
            'equivalent_flip_rate_for_v6_ceiling':effective_flip_rate(qobs,regions['candidate']['full_auc']),
            'equivalent_flip_rate_for_stretch_ceiling':effective_flip_rate(qobs,.9621),
            'distance_v6_to_stretch':.9621-regions['candidate']['full_auc'],
            'v6_gain_vs_aux10':gain,'v6_gain_on_pairs_involving_original_discordance':ds,
            'fraction_of_gain_on_those_pairs':ds/gain,
            'uncertainty':'Wilson and DeLong intervals condition on fixed models/regions and independent-row approximation; no refitting/domain-shift or flip-rate identification uncertainty included. Derived ceilings are sensitivity scenarios, not estimated competition bounds.',
            'source_sha256':{p:file_sha256(p) for p in ['scripts/audit_phase16_noise.py','scripts/phase16_common.py']},
            'verdict':'SOURCE_MODEL_DISAGREEMENT_NOT_IDENTIFIED_IRREDUCIBLE_NOISE','seconds':time.monotonic()-start}
    save_json(result,out)
    with Path('experiments/ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'exp_id':'phase16_noise_audit','ts':result['utc'],'kind':'NOISE_DIAGNOSTICS_NO_FITTING',
                           'report':str(out),'report_sha256':file_sha256(out),'verdict':result['verdict']})+'\n')
    print(json.dumps({k:result[k] for k in ['near_certain_rows','discordant_row_fraction','equivalent_flip_rate_for_stretch_ceiling','seconds','verdict']}))


if __name__=='__main__':main()
