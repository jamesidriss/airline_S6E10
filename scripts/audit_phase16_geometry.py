"""Three frozen aggregation geometries from certified method predictions."""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, save_json
from src.validation.compare import logit
from src.ensemble.lab import admission_gate
from scripts.phase16_common import bank, rank_geometry, SCOPE
from scripts.audit_sol_phase15 import evidence
from scripts.score_sol_foundation import paired_bootstrap, correlations


def ordering_sensitivity(a, b, n=2000000):
    rng=np.random.default_rng(20261010);i=rng.integers(0,len(a),n);j=rng.integers(0,len(a),n)
    x=np.sign(a[i]-a[j]);z=np.sign(b[i]-b[j])
    return {'sampled_unlabelled_row_pairs':n,'seed':20261010,
            'strict_order_reversals':int(((x*z)<0).sum()),
            'tie_status_changes':int(((x==0)!=(z==0)).sum()),
            'limitation':'Monte Carlo ordering diagnostic; no test label used'}


def main():
    start=time.monotonic();out=Path('reports/phase16_geometry_20261009.json');root=ARTIFACTS/'phase16_geometry'
    if out.exists() or root.exists():raise FileExistsError('Preserve prior evidence')
    scope=json.loads(SCOPE.read_text());tr,te,y,folds,vectors,tests,proof=bank()
    oofs={'G0':vectors['candidate'],'G1':((vectors['route']+vectors['aux10'])/2).astype('float32'),
          'G2':rank_geometry(vectors['route'],vectors['aux10'],folds)}
    tps={'G0':tests['candidate'],'G1':((tests['route']+tests['aux10'])/2).astype('float32'),
         'G2':rank_geometry(tests['route'],tests['aux10'])}
    assert np.max(np.abs(expit(.5*logit(vectors['route'])+.5*logit(vectors['aux10']))-oofs['G0']))<2e-6
    root.mkdir();np.save(root/'train_ids.npy',tr.id.to_numpy());np.save(root/'test_ids.npy',te.id.to_numpy())
    rows={}
    for n,p in oofs.items():
        assert p.shape==y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        r=evidence(y,folds,p,oofs['G0'],tr,{'audit':scope['geometry']})
        r['bootstrap']=paired_bootstrap(y,{'G0':oofs['G0'],n:p},reference='G0') if n!='G0' else None
        r['test_correlation_vs_v6']=correlations(tps[n],tps['G0'])
        r['test_ordering_sensitivity']=ordering_sensitivity(tps[n],tps['G0'])
        r['test_sha256']=arr_sha256(tps[n]);r['distance_to_stretch']=.9621-r['pooled_auc']
        base_aucs=[float(__import__('sklearn').metrics.roc_auc_score(y[folds==k],oofs['G0'][folds==k])) for k in range(5)]
        new_aucs=[a+d for a,d in zip(base_aucs,r['paired_fold_deltas'])]
        r['admission_gate']=admission_gate(n,base_aucs,new_aucs) if n!='G0' else None
        r['verdict']='CONTROL' if n=='G0' else ('PRIMARY_PASS_REQUIRES_UNTOUCHED_INDEPENDENT' if r['admission_gate']['admit'] and r['pooled_delta_vs_v6']>=3e-5 else 'PRIMARY_GATE_FAIL_NO_PROMOTION')
        np.save(root/(n+'_oof.npy'),p);np.save(root/(n+'_test.npy'),tps[n]);rows[n]=r
    result={'utc':datetime.now(timezone.utc).isoformat(),'scope_sha256':file_sha256(SCOPE),'bank':proof,
            'geometries':rows,'rank_policy':'Midrank percentiles within each applied primary fold; full test population separately; transform accepts predictions/groups only, no labels',
            'limitation':'Fixed weights and label-free transforms; choosing a geometry after OOF requires independent confirmation. Base predictions are certified; this comparison is not nested meta-parameter selection.',
            'source_sha256':{p:file_sha256(p) for p in ['scripts/audit_phase16_geometry.py','scripts/phase16_common.py','scripts/audit_sol_phase15.py']},'seconds':time.monotonic()-start}
    save_json(result,out)
    with Path('experiments/ledger.jsonl').open('a',encoding='utf-8') as f:
        for n,r in rows.items():
            f.write(json.dumps({'exp_id':'phase16_geometry_'+n,'ts':result['utc'],'kind':'FIXED_AGGREGATION_GEOMETRY',
                'report':str(out),'report_sha256':file_sha256(out),'paired_fold_deltas':r['paired_fold_deltas'],
                'corr_with_champion':r['oof_correlation_vs_v6'],'verdict':r['verdict']})+'\n')
    print(json.dumps({n:{k:r[k] for k in ['pooled_auc','pooled_delta_vs_v6','positive_folds','verdict']} for n,r in rows.items()}))


if __name__=='__main__':main()
