"""Compare one fixed small pseudo-test policy audit; never select a weight."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,load_cached_parquet,save_json
from src.validation.folds import get_scheme
from scripts.run_sol_policy_pseudotest import partitions
from scripts.score_sol_foundation import correlations,paired_bootstrap


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='sol_policy_pseudotest');args=ap.parse_args()
    output=REPORTS/f'{args.tag}_comparison.json'
    if output.exists():raise FileExistsError('Preserve existing policy comparison')
    tr,_=load_cached_parquet();ids=tr.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    primary=get_scheme('primary',y,ids).folds;shadow=get_scheme('shadow',y,ids).folds
    fits,query=partitions(primary,shadow,ids)
    predictions={};proofs=[];first=None
    for context in ['all','0','1','2','3','4']:
        path=REPORTS/args.tag/f'context_{context}.json'
        record=json.loads(path.read_text(encoding='utf-8'));c=record['contract']
        assert record['status']=='COMPLETE_SMALL_POLICY_AUDIT_NOT_FULL_OOF'
        if first is None:first=c
        for field in ('params','seed','query_ids_sha256','reference_report_sha256','feature_query_sha256','checkpoint'):
            assert c[field]==first[field], 'Pseudo-test policy recipe or query changed'
        assert c['seed']==1201 and not c['outer_query_labels_used_in_fit'] and c['early_stopping']=='none'
        assert c['context']==context and c['outer_scheme']=='shadow' and c['outer_holdout']==0
        assert c['train_rows']==len(fits[context]) and c['query_rows']==len(query)
        assert c['fit_ids_sha256']==arr_sha256(ids[fits[context]]) and c['query_ids_sha256']==arr_sha256(ids[query])
        assert c['primary_fold_sha256']==arr_sha256(primary) and c['shadow_fold_sha256']==arr_sha256(shadow)
        assert c['policy_scope_sha256']==file_sha256('research/sol_recovery_scope_20261008.json')
        assert all(file_sha256(p)==h for p,h in c['source_sha256'].items())
        assert all(file_sha256(f'data/raw/{s}.csv')==h for s,h in c['data_sha256'].items())
        prediction=np.load(ARTIFACTS/args.tag/f'context_{context}.npy')
        assert prediction.shape==(len(query),) and arr_sha256(prediction)==record['prediction_sha256']
        assert np.array_equal(np.load(ARTIFACTS/args.tag/f'query_ids_{context}.npy'),ids[query])
        assert np.isfinite(prediction).all() and ((prediction>=0)&(prediction<=1)).all()
        assert abs(float(roc_auc_score(y[query],prediction))-record['auc_on_fixed_small_outer_query'])<1e-14
        predictions[context]=prediction;proofs.append({'context':context,'report_sha256':file_sha256(path),
            'fit_rows':len(fits[context]),'prediction_sha256':arr_sha256(prediction)})
    average=np.mean([predictions[str(k)].astype('float64') for k in range(5)],axis=0).astype('float32')
    np.save(ARTIFACTS/args.tag/'average.npy',average)
    baseline=predictions['all'];yy=y[query]
    result={'status':'COMPLETE_SMALL_POLICY_COMPARISON_NOT_FINALIST_OOF',
        'query_rows':len(query),'single_context_rows':len(fits['all']),
        'single_auc':float(roc_auc_score(yy,baseline)),'five_context_average_auc':float(roc_auc_score(yy,average)),
        'delta_average_minus_single':float(roc_auc_score(yy,average)-roc_auc_score(yy,baseline)),
        'correlations':correlations(average,baseline,yy),
        'paired_bootstrap':paired_bootstrap(yy,{'single':baseline,'average':average},reference='single'),
        'prediction_sha256':arr_sha256(average),'query_ids_sha256':arr_sha256(ids[query]),'proofs':proofs,
        'limitation':'One fixed20k-row shadow holdout with100k-row training pool. Each averaged member uses80percent of that pool. This audits averaging under strict exclusion; it cannot establish full-data transfer or a new full OOF score.'}
    save_json(result,output)
    print(json.dumps({k:result[k] for k in ('status','single_auc','five_context_average_auc','delta_average_minus_single')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
