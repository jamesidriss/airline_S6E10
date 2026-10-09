"""Hash-verified five-fold OOF assembly; partial results cannot qualify."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,arr_sha256,file_sha256,save_json
from scripts.phase16_common import bank
from scripts.phase16_pair_metrics import pair_rescue_damage
from scripts.score_sol_foundation import correlations,paired_bootstrap


def scatter_predictions(parts,ids,folds):
    if set(parts)!=set(range(5)): raise ValueError('All five folds are required')
    out=np.full(len(ids),np.nan,dtype='float32')
    for k,(p,applied_ids) in parts.items():
        m=np.flatnonzero(folds==k)
        if not np.array_equal(applied_ids,ids[m]) or p.shape!=(len(m),):
            raise ValueError('Prediction/ID alignment mismatch')
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
            raise ValueError('Invalid probability')
        out[m]=p
    if not np.isfinite(out).all(): raise ValueError('Incomplete coverage')
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--fold-tags',help='JSON mapping of fold number to a verified resumed run tag')
    args=ap.parse_args()
    tags={k:args.tag for k in range(5)}
    if args.fold_tags:
        overrides=json.loads(Path(args.fold_tags).read_text())
        assert all(str(k) in ('0','1','2','3','4') for k in overrides)
        tags.update({int(k):v for k,v in overrides.items()})
    root=ARTIFACTS/args.tag/'primary';path=Path('reports')/(args.tag+'_primary.json')
    if root.exists() or path.exists(): raise FileExistsError('Preserve completed assembly')
    tr,_,y,folds,vectors,_,proof=bank();ids=tr.id.to_numpy()
    scope_path=Path('research/phase17_scope_20261009.json');scope=json.loads(scope_path.read_text())
    route_parts={};portfolio_parts={};evidence=[];counts=[];deltas=[];frozen_recipe=None
    for k in range(5):
        rp=Path('reports')/(tags[k]+f'_f{k}.json');ep=Path('reports')/(tags[k]+f'_f{k}_evaluation.json')
        r=json.loads(rp.read_text());e=json.loads(ep.read_text());c=r['contract'];folder=ARTIFACTS/tags[k]/f'f{k}'
        assert r['status']=='COMPLETE_FROZEN_PRIMARY_FOLD' and c['scheme']=='primary' and c['fold']==k
        assert c['scope_sha256']==file_sha256(scope_path) and c['bank']==proof
        assert c['seed']==1201 and not c['outer_labels_used_for_fit_or_configuration']
        recipe={s:c[s] for s in ('params','seed','feature_names','label_free_category_maps','checkpoint',
            'library_versions','library_source_sha256','test_policy')}
        if frozen_recipe is None:frozen_recipe=recipe
        else:assert recipe==frozen_recipe,'Recipe/library drift between primary folds'
        assert all(file_sha256(s)==h for s,h in c['source_sha256'].items())
        assert e['source_report_sha256']==file_sha256(rp) and e['scope_sha256']==file_sha256(scope_path)
        assert all(file_sha256(s)==h for s,h in e['source_sha256'].items())
        assert c['fit_ids_sha256']==arr_sha256(ids[folds!=k]) and c['train_rows']==int((folds!=k).sum())
        aid=np.load(folder/'apply_ids.npy');route=np.load(folder/'prediction.npy');p=np.load(folder/'portfolio.npy')
        assert arr_sha256(route)==r['prediction_sha256'] and arr_sha256(p)==e['portfolio_prediction_sha256']
        m=folds==k;d=float(roc_auc_score(y[m],p)-roc_auc_score(y[m],vectors['candidate'][m]))
        assert abs(d-e['portfolio_delta'])<1e-14
        counts.append(c['params']['n_estimators']);deltas.append(d)
        route_parts[k]=(route,aid);portfolio_parts[k]=(p,aid)
        evidence.append({'fold':k,'source_tag':tags[k],'run_report_path':str(rp),'evaluation_path':str(ep),
            'run_report_sha256':file_sha256(rp),'evaluation_sha256':file_sha256(ep),
            'route_auc':r['auc'],'portfolio_auc':e['portfolio_auc'],'delta':d,'seconds':r['seconds']})
    assert len(set(counts))==1 and counts[0] in (2,4)
    route=scatter_predictions(route_parts,ids,folds);p=scatter_predictions(portfolio_parts,ids,folds)
    mean=float(np.mean(deltas));se=float(np.std(deltas,ddof=1)/np.sqrt(5));positive=int((np.array(deltas)>0).sum())
    auc=float(roc_auc_score(y,p));pooled_delta=auc-float(roc_auc_score(y,vectors['candidate']))
    g=scope['full_performance'];corr=correlations(p,vectors['candidate'],y);pairs=pair_rescue_damage(y,p,vectors['candidate'])
    assert abs(pairs['net_auc_gain']-pooled_delta)<1e-14
    segments=[]
    for col in ('Class','Type of Travel','Customer Type','Gender'):
        for v in sorted(tr[col].astype(str).unique()):
            m=tr[col].astype(str).to_numpy()==v
            segments.append({'column':col,'value':v,'rows':int(m.sum()),
                'delta':float(roc_auc_score(y[m],p[m])-roc_auc_score(y[m],vectors['candidate'][m]))})
    passed=pooled_delta>=g['pooled_gain_min'] and mean>=g['mean_paired_gain_min'] and positive>=g['positive_folds_min'] and mean>=g['paired_se_multiple']*se
    h=scope['hedge'];hedge=(pooled_delta>=-15e-6 and mean>=-15e-6 and int((np.array(deltas)>=-30e-6).sum())>=4
        and corr['spearman']<=h['portfolio_oof_spearman_max'] and pairs['rescued_pair_fraction']>=h['rescued_pair_fraction_min']
        and pairs['rescued_pair_fraction']/max(pairs['damaged_pair_fraction'],1e-30)>=h['rescue_damage_ratio_min']
        and min(s['delta'] for s in segments)>=h['worst_segment_delta_min'])
    root.mkdir();np.save(root/'route_oof.npy',route);np.save(root/'portfolio_oof.npy',p);np.save(root/'train_ids.npy',ids)
    result={'utc':datetime.now(timezone.utc).isoformat(),'bank':proof,'scope_sha256':file_sha256(scope_path),'internal_estimators':counts[0],
        'folds':evidence,'full_pooled_route_auc':float(roc_auc_score(y,route)),'full_pooled_oof_auc':auc,'pooled_delta':pooled_delta,
        'paired_fold_deltas':deltas,'mean_paired_gain':mean,'paired_se':se,'positive_folds':positive,'correlation_vs_v6':corr,
        'pair_rescue_damage':pairs,'segments':segments,'bootstrap':paired_bootstrap(y,{'candidate':p,'v6':vectors['candidate']},reference='v6'),
        'performance_admission':bool(passed),'hedge_primary_admission':bool(hedge),'test_and_shadow_required':True,
        'distance_to_stretch':scope['stretch_auc']-auc,'stretch_reached':bool(auc>scope['stretch_auc']),
        'route_sha256':arr_sha256(route),'portfolio_sha256':arr_sha256(p),'train_ids_sha256':arr_sha256(ids),
        'source_sha256':{s:file_sha256(s) for s in (__file__,'scripts/phase16_pair_metrics.py','scripts/score_sol_foundation.py')},
        'status':'PRIMARY_PASS_REQUIRES_INDEPENDENT_CONFIRMATION_AND_TEST' if passed or hedge else 'PRIMARY_FAIL_NO_PROMOTION'}
    save_json(result,path)
    row={'exp_id':args.tag+'_primary','kind':'FULL_FIVE_FOLD_ADMISSION','ts':result['utc'],'report':str(path),'report_sha256':file_sha256(path),
        'auc':auc,'paired_fold_deltas':deltas,'correlation_with_champion':corr,'full_oof':True,'verdict':result['status']}
    with Path('experiments/ledger.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps({k:result[k] for k in ('status','full_pooled_oof_auc','pooled_delta','mean_paired_gain','paired_se','positive_folds','distance_to_stretch')}))


if __name__=='__main__':main()
