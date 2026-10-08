"""Verify native39 saved evidence and its frozen marginal effect on both stacks."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.validation.compare import logit
from scripts.assemble_sol_foundation_test import verify_primary, checked_probability
from scripts.replay_sol_classical import frozen_roles, resolved_params
from scripts.score_sol_foundation import paired_bootstrap, correlations


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--raw-report',default='reports/sol_phase15_raw_twofold.json');args=ap.parse_args()
    ks=list(map(int,args.folds.split(',')));assert ks==list(range(len(ks))) and len(ks)<=5
    path=REPORTS/(args.name+'.json');root=ARTIFACTS/args.name
    if path.exists() or root.exists():raise FileExistsError('Preserve earlier evidence')
    tr,_=load_cached_parquet();ids=tr.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    dh={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    _,vectors,folds=verify_primary('reports/sol_route_aux10_primary_final.json',ids,y,dh)
    rawpath=Path(args.raw_report);raw_report=json.loads(rawpath.read_text());rawroot=ARTIFACTS/rawpath.stem
    assert raw_report['weights']=={'route':.375,'raw':.125,'aux10':.5}
    assert raw_report['data_sha256']==dh and raw_report['fold_sha256']==arr_sha256(folds)
    role=next(r for r in frozen_roles() if r['aux_arm']=='C1')
    root.mkdir();records=[];candidate=np.full(len(ids),np.nan,dtype='float32');joint=candidate.copy();rawbase=candidate.copy()
    for k in ks:
        va=np.flatnonzero(folds==k);fi=np.flatnonzero(folds!=k)
        source=REPORTS/'sol_phase15_native39'/f'cross_f{k}.json';r=json.loads(source.read_text());c=r['contract']
        assert sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==r['fingerprint']
        assert c['fold']==k and c['fold_sha256']==arr_sha256(folds) and c['data_sha256']==dh
        assert c['role']==role and c['params']==resolved_params(role)
        assert c['outer_fit_ids_sha256']==arr_sha256(ids[fi]) and c['validation_ids_sha256']==arr_sha256(ids[va])
        assert all(file_sha256(p)==h for p,h in c['source_sha256'].items())
        control_path=REPORTS/'sol_clean_aux10_primary'/f'{role["member"]}_f{k}.json';control=json.loads(control_path.read_text())
        assert c['control_report_sha256']==file_sha256(control_path)
        assert c['numeric_fit_sha256']==control['contract']['feature_fit_sha256']
        assert c['numeric_validation_sha256']==control['contract']['feature_val_sha256']
        assert len(c['feature_names'])==control['contract']['model_feature_count']+39
        assert len(c['native_columns'])==len(control['contract']['native_categorical_columns'])+39
        assert c['train_ids_sha256']==control['contract']['train_ids_sha256'] and c['es_ids_sha256']==control['contract']['es_ids_sha256']
        baseline=checked_probability(ARTIFACTS/'sol_clean_aux10_primary'/f'{role["member"]}_f{k}.npy',
            ARTIFACTS/'sol_clean_aux10_primary'/f'{role["member"]}_f{k}_ids.npy',c['control_prediction_sha256'],ids[va])
        native=checked_probability(ARTIFACTS/'sol_phase15_native39'/f'cross_f{k}.npy',
            ARTIFACTS/'sol_phase15_native39'/f'ids_f{k}.npy',r['prediction_sha256'],ids[va])
        p=checked_probability(ARTIFACTS/'sol_phase15_native39'/f'portfolio_f{k}.npy',
            ARTIFACTS/'sol_phase15_native39'/f'ids_f{k}.npy',r['portfolio_prediction_sha256'],ids[va])
        raw_row=next(f for f in raw_report['folds'] if f['fold']==k)
        raw=checked_probability(rawroot/f'candidate_f{k}.npy',rawroot/f'ids_f{k}.npy',raw_row['candidate_sha256'],ids[va])
        q=expit(logit(raw)+.05*(logit(native)-logit(baseline))).astype('float32')
        standalone=float(roc_auc_score(y[va],native)-roc_auc_score(y[va],baseline))
        assert abs(standalone-r['standalone_delta'])<1e-14 and abs(float(roc_auc_score(y[va],native))-r['auc'])<1e-14
        gain=float(roc_auc_score(y[va],p)-roc_auc_score(y[va],vectors['candidate'][va]))
        joint_gain=float(roc_auc_score(y[va],q)-roc_auc_score(y[va],raw))
        records.append({'fold':k,'standalone_auc':r['auc'],'standalone_gain':standalone,'v6_candidate_auc':float(roc_auc_score(y[va],p)),
            'v6_gain':gain,'joint_candidate_auc':float(roc_auc_score(y[va],q)),'joint_gain_vs_raw_stack':joint_gain,
            'correlation_vs_v6':correlations(p,vectors['candidate'][va]),'correlation_joint_vs_raw_stack':correlations(q,raw),
            'source_report_sha256':file_sha256(source),'seconds':r['seconds'],'trees':r['n_trees']})
        candidate[va]=p;joint[va]=q;rawbase[va]=raw
        np.save(root/f'joint_f{k}.npy',q);np.save(root/f'ids_f{k}.npy',ids[va])
    mask=np.isin(folds,ks);ds=np.array([r['v6_gain'] for r in records]);js=np.array([r['joint_gain_vs_raw_stack'] for r in records])
    def stats(d):return {'mean':float(d.mean()),'se':float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None,'positive':int((d>0).sum())}
    out={'utc':datetime.now(timezone.utc).isoformat(),'git':git_commit(),'scope':'full primary OOF' if len(ks)==5 else 'selected primary folds; not full OOF',
        'folds':records,'v6_marginal':stats(ds),'raw_stack_marginal':stats(js),'data_sha256':dh,'fold_sha256':arr_sha256(folds),
        'v6_selected_auc':float(roc_auc_score(y[mask],candidate[mask])), 'joint_selected_auc':float(roc_auc_score(y[mask],joint[mask])),
        'v6_bootstrap':paired_bootstrap(y[mask],{'v6':vectors['candidate'][mask],'candidate':candidate[mask]},reference='v6'),
        'joint_bootstrap':paired_bootstrap(y[mask],{'raw_stack':rawbase[mask],'joint':joint[mask]},reference='raw_stack'),
        'source_sha256':{__file__:file_sha256(__file__)},'raw_report_sha256':file_sha256(rawpath),
        'combination_protocol_sha256':file_sha256('research/sol_phase15_combination_protocol.json'),
        'verdict':'POSITIVE_PARTIAL_REQUIRES_REPLICATION_OR_ADMISSION' if (ds>0).all() and all(r['standalone_gain']>0 for r in records) else 'NOT_REPLICATED_FOR_V6_ADMISSION'}
    np.save(root/'joint_selected.npy',joint[mask]);np.save(root/'native_v6_selected.npy',candidate[mask]);np.save(root/'ids_selected.npy',ids[mask])
    if len(ks)==5:
        np.save(root/'joint_oof.npy',joint);np.save(root/'native_v6_oof.npy',candidate);np.save(root/'train_ids.npy',ids)
    save_json(out,path)
    with Path('experiments/ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'exp_id':args.name,'ts':out['utc'],'kind':'FROZEN_NATIVE39_CURRENT_STACK_MARGINAL','report':str(path),
            'report_sha256':file_sha256(path),'paired_fold_deltas':ds.tolist(),'joint_stack_deltas':js.tolist(),
            'corr_with_champion':[r['correlation_vs_v6'] for r in records],'verdict':out['verdict']})+'\n')
    print(json.dumps({k:out[k] for k in ('scope','v6_marginal','raw_stack_marginal','verdict')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
