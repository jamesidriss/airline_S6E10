"""Frozen raw/route replication, with complete per-fold prediction contracts."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.validation.compare import logit
from scripts.assemble_sol_foundation_test import checked_probability, verify_primary
from scripts.run_sol_tabpfn import frames
from scripts.score_sol_foundation import paired_bootstrap, correlations
from scripts.audit_sol_state import pair_diagnostic


def verify_raw(k, tag, tr, te, ids, y, folds, data_hash):
    path = REPORTS/tag/f'raw_f{k}.json'; r=json.loads(path.read_text())
    source=json.loads((REPORTS/'sol_tabpfn35_predict_guard/raw_f0.json').read_text())
    assert r['library_version']==version('tabpfn')
    fi,va=np.flatnonzero(folds!=k),np.flatnonzero(folds==k)
    assert r['arm']=='raw' and r['fold']==k and r['full_intended_population'] and not r['timing_only']
    assert r['train_rows']==len(fi) and r['validation_rows']==len(va)
    assert r['fold_sha256']==arr_sha256(folds) and r['data_sha256']==data_hash
    assert r['fit_ids_sha256']==arr_sha256(ids[fi]) and r['validation_ids_sha256']==arr_sha256(ids[va])
    for key in ('checkpoint','library_version','params','seed','precision','icl_bf16','decoder_inplace_gelu',
                'inference_chunk_cells','inference_col_chunk_size','batch_size','gpu_memory_fraction',
                'prefit_host_reserve_gib','prediction_host_reserve_gib','windows_mqa_backend','reuse_query_output'):
        assert r[key]==source[key],key
    assert file_sha256(r['params']['model_path'])==r['checkpoint']['checkpoint_sha256']
    for p,key in [('scripts/run_sol_tabpfn.py','source_sha256'),('src/models/windows_attention.py','backend_source_sha256'),
                  ('src/models/pointwise_inference.py','decoder_chunk_source_sha256'),('src/models/resource_guard.py','resource_guard_source_sha256')]:
        assert file_sha256(p)==r[key]
    x,_,names,cats,maps=frames(tr,te,False)
    assert names==r['feature_names'] and cats==r['params']['categorical_features_indices'] and maps==r['label_free_category_maps']
    assert arr_sha256(x[fi])==r['feature_fit_sha256'] and arr_sha256(x[va])==r['feature_val_sha256']
    p=checked_probability(ARTIFACTS/tag/f'raw_f{k}.npy',ARTIFACTS/tag/f'ids_f{k}.npy',r['prediction_sha256'],ids[va])
    assert abs(float(roc_auc_score(y[va],p))-r['auc'])<1e-14
    return p,{'tag':tag,'report_sha256':file_sha256(path),'prediction_sha256':arr_sha256(p),'auc':r['auc']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',required=True);ap.add_argument('--name',required=True)
    args=ap.parse_args();ks=list(map(int,args.folds.split(',')));assert ks==list(range(len(ks))) and len(ks)<=5
    path=REPORTS/(args.name+'.json');root=ARTIFACTS/args.name
    if path.exists() or root.exists():raise FileExistsError('Preserve completed evidence; choose a new name')
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    dh={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    primary,vectors,folds=verify_primary('reports/sol_route_aux10_primary_final.json',ids,y,dh)
    root.mkdir();records=[];oof=np.full(len(y),np.nan,dtype='float32');raw_oof=oof.copy()
    for k in ks:
        va=np.flatnonzero(folds==k);raw,proof=verify_raw(k,'sol_tabpfn35_predict_guard' if k==0 else 'sol_phase15_raw',tr,te,ids,y,folds,dh)
        route,aux,base=vectors['route'][va],vectors['aux10'][va],vectors['candidate'][va]
        candidate=expit(.375*logit(route)+.125*logit(raw)+.5*logit(aux)).astype('float32')
        standalone=expit(.75*logit(route)+.25*logit(raw)).astype('float32')
        standalone_gain=float(roc_auc_score(y[va],standalone)-roc_auc_score(y[va],route))
        gain=float(roc_auc_score(y[va],candidate)-roc_auc_score(y[va],base))
        records.append({'fold':k,'raw_proof':proof,'candidate_auc':float(roc_auc_score(y[va],candidate)),
            'v6_auc':float(roc_auc_score(y[va],base)),'delta_vs_v6':gain,'standalone_gain':standalone_gain,
            'candidate_sha256':arr_sha256(candidate),'ids_sha256':arr_sha256(ids[va]),
            'discovery_pass':bool(standalone_gain>=5e-5 or gain>=1e-5),
            'candidate_correlation_vs_v6':correlations(candidate,base),'raw_route_correlation':correlations(raw,route),
            'pair_rescue_damage':pair_diagnostic(y[va],candidate,base)})
        np.save(root/f'candidate_f{k}.npy',candidate);np.save(root/f'ids_f{k}.npy',ids[va])
        oof[va]=candidate;raw_oof[va]=raw
    mask=np.isin(folds,ks);ds=np.array([r['delta_vs_v6'] for r in records])
    bootstrap=paired_bootstrap(y[mask],{'v6':vectors['candidate'][mask],'candidate':oof[mask]},reference='v6')
    np.save(root/'candidate_selected.npy',oof[mask]);np.save(root/'raw_selected.npy',raw_oof[mask]);np.save(root/'ids_selected.npy',ids[mask])
    full=len(ks)==5
    if full:
        np.save(root/'candidate_oof.npy',oof);np.save(root/'raw_oof.npy',raw_oof);np.save(root/'train_ids.npy',ids)
    result={'utc':datetime.now(timezone.utc).isoformat(),'git':git_commit(),'scope':'full primary OOF' if full else 'selected primary folds; not full OOF',
        'folds':records,'weights':{'route':.375,'raw':.125,'aux10':.5},'data_sha256':dh,'fold_sha256':arr_sha256(folds),
        'selected_ids_sha256':arr_sha256(ids[mask]),'selected_oof_sha256':arr_sha256(oof[mask]),
        'scope_sha256':file_sha256('research/sol_phase15_scope_20261008.json'),
        'source_sha256':{p:file_sha256(p) for p in ('scripts/evaluate_sol_phase15_raw.py','scripts/run_sol_tabpfn.py')},
        'selected_pooled_auc':float(roc_auc_score(y[mask],oof[mask])),
        'selected_pooled_delta_vs_v6':float(roc_auc_score(y[mask],oof[mask])-roc_auc_score(y[mask],vectors['candidate'][mask])),
        'mean_paired_gain':float(ds.mean()),'paired_se':float(ds.std(ddof=1)/np.sqrt(len(ds))) if len(ds)>1 else None,
        'positive_folds':int((ds>0).sum()),'paired_bootstrap':bootstrap,
        'verdict':'POSITIVE_REPLICATION' if (ds>0).all() and all(r['discovery_pass'] for r in records) else 'FAILED_REPLICATION_STOP',
        'test_policy_if_admitted':'Equal probability average of five exact primary FIT contexts separately for raw and route, then frozen method-logit mixture; never699635-row GPU context'}
    save_json(result,path)
    with Path('experiments/ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'exp_id':args.name,'ts':result['utc'],'kind':'FROZEN_REPRESENTATION_REPLICATION','report':str(path),
            'report_sha256':file_sha256(path),'paired_fold_deltas':ds.tolist(),'corr_with_champion':[r['candidate_correlation_vs_v6'] for r in records],
            'verdict':result['verdict']})+'\n')
    print(json.dumps({k:result[k] for k in ('scope','selected_pooled_auc','selected_pooled_delta_vs_v6','mean_paired_gain','paired_se','verdict')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
