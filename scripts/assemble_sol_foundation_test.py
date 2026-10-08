"""Certify complete fixed-method OOF/test vectors, without changing weights."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,load_cached_parquet,save_json
from src.validation.compare import logit
from src.validation.folds import get_scheme
from src.submission import store
from scripts.replay_sol_classical import frozen_roles,resolved_params,PROTOCOL
from scripts.assemble_sol_clean import certify


def checked_probability(path,ids_path,expected_hash,ids):
    p=np.load(path)
    assert p.shape==(len(ids),) and p.dtype==np.float32
    assert arr_sha256(p)==expected_hash and np.array_equal(np.load(ids_path),ids)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p


def verify_primary(path,ids,y,data_hash):
    r=json.loads(Path(path).read_text(encoding='utf-8'))
    assert r['scope']=='full primary OOF' and r['scheme']=='primary'
    assert r['admission_gate_against_clean_auxiliary_stack']['admit']
    assert r['data_sha256']==data_hash and r['ordered_train_ids_sha256']==arr_sha256(ids)
    folds=get_scheme('primary',y,ids).folds
    assert r['fold_sha256']==arr_sha256(folds)
    assert sorted(s['fold'] for s in r['folds'])==list(range(5))
    root=ARTIFACTS/Path(path).stem
    vectors={name:checked_probability(root/(name+'_oof.npy'),root/'train_ids.npy',r[key],ids)
        for name,key in [('candidate','oof_sha256'),('aux10','strict_aux10_oof_sha256'),('route','route_oof_sha256')]}
    return r,vectors,folds


def context_average_test(tag,primary,folds,ids,test_ids,data_hash):
    total=np.zeros(len(test_ids));proofs=[]
    for k in range(5):
        path=REPORTS/tag/f'test_f{k}.json'
        r=json.loads(path.read_text(encoding='utf-8'));c=r['contract']
        assert r['status']=='COMPLETE_ONE_PRIMARY_CONTEXT_TEST_REQUIRES_ALL_FIVE'
        assert c['fold']==k and c['scheme']=='primary' and not c['entire_competition_training_context']
        assert not c['timing_only'] and c['seed']==1201
        assert c['train_rows']==int((folds!=k).sum()) and c['test_rows']==len(test_ids)
        assert c['fit_ids_sha256']==arr_sha256(ids[folds!=k]) and c['test_ids_sha256']==arr_sha256(test_ids)
        assert c['data_sha256']==data_hash and c['fold_sha256']==arr_sha256(folds)
        assert all(file_sha256(p)==h for p,h in c['source_sha256'].items())
        proof=next(f for f in primary['folds'] if f['fold']==k)['foundation']
        assert c['reference_report_sha256']==proof['report_sha256']
        reference=json.loads((REPORTS/proof['tag']/f'route_f{k}.json').read_text(encoding='utf-8'))
        assert c['params']==reference['params'] and c['feature_fit_sha256']==reference['feature_fit_sha256']
        assert c['checkpoint']==reference['checkpoint'] and c['test_policy_sha256']==file_sha256('research/sol_test_inference_contract.md')
        p=checked_probability(ARTIFACTS/tag/f'test_f{k}.npy',ARTIFACTS/tag/f'test_ids_f{k}.npy',r['test_prediction_sha256'],test_ids)
        total+=p/5
        proofs.append({'fold':k,'report_sha256':file_sha256(path),'prediction_sha256':arr_sha256(p),
            'fit_ids_sha256':c['fit_ids_sha256'],'train_rows':c['train_rows']})
    return total.astype('float32'),proofs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary-report',default='reports/sol_route_aux10_primary.json')
    ap.add_argument('--classical-tag',default='sol_clean_aux10_primary')
    ap.add_argument('--foundation-tag',required=True)
    ap.add_argument('--foundation-policy',choices=['full','context_average'],default='full')
    ap.add_argument('--name',default='sol_route_aux10_primary')
    args=ap.parse_args()
    output=REPORTS/(args.name+'_test_certificate.json')
    if output.exists():raise FileExistsError('Preserve the existing test certificate')
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();test_ids=te.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    primary,vectors,folds=verify_primary(args.primary_report,ids,y,data_hash)
    roles=[r for r in frozen_roles() if r['aux_arm']];assert len(roles)==10
    index=store._load_index();aux_logit=np.zeros(len(test_ids));proofs=[]
    for role in roles:
        entry={'source_role':role['member'],'store_id':f'{args.classical_tag}_{role["member"]}',
            'tag':args.classical_tag,'scheme':'primary','protocol':PROTOCOL,
            'family':role['family'],'view':role['view']}
        _,test,proof=certify(entry,index,ids,test_ids,y)
        rp=REPORTS/args.classical_tag/f'{role["member"]}_test.json'
        rec=json.loads(rp.read_text(encoding='utf-8'));c=rec['contract']
        counts=[json.loads((REPORTS/args.classical_tag/f'{role["member"]}_f{k}.json').read_text())['n_trees'] for k in range(5)]
        assert c['role']==role and c['params']==resolved_params(role)
        assert c['capacity']['selected_fold_counts']==counts and c['capacity']['n_trees']==int(np.median(counts))
        assert c['data_sha256']==data_hash and c['train_rows']==len(tr) and c['test_rows']==len(te)
        assert c['upstream']['contract']['inner_folds']==3
        for k in range(5):
            fold=next(f for f in primary['folds'] if f['fold']==k)
            old=next(p for p in fold['classical_certificates'] if p['member']==role['member'])
            assert old['report_sha256']==index[entry['store_id']]['meta']['fold_reports_sha256'][k]
        aux_logit+=logit(test)/10
        proofs.append({**proof,'test_report_sha256':file_sha256(rp),'median_trees':int(np.median(counts))})
    if args.foundation_policy=='full':
        rp=REPORTS/args.foundation_tag/'test.json';r=json.loads(rp.read_text(encoding='utf-8'));c=r['contract']
        assert r['status']=='COMPLETE_TEST_REQUIRES_FULL_OOF_SCORECARD' and not c['timing_only']
        assert c['entire_competition_training_context'] and c['train_rows']==len(tr) and c['test_rows']==len(te)
        assert c['fit_ids_sha256']==arr_sha256(ids) and c['test_ids_sha256']==arr_sha256(test_ids)
        assert c['data_sha256']==data_hash and all(file_sha256(p)==h for p,h in c['source_sha256'].items())
        first=next(s for s in primary['folds'] if s['fold']==0)
        assert c['reference_report_sha256']==first['foundation']['report_sha256']
        reference=json.loads((REPORTS/first['foundation']['tag']/'route_f0.json').read_text())
        assert c['params']==reference['params'] and c['seed']==reference['seed']
        route=checked_probability(ARTIFACTS/args.foundation_tag/'test.npy',ARTIFACTS/args.foundation_tag/'test_ids.npy',r['test_prediction_sha256'],test_ids)
        foundation_proofs=[{'report_sha256':file_sha256(rp),'prediction_sha256':arr_sha256(route)}]
        foundation_policy=c['test_policy']
    else:
        route,foundation_proofs=context_average_test(args.foundation_tag,primary,folds,ids,test_ids,data_hash)
        foundation_policy='Equal probability average of five exact immutable primary FIT contexts; not a full-data fit'
    test=(1/(1+np.exp(-(logit(route)+aux_logit)/2))).astype('float32')
    auxiliary=(1/(1+np.exp(-aux_logit))).astype('float32')
    root=ARTIFACTS/(args.name+'_test');root.mkdir(exist_ok=False)
    np.save(root/'candidate.npy',test);np.save(root/'aux10.npy',auxiliary);np.save(root/'route.npy',route);np.save(root/'test_ids.npy',test_ids)
    certificate={'status':'CERTIFIED_FIXED_OOF_TEST_REQUIRES_ROBUSTNESS_SCORECARD',
        'primary_report_sha256':file_sha256(args.primary_report),'data_sha256':data_hash,
        'fold_sha256':arr_sha256(folds),'ordered_train_ids_sha256':arr_sha256(ids),'ordered_test_ids_sha256':arr_sha256(test_ids),
        'weights':{'route':.5,'equal_logit_auxiliary10':.5},'classical_proofs':proofs,
        'foundation_reports':foundation_proofs,'foundation_test_policy':foundation_policy,
        'oof_sha256':{n:arr_sha256(p) for n,p in vectors.items()},
        'test_sha256':{n:arr_sha256(p) for n,p in [('candidate',test),('aux10',auxiliary),('route',route)]},
        'test_artifact_root':str(root),'test_policy_sha256':file_sha256('research/sol_test_inference_contract.md')}
    save_json(certificate,output)
    store.save(args.name,vectors['candidate'],test,fold_scheme='primary',meta={
        'family':'blend','auc':primary['pooled_oof_auc'],'training_protocol':'sol_route_aux10_fixed_50_50_v1',
        'geometry':primary['hypothesis'],'test_policy':foundation_policy,'ordered_train_ids_sha256':arr_sha256(ids),
        'ordered_test_ids_sha256':arr_sha256(test_ids),'certificate_sha256':file_sha256(output)})
    print(f'Certified {len(test)} test rows; robustness scorecard still required')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
