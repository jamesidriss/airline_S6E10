"""Certify complete, corrected neural OOF/test pairs from immutable fold files."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,load_cached_parquet,save_json
from src.submission import store
from src.validation.folds import get_scheme
from src.models.realmlp import TRAINING_PROTOCOL


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag',default='sol_neural_clean_compact')
    ap.add_argument('--members',default='all')
    args=ap.parse_args()
    frozen=json.loads((REPORTS/'finalist_v3_final.json').read_text(encoding='utf-8'))['members']
    specs={m['exp_id']:m for m in frozen if m['family'] in ('realmlp','tabm')}
    assert len(specs)==12
    chosen=list(specs) if args.members=='all' else args.members.split(',')
    tr,te=load_cached_parquet()
    y,ids=tr['satisfaction'].to_numpy(dtype='int8'),tr['id'].to_numpy()
    folds=get_scheme('primary',y,ids).folds
    root,reports=ARTIFACTS/args.tag,REPORTS/args.tag
    for member in chosen:
        spec=specs[member]
        oof=np.full(len(y),np.nan,dtype='float32'); test=np.zeros(len(te),dtype='float64')
        records=[]; first=None
        for k in range(5):
            path=reports/f'{member}_f{k}.json'
            r=json.loads(path.read_text(encoding='utf-8')); c=r['contract']
            assert c['training_protocol']==TRAINING_PROTOCOL and c['member']==member and c['fold']==k
            assert c['family']==spec['family']
            assert c['fold_sha256']==arr_sha256(folds) and c['save_test'] is True
            assert all(file_sha256(p)==h for p,h in c['source_sha256'].items())
            assert all(file_sha256(f'data/raw/{s}.csv')==h for s,h in c['data_sha256'].items())
            if first is None:
                first=c
            assert c['params']==first['params'] and c['feature_names']==first['feature_names']
            va=np.flatnonzero(folds==k)
            p=np.load(root/f'{member}_f{k}.npy'); t=np.load(root/f'{member}_test_f{k}.npy')
            assert arr_sha256(p)==r['prediction_sha256'] and arr_sha256(t)==r['test_prediction_sha256']
            assert np.array_equal(np.load(root/f'ids_f{k}.npy'),ids[va])
            assert arr_sha256(ids[va])==c['validation_ids_sha256']
            assert p.shape==(len(va),) and t.shape==(len(te),)
            assert np.isfinite(p).all() and np.isfinite(t).all()
            assert ((p>=0)&(p<=1)).all() and ((t>=0)&(t<=1)).all()
            assert r['test_ids_sha256']==arr_sha256(te['id'].to_numpy())
            oof[va]=p; test+=t/5; records.append(file_sha256(path))
        assert np.isfinite(oof).all() and ((oof>=0)&(oof<=1)).all() and ((test>=0)&(test<=1)).all()
        assert np.array_equal(np.load(root/'test_ids.npy'),te['id'].to_numpy())
        auc=float(roc_auc_score(y,oof)); fa=[float(roc_auc_score(y[folds==k],oof[folds==k])) for k in range(5)]
        meta={'family':spec['family'],'featureset':spec['featureset'],'params':first['params'],
            'seed':first['params']['random_state'],'training_protocol':TRAINING_PROTOCOL,'source_role':member,
            'auc':auc,'fold_aucs':fa,'test_policy':'average the exact fitted CV models in probability space',
            'fold_reports_sha256':records,'ordered_train_ids_sha256':arr_sha256(ids),
            'ordered_test_ids_sha256':arr_sha256(te['id'].to_numpy())}
        store.save(f'{args.tag}_{member}',oof,test,fold_scheme='primary',meta=meta)
        save_json({'meta':meta,'oof_sha256':arr_sha256(oof),'test_sha256':arr_sha256(test.astype('float32'))},
            reports/f'{member}_complete.json')
        print(f'{member}: honest OOF {auc:.9f}, exact CV-model test average certified',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
