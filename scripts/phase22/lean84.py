"""Independent leak-free CPU adaptation of the published lean84 hypothesis.

Static features use no competition labels. The teacher uses original labels only.
Competition target encodings are cross-fitted separately at each training level.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '6')
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import load_cached_parquet, file_sha256, arr_sha256, git_commit, save_json
from src.features.s6e10 import SURVEY13, META4, NUMS, FoldSafeTE
from src.validation.folds import get_scheme

PROTOCOL = ROOT / 'configs/phase22/private_breakthrough_20261019.json'
PROTOCOL_SHA = '536355875e034df2c45112583a375f1afceb195976e4c6d10339dfb4a5172fef'
TE_KEYS = [ ['Flight Distance'], ['Age'], ['Flight Distance', 'Age'],
            ['Flight Distance', 'Class'], ['Flight Distance', 'Type of Travel'],
            ['Age', 'Class', 'Type of Travel', 'Customer Type'],
            ['Inflight wifi service', 'Online boarding'], ['Class', 'Type of Travel'] ]

def static_features(frame, teacher):
    """Label-free combined train/test category vocabulary; no id accepted."""
    raw = frame[SURVEY13 + META4 + NUMS].copy()
    raw['Arrival Delay in Minutes'] = raw['Arrival Delay in Minutes'].fillna(0)
    raw['Departure Delay in Minutes'] = raw['Departure Delay in Minutes'].fillna(0)
    x = raw.drop(columns=['Customer Type', 'Departure Delay in Minutes']).copy()
    a, d = raw['Arrival Delay in Minutes'], raw['Departure Delay in Minutes']
    x['delay_sum'] = a+d; x['delay_diff'] = a-d
    x['arr_log'] = np.log1p(a.clip(lower=0)); x['dep_log'] = np.log1p(d.clip(lower=0))
    x['delay_distance'] = (a+d)/(raw['Flight Distance']+1)
    ratings = raw[SURVEY13].replace(0, np.nan)
    x['zero_count'] = (raw[SURVEY13] == 0).sum(axis=1)
    x['rating_mean'] = ratings.mean(axis=1); x['rating_std'] = ratings.std(axis=1).fillna(0)
    x['rating_max'] = ratings.max(axis=1); x['rating_min'] = ratings.min(axis=1)
    cats = ['Gender', 'Type of Travel', 'Class']
    for name in ['Age','Flight Distance','Arrival Delay in Minutes']:
        c=name+'__cat'; x[c]=raw[name].astype(str); cats.append(c)
    for rating in SURVEY13:
        for bg in ['Class','Type of Travel','Customer Type']:
            if rating=='Food and drink' and bg=='Customer Type': continue
            c=rating+'__X__'+bg; x[c]=raw[rating].astype(str)+'|'+raw[bg].astype(str); cats.append(c)
    for i,(left,right) in enumerate([
        ('Inflight wifi service','Ease of Online booking'),
        ('Inflight wifi service','Inflight entertainment'),
        ('Inflight wifi service','Online boarding'),
        ('Food and drink','Inflight entertainment'),
        ('Inflight entertainment','Seat comfort')]): x['gap_'+str(i)]=raw[left]-raw[right]
    x['teach_lgbm']=np.asarray(teacher)
    for c in cats: x[c]=pd.factorize(x[c].astype(object),sort=True)[0].astype('int32')
    keys={f'te_{i}':raw[cols].astype(str).agg('|'.join,axis=1) for i,cols in enumerate(TE_KEYS)}
    assert x.shape[1]==76 and len(cats)==44
    return x.to_numpy(dtype='float32'), list(x.columns), [x.columns.get_loc(c) for c in cats], keys

def encoded(keys, y, fit, apply, seed):
    """Only y[fit] may influence any returned encoding, including its prior."""
    kfit={n:s.iloc[fit].reset_index(drop=True) for n,s in keys.items()}
    kapp={n:s.iloc[apply].reset_index(drop=True) for n,s in keys.items()}
    te=FoldSafeTE(kfit,smooths=(20.0,),inner_folds=5,inner_seed=seed)
    a,_=te.crossfit(kfit,np.asarray(y)[fit]); b,_=te.apply(kfit,np.asarray(y)[fit],kapp)
    return a[:,::2],b[:,::2]  # Means only, as preregistered; omit count columns.

def main():
    import lightgbm as lgb
    import subprocess
    started=time.monotonic()
    assert file_sha256(PROTOCOL)==PROTOCOL_SHA
    spec=json.loads(PROTOCOL.read_text())
    assert spec==json.loads(subprocess.check_output(['git','show','80d9eea:'+PROTOCOL.relative_to(ROOT).as_posix()]))
    cfg=spec['fresh_CPU']; out=ROOT/'artifacts/phase22/lean84'; out.mkdir(parents=True,exist_ok=True)
    report=ROOT/'reports/phase22/lean84_primary.json'; assert not report.exists()
    for p,h in spec['protected_files'].items(): assert file_sha256(p)==h,p
    assert file_sha256(cfg['teacher_cache'])==cfg['teacher_cache_sha256']
    assert file_sha256('artifacts/features/static_core3.json')==cfg['teacher_names_sha256']
    tr,te=load_cached_parquet(); y=tr.satisfaction.to_numpy(dtype='int8'); ids=tr.id.to_numpy()
    folds=get_scheme('primary',y,ids).folds
    names=json.loads(Path('artifacts/features/static_core3.json').read_text()); j=names.index('teach_lgbm')
    with np.load(cfg['teacher_cache'],allow_pickle=False) as z:
        teacher=np.concatenate([z['tr'][:,j],z['te'][:,j]])
    comb=pd.concat([tr,te],ignore_index=True)
    x,names,cats,keys=static_features(comb,teacher)
    del teacher,comb; gc.collect()
    names=names+[f'te_{i}' for i in range(8)]
    seed=cfg['seed']; params={**cfg['model_params'],'seed':seed,'bagging_seed':seed+1,'feature_fraction_seed':seed+2}
    oof=np.full(len(y),np.nan,dtype='float32'); tests=np.zeros(len(te),dtype='float64'); details=[]
    test_idx=np.arange(len(tr),len(x))
    def deadline(_env=None):
        if time.monotonic()-started>cfg['wall_limit_hours']*3600: raise TimeoutError('Frozen CPU wall limit reached')
    for k in cfg['primary_folds']:
        deadline(); fold_started=time.monotonic()
        fit=np.flatnonzero(folds!=k); val=np.flatnonzero(folds==k)
        inner,es=train_test_split(fit,test_size=.1,random_state=seed+k,stratify=y[fit])
        assert not np.intersect1d(inner,es).size and not np.intersect1d(fit,val).size
        a,b=encoded(keys,y,inner,es,seed+k)
        xa=np.column_stack([x[inner],a]); xb=np.column_stack([x[es],b])
        ds=lgb.Dataset(xa,label=y[inner],categorical_feature=cats,free_raw_data=True)
        dv=lgb.Dataset(xb,label=y[es],categorical_feature=cats,reference=ds,free_raw_data=True)
        selected=lgb.train(params,ds,num_boost_round=cfg['max_iterations'],valid_sets=[dv],
                           callbacks=[lgb.early_stopping(cfg['inner_ES_patience'],verbose=False),deadline])
        best=selected.best_iteration; del selected,ds,dv,xa,xb,a,b; gc.collect()
        apply=np.concatenate([val,test_idx]); a,b=encoded(keys,y,fit,apply,seed+100+k)
        xa=np.column_stack([x[fit],a]); xb=np.column_stack([x[apply],b])
        ds=lgb.Dataset(xa,label=y[fit],categorical_feature=cats,free_raw_data=True)
        model=lgb.train(params,ds,num_boost_round=best,callbacks=[deadline])
        p=model.predict(xb,num_threads=6).astype('float32')
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        oof[val]=p[:len(val)]; tests+=p[len(val):]/5
        fold_dir=out/f'f{k}'; fold_dir.mkdir(); model.save_model(str(fold_dir/'model.txt'))
        np.save(fold_dir/'oof.npy',p[:len(val)]); np.save(fold_dir/'test.npy',p[len(val):]); np.save(fold_dir/'eval_ids.npy',ids[val])
        item={'fold':k,'best_iteration':best,'auc':float(roc_auc_score(y[val],p[:len(val)])),
              'seconds':time.monotonic()-fold_started,'fit_ids_sha256':arr_sha256(ids[fit]),
              'eval_ids_sha256':arr_sha256(ids[val]),'inner_fit_ids_sha256':arr_sha256(ids[inner]),
              'inner_ES_ids_sha256':arr_sha256(ids[es]),'prediction_sha256':arr_sha256(p),
              'model_sha256':file_sha256(fold_dir/'model.txt'),'fit_rows':len(fit),'eval_rows':len(val)}
        save_json(item,fold_dir/'report.json'); details.append(item)
        save_json({'status':'RUNNING','complete_folds':details,'elapsed_seconds':time.monotonic()-started},out/'progress.json')
        print(json.dumps(item),flush=True)
        del model,ds,xa,xb,a,b,p; gc.collect()
    assert np.isfinite(oof).all(); tests=tests.astype('float32')
    np.save(out/'oof.npy',oof); np.save(out/'test.npy',tests)
    for p,h in spec['protected_files'].items(): assert file_sha256(p)==h,p
    save_json({'status':'COMPLETE_FIVE_FOLD_PRIMARY','git':git_commit(),'prereg_sha256':PROTOCOL_SHA,
               'source_sha256':file_sha256(__file__),'folds':details,'oof_auc':float(roc_auc_score(y,oof)),
               'oof_sha256':arr_sha256(oof),'test_sha256':arr_sha256(tests),'feature_names':names,
               'categorical_indices':cats,'CPU_wall_seconds':time.monotonic()-started,'GPU_hours':0,
               'protected_files':spec['protected_files'],'train_ids_sha256':arr_sha256(ids),
               'test_ids_sha256':arr_sha256(te.id.to_numpy()),'verdict':'AWAITING_FROZEN_PAIRED_COMPARISON'},report)
    print('COMPLETE lean84 five-fold CPU',flush=True)

if __name__=='__main__': main()
