"""CPU CatBoost on frozen lean84 features; inner-only iteration selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'6')
import gc,json,sys,time,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.common import load_cached_parquet,file_sha256,arr_sha256,save_json,git_commit
from src.validation.folds import get_scheme
from scripts.phase22.lean84 import static_features,encoded

def cat_frame(x,cats):
    frame=pd.DataFrame(x,columns=[f'f{i}' for i in range(x.shape[1])])
    for i in cats:frame[f'f{i}']=frame[f'f{i}'].astype('int32').astype('category')
    return frame

def main():
    from catboost import CatBoostClassifier
    start=time.monotonic();run_git=git_commit()
    protocol=Path('configs/phase22/lean84_cat_20261019.json')
    assert file_sha256(protocol)=='b3640e52bb09d7b923b0df326759b8799cdae8fe1e3cb4babbadd5d09c48ca03'
    spec=json.loads(protocol.read_text());assert spec==json.loads(subprocess.check_output(['git','show','479b642:'+protocol.as_posix()]))
    assert file_sha256(spec['feature_code'])==spec['feature_code_sha256']
    assert file_sha256(spec['feature_protocol'])==spec['feature_protocol_sha256']
    original=json.loads(Path(spec['feature_protocol']).read_text());cfg=original['fresh_CPU']
    for p,h in original['protected_files'].items():assert file_sha256(p)==h,p
    assert file_sha256(cfg['teacher_cache'])==cfg['teacher_cache_sha256']
    assert file_sha256('artifacts/features/static_core3.json')==cfg['teacher_names_sha256']
    out=Path('artifacts/phase22/lean84_cat');out.mkdir(parents=True,exist_ok=True)
    report=Path('reports/phase22/lean84_cat_primary.json');assert not report.exists()
    tr,te=load_cached_parquet();y=tr.satisfaction.to_numpy(dtype='int8');ids=tr.id.to_numpy();folds=get_scheme('primary',y,ids).folds
    names=json.loads(Path('artifacts/features/static_core3.json').read_text());j=names.index('teach_lgbm')
    with np.load(cfg['teacher_cache'],allow_pickle=False) as z:teacher=np.concatenate([z['tr'][:,j],z['te'][:,j]])
    x,names,cats,keys=static_features(pd.concat([tr,te],ignore_index=True),teacher);del teacher;gc.collect()
    oof=np.full(len(y),np.nan,dtype='float32');tests=np.zeros(len(te));details=[];seed=spec['seed']
    class WallLimit:
        def after_iteration(self,info):
            if time.monotonic()-start>spec['resources']['wall_limit_hours']*3600:raise TimeoutError('Preregistered CPU wall limit reached')
            return True
    for k in spec['primary_folds']:
        began=time.monotonic();fit=np.flatnonzero(folds!=k);val=np.flatnonzero(folds==k)
        inner,es=train_test_split(fit,test_size=.1,random_state=seed+k,stratify=y[fit])
        assert not np.intersect1d(fit,val).size and not np.intersect1d(inner,es).size
        a,b=encoded(keys,y,inner,es,seed+k)
        xa=cat_frame(np.column_stack([x[inner],a]),cats);xb=cat_frame(np.column_stack([x[es],b]),cats)
        model=CatBoostClassifier(**spec['params'])
        model.fit(xa,y[inner],cat_features=cats,eval_set=(xb,y[es]),early_stopping_rounds=spec['inner_ES_patience'],use_best_model=True,callbacks=[WallLimit()])
        best=model.get_best_iteration()+1;assert best>=1
        print(json.dumps({'fold':k,'stage':'inner_selection_complete','iterations':best,'seconds':time.monotonic()-began}),flush=True)
        del model,xa,xb,a,b;gc.collect()
        apply=np.concatenate([val,np.arange(len(tr),len(x))]);a,b=encoded(keys,y,fit,apply,seed+100+k)
        xa=cat_frame(np.column_stack([x[fit],a]),cats);xb=cat_frame(np.column_stack([x[apply],b]),cats)
        model=CatBoostClassifier(**{**spec['params'],'iterations':best})
        model.fit(xa,y[fit],cat_features=cats,callbacks=[WallLimit()]);p=model.predict_proba(xb)[:,1].astype('float32')
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        oof[val]=p[:len(val)];tests+=p[len(val):].astype('float64')/5
        folder=out/f'f{k}';folder.mkdir();model.save_model(str(folder/'model.cbm'))
        np.save(folder/'oof.npy',p[:len(val)]);np.save(folder/'test.npy',p[len(val):]);np.save(folder/'eval_ids.npy',ids[val])
        item={'fold':k,'iterations':best,'auc':float(roc_auc_score(y[val],p[:len(val)])),'seconds':time.monotonic()-began,
              'fit_ids_sha256':arr_sha256(ids[fit]),'eval_ids_sha256':arr_sha256(ids[val]),
              'inner_fit_ids_sha256':arr_sha256(ids[inner]),'inner_ES_ids_sha256':arr_sha256(ids[es]),
              'prediction_sha256':arr_sha256(p),'model_sha256':file_sha256(folder/'model.cbm')}
        save_json(item,folder/'report.json');details.append(item)
        save_json({'status':'RUNNING','complete_folds':details,'elapsed_seconds':time.monotonic()-start},out/'progress.json')
        print(json.dumps(item),flush=True);del model,xa,xb,a,b,p;gc.collect()
    assert np.isfinite(oof).all();tests=tests.astype('float32');np.save(out/'oof.npy',oof);np.save(out/'test.npy',tests)
    for p,h in original['protected_files'].items():assert file_sha256(p)==h,p
    save_json({'status':'COMPLETE_FIVE_FOLD_PRIMARY','run_git':run_git,'source_sha256':file_sha256(__file__),
               'prereg_sha256':file_sha256(protocol),'feature_source_sha256':spec['feature_code_sha256'],
               'oof_auc':float(roc_auc_score(y,oof)),'oof_sha256':arr_sha256(oof),'test_sha256':arr_sha256(tests),
               'folds':details,'train_ids_sha256':arr_sha256(ids),'test_ids_sha256':arr_sha256(te.id.to_numpy()),
               'CPU_wall_seconds':time.monotonic()-start,'GPU_hours':0,'verdict':'AWAITING_FROZEN_PAIRED_COMPARISON'},report)
    print('COMPLETE lean84 CatBoost five-fold CPU',flush=True)

if __name__=='__main__':main()
