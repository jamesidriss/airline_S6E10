"""Controlled label perturbation of the published globally cross-fitted TE recipe."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import ast,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.common import load_cached_parquet,save_json,file_sha256,arr_sha256
from scripts.phase22.lean84 import encoded

def main():
    tr,_=load_cached_parquet(); y=tr.satisfaction.to_numpy(dtype='int8')
    keys=tr['Flight Distance'].astype(str); folds=np.empty(len(y),dtype='int8')
    for k,(_,b) in enumerate(StratifiedKFold(15,shuffle=True,random_state=42).split(np.zeros(len(y)),y)):folds[b]=k
    def published(labels):
        p=np.empty(len(labels),dtype='float64'); prior=labels.mean()
        for k in range(15):
            fit=folds!=k; val=~fit
            stats=pd.DataFrame({'key':keys[fit].to_numpy(),'label':labels[fit]}).groupby('key').label.agg(['sum','count'])
            mapping=(stats['sum']+20*prior)/(stats['count']+20)
            p[val]=keys[val].map(mapping).fillna(prior)
        return p
    before=published(y); changed=y.copy();changed[folds==0]=1-y[folds==0];after=published(changed)
    fit=np.flatnonzero(folds!=0); val=np.flatnonzero(folds==0)
    a,b=encoded({'te_0':keys},y,fit,val,20261019); c,d=encoded({'te_0':keys},changed,fit,val,20261019)
    assert np.array_equal(a,c) and np.array_equal(b,d)
    assert (before[fit]!=after[fit]).any()
    nbpath=Path('research/raw/phase22/jiatianzhang2_v4.ipynb');nb=json.loads(nbpath.read_text(encoding='utf-8'))
    cells={i:(''.join(c['source']) if isinstance(c['source'],list) else c['source']) for i,c in enumerate(nb['cells']) if c['cell_type']=='code'}
    for s in cells.values():ast.parse(s)
    result={'status':'PUBLISHED_PROTOCOL_REJECTED_AS_HONEST_OOF','notebook':'https://www.kaggle.com/code/jiatianzhang2/s6e10-15-fold-gpu-stacking-pipeline','version':4,'source_sha256':file_sha256(nbpath),'code_cells_parse':True,'nonempty_output_cells':sum(bool(c.get('outputs')) for c in nb['cells']),
            'controlled_perturbation':{'labels_changed':'Only public15fold0 evaluation labels, fixed original split','key':'Flight Distance','rows':len(y),'outer_fit_rows':len(fit),'outer_eval_rows':len(val),'FIT_TE_rows_changed':int((before[fit]!=after[fit]).sum()),'FIT_TE_maxabs_change':float(np.max(np.abs(before[fit]-after[fit]))),'EVAL_TE_rows_changed_through_global_prior':int((before[val]!=after[val]).sum()),'EVAL_TE_maxabs_change':float(np.max(np.abs(before[val]-after[val]))),'corrected_nested_FIT_rows_changed':int((a!=c).sum()),'corrected_nested_EVAL_rows_changed':int((b!=d).sum()),'fixed_fold_sha256':arr_sha256(folds)},
            'findings':[{'cell':8,'issue':'Global15fold TE prepared before outer model loop: FIT feature rows contain outer evaluation labels through other folds; global prior also includes held-out labels.'},{'cell':10,'issue':'All three tree learners select iterations using their outer evaluation fold.'},{'cell':12,'issue':'meta_clf.fit(S_oof,y) followed by predict_proba(S_oof) and scoring on y is in-sample meta evaluation.'},{'cell':12,'issue':'Published notebook stacks only three tree vectors; the 11-TFM fusion advertised in discussion is absent.'},{'cell':6,'issue':'Teacher source can be precomputed arrays without IDs/provenance, a local original-data model, or constant0.5 fallback; exact reported run path not established.'}],
            'interpretation':'These prove evaluation protocol defects, not the numerical size of score inflation and not misconduct. The reported0.96238 remains unreproduced. No public code was executed as training code.'}
    out=Path('reports/phase22/public_notebook_leakage_audit.json');assert not out.exists();save_json(result,out)
    print(json.dumps(result['controlled_perturbation']),flush=True)

if __name__=='__main__':main()
