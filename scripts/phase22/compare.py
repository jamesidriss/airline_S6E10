"""Frozen comparisons. Imported vectors are diagnostics, never eligible candidates."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse
import json
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.common import load_cached_parquet,arr_sha256,file_sha256,save_json,git_commit
from src.submission.store import load_oof
from src.validation.folds import get_scheme
from src.validation.compare import logit
from src.validation.paired_auc import paired_auc,holm_adjust
from src.validation.private_sim import RankedAUC,summarize
from scripts.private_lb_simulator import make_partition
from scripts.phase22.lean84 import PROTOCOL,PROTOCOL_SHA

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['external','fresh']);mode=ap.parse_args().mode
    started=time.monotonic(); assert file_sha256(PROTOCOL)==PROTOCOL_SHA
    spec=json.loads(PROTOCOL.read_text()); assert spec==json.loads(subprocess.check_output(['git','show','80d9eea:'+PROTOCOL.relative_to(ROOT).as_posix()]))
    out=ROOT/f'reports/phase22/{mode}_comparison.json'; assert not out.exists()
    tr,te=load_cached_parquet(); y=tr.satisfaction.to_numpy(dtype='int8'); ids=tr.id.to_numpy()
    folds=get_scheme('primary',y,ids).folds; base=load_oof(spec['champion']); zbase=logit(base)
    cfg=spec['simulations']; masks=[make_partition(y,folds,cfg['seed']+i*104729,'random_stratified',cfg['population']) for i in range(cfg['repeats'])]
    assert all(int((m==1).sum())==cfg['public_rows'] and int((m==2).sum())==cfg['private_rows'] for m in masks)
    # make_partition returns the full disjoint 0/1/2 assignment on OOF rows.
    scorer=RankedAUC(y,base); pub0=[scorer.auc(m==1) for m in masks]; priv0=[scorer.auc(m==2) for m in masks]
    baseline_folds=[roc_auc_score(y[folds==k],base[folds==k]) for k in range(5)]
    candidates=[]; source_audit=[]; compatibility=None
    if mode=='external':
        cfgx=spec['external']; root=Path(cfgx['directory']); zs={}
        for name in cfgx['models']:
            files=[root/f'{prefix}_{name}.npy' for prefix in ['oof','test']]
            for p in files: assert file_sha256(p)==cfgx['file_sha256'][p.name]
            p,q=[np.load(f,allow_pickle=False) for f in files]
            assert p.shape==(len(y),) and q.shape==(len(te),) and np.isfinite(p).all() and np.isfinite(q).all()
            assert ((p>=0)&(p<=1)).all() and ((q>=0)&(q<=1)).all()
            zs[name]=logit(p)
            source_audit.append({'name':name,'oof_auc_assuming_official_row_order':float(roc_auc_score(y,p)),
                                 'oof_shape':list(p.shape),'test_shape':list(q.shape),'dtype':str(p.dtype),
                                 'oof_sha256':arr_sha256(p),'test_sha256':arr_sha256(q),
                                 'oof_range':[float(p.min()),float(p.max())],'row_ids_present':False})
            for w in cfgx['weights']: candidates.append((name+f'_w{w:.2f}',expit((1-w)*zbase+w*zs[name]).astype('float32')))
        tfm=np.mean([z for name,z in zs.items() if name not in ['realmlp','xgboost']],axis=0)
        for w in cfgx['equal_tfm9_weights']: candidates.append((f'equal_tfm9_w{w:.2f}',expit((1-w)*zbase+w*tfm).astype('float32')))
        reported=np.empty(len(y),dtype='int8')
        for k,(_,v) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y)): reported[v]=k
        table=np.array([[int(((folds==i)&(reported==j)).sum()) for j in range(5)] for i in range(5)])
        a,b=linear_sum_assignment(-table)
        compatibility={'declared_external_scheme':'StratifiedKFold5_seed42','local_scheme':'immutable_primary_seed20261010',
                       'contingency':table.tolist(),'best_label_permutation_match_fraction':float(table[a,b].sum()/len(y)),
                       'reported_fold_assignment_sha256':arr_sha256(reported),'source_producer_context_manifest_present':False,
                       'verdict':'INCOMPATIBLE_FOR_PRIMARY_OR_SHADOW_CERTIFICATION'}
    else:
        report=json.loads(Path('reports/phase22/lean84_primary.json').read_text())
        p=np.load('artifacts/phase22/lean84/oof.npy',allow_pickle=False)
        assert arr_sha256(p)==report['oof_sha256']; z=logit(p)
        for w in spec['fresh_CPU']['portfolio_weights']: candidates.append((f'lean84_w{w:.2f}',expit((1-w)*zbase+w*z).astype('float32')))
    rows=[]
    for name,p in candidates:
        pair=paired_auc(y,p,base); d=np.array([roc_auc_score(y[folds==k],p[folds==k])-baseline_folds[k] for k in range(5)])
        se=float(d.std(ddof=1)/np.sqrt(5)); score=RankedAUC(y,p)
        pub=[score.auc(m==1)-b for m,b in zip(masks,pub0)]; priv=[score.auc(m==2)-b for m,b in zip(masks,priv0)]
        row={'name':name,'pooled':pair,'paired_fold_deltas':d.tolist(),'mean_fold_delta':float(d.mean()),'fold_se':se,
             'fold_t':float(d.mean()/se) if se else None,'positive_folds':int((d>0).sum()),
             'private80':summarize(pub,priv,pair['delta']),'logit_corr':float(np.corrcoef(zbase,logit(p))[0,1]),
             'spearman':float(spearmanr(base,p).statistic),'prediction_sha256':arr_sha256(p)}
        rows.append(row); print(json.dumps({'name':name,'delta':pair['delta'],'folds_positive':row['positive_folds']}),flush=True)
    nominations=[]
    for r,pval in zip(rows,holm_adjust([r['pooled']['one_sided_p'] for r in rows])):
        r['holm_p']=float(pval)
        passed=r['mean_fold_delta']>=1.5e-5 and r['mean_fold_delta']>=2.5*r['fold_se'] and r['positive_folds']==5 and pval<=.05 and r['private80']['private_p05']>=0
        r['numerical_gate_pass']=bool(passed); r['eligible_for_confirmation']=bool(passed and mode=='fresh')
        r['verdict']='INELIGIBLE_EXTERNAL_DIAGNOSTIC' if mode=='external' else ('NOMINATE_MATCHED_SHADOW' if passed else 'NO_ADMISSION')
        if r['eligible_for_confirmation']: nominations.append(r['name'])
    for p,h in spec['protected_files'].items(): assert file_sha256(p)==h
    save_json({'status':'COMPLETE','mode':mode,'git':git_commit(),'prereg_sha256':PROTOCOL_SHA,'source_sha256':file_sha256(__file__),
               'candidates':rows,'nominations':nominations,'source_audit':source_audit,'fold_compatibility':compatibility,
               'simulation':spec['simulations'],'mask_sha256':[arr_sha256(m) for m in masks],
               'baseline_auc':float(roc_auc_score(y,base)),'baseline_sha256':arr_sha256(base),
               'CPU_wall_seconds':time.monotonic()-started,'GPU_hours':0,'new_submissions':0,
               'limitation':'Conditional saved-OOF diagnostics; repeated model selection and fit variation excluded. Imported scores require producer and row-ID evidence; numerically strong does not mean certified.'},out)
    print(json.dumps({'complete':mode,'nominations':nominations,'seconds':time.monotonic()-started}),flush=True)

if __name__=='__main__': main()
