"""Frozen eight-comparison-family gate for the clean CPU CatBoost challenger."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,time,subprocess
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.common import load_cached_parquet,file_sha256,arr_sha256,save_json,git_commit
from src.validation.folds import get_scheme
from src.validation.compare import logit
from src.validation.paired_auc import paired_auc
from src.validation.private_sim import RankedAUC,summarize
from src.submission.store import load_oof
from scripts.private_lb_simulator import make_partition

def main():
    started=time.monotonic();pr=Path('configs/phase22/lean84_cat_20261019.json')
    assert file_sha256(pr)=='b3640e52bb09d7b923b0df326759b8799cdae8fe1e3cb4babbadd5d09c48ca03'
    spec=json.loads(pr.read_text());assert spec==json.loads(subprocess.check_output(['git','show','479b642:'+pr.as_posix()]))
    original=json.loads(Path(spec['feature_protocol']).read_text())
    output=Path('reports/phase22/cat_comparison.json');assert not output.exists()
    train,test=load_cached_parquet();y=train.satisfaction.to_numpy(dtype='int8');ids=train.id.to_numpy();folds=get_scheme('primary',y,ids).folds
    report=json.loads(Path('reports/phase22/lean84_cat_primary.json').read_text());root=Path('artifacts/phase22/lean84_cat')
    p=np.load(root/'oof.npy',allow_pickle=False);q=np.load(root/'test.npy',allow_pickle=False)
    assert arr_sha256(p)==report['oof_sha256'] and arr_sha256(q)==report['test_sha256']
    assert arr_sha256(ids)==report['train_ids_sha256'] and arr_sha256(test.id.to_numpy())==report['test_ids_sha256']
    assert file_sha256('scripts/phase22/lean84_cat.py')==report['source_sha256']
    for f in report['folds']:
        k=f['fold'];assert f['fit_ids_sha256']==arr_sha256(ids[folds!=k]) and f['eval_ids_sha256']==arr_sha256(ids[folds==k])
        assert file_sha256(root/f'f{k}/model.cbm')==f['model_sha256']
        assert np.array_equal(np.load(root/f'f{k}/oof.npy',allow_pickle=False),p[folds==k])
    base=load_oof(original['champion']);z=logit(base);zc=logit(p);cfg=original['simulations']
    masks=[make_partition(y,folds,cfg['seed']+i*104729,'random_stratified',cfg['population']) for i in range(cfg['repeats'])]
    bs=RankedAUC(y,base);bu=[bs.auc(m==1) for m in masks];bp=[bs.auc(m==2) for m in masks];rows=[]
    for w in spec['portfolio_weights']:
        candidate=expit((1-w)*z+w*zc).astype('float32');pa=paired_auc(y,candidate,base)
        d=np.array([roc_auc_score(y[folds==k],candidate[folds==k])-roc_auc_score(y[folds==k],base[folds==k]) for k in range(5)])
        se=float(d.std(ddof=1)/np.sqrt(5));sc=RankedAUC(y,candidate)
        sim=summarize([sc.auc(m==1)-b for m,b in zip(masks,bu)],[sc.auc(m==2)-b for m,b in zip(masks,bp)],pa['delta'])
        passed=bool(d.mean()>=1.5e-5 and d.mean()>=2.5*se and (d>0).sum()==5 and pa['one_sided_p']<=spec['alpha_per_comparison'] and sim['private_p05']>=0)
        row={'name':f'lean84_cat_w{w:.2f}','weight':w,'pooled':pa,'paired_fold_deltas':d.tolist(),'mean_fold_delta':float(d.mean()),'fold_se':se,'fold_t':float(d.mean()/se),'positive_folds':int((d>0).sum()),'private80':sim,'logit_corr':float(np.corrcoef(z,logit(candidate))[0,1]),'prediction_sha256':arr_sha256(candidate),'primary_gate_pass':passed,'verdict':'NOMINATE_MATCHED_SHADOW' if passed else 'NO_ADMISSION'}
        rows.append(row);print(json.dumps(row),flush=True)
    save_json({'status':'COMPLETE','git':git_commit(),'source_sha256':file_sha256(__file__),'prereg_sha256':file_sha256(pr),'candidates':rows,'nominations':[r['name'] for r in rows if r['primary_gate_pass']],'multiplicity':'Bonferroni across8 eligible phase22 comparisons; primary adaptive discovery, shadow required','GPU_hours':0,'CPU_wall_seconds':time.monotonic()-started,'new_submissions':0},output)

if __name__=='__main__':main()
