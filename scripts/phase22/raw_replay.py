"""Verify our historical raw contexts and test two newly frozen v7 mixtures."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,subprocess,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.common import load_cached_parquet,file_sha256,arr_sha256,save_json,git_commit
from src.validation.folds import get_scheme
from src.validation.compare import logit
from src.validation.paired_auc import paired_auc
from src.validation.private_sim import RankedAUC,summarize
from src.submission.store import load_oof
from scripts.evaluate_sol_phase15_raw import verify_raw
from scripts.private_lb_simulator import make_partition

def main():
    started=time.monotonic();pr=Path('configs/phase22/own_raw_replay_20261019.json')
    assert file_sha256(pr)=='43cfe62ca785a2af7172da3163c79b7a8d47c1b78c07431cd82f83d729e76789'
    spec=json.loads(pr.read_text());assert spec==json.loads(subprocess.check_output(['git','show','1d7a1eb:'+pr.as_posix()]))
    output=Path('reports/phase22/own_raw_replay.json');assert not output.exists()
    assert file_sha256(spec['primary_report'])==spec['primary_report_sha256']
    assert file_sha256(spec['raw_oof_path'])==spec['raw_oof_file_sha256']
    tr,te=load_cached_parquet();y=tr.satisfaction.to_numpy(dtype='int8');ids=tr.id.to_numpy();folds=get_scheme('primary',y,ids).folds
    dh={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    raw=np.load(spec['raw_oof_path'],allow_pickle=False);proofs=[]
    for k in range(5):
        p,proof=verify_raw(k,'sol_tabpfn35_predict_guard' if k==0 else 'sol_phase15_raw',tr,te,ids,y,folds,dh)
        assert np.array_equal(p,raw[folds==k]);proofs.append(proof)
        print('Verified raw fold',k,flush=True)
    base=load_oof(spec['baseline']);z=logit(base);zr=logit(raw);cfg=spec['simulation']
    masks=[make_partition(y,folds,cfg['seed']+i*104729,'random_stratified',cfg['population']) for i in range(cfg['repeats'])]
    bs=RankedAUC(y,base);bu=[bs.auc(m==1) for m in masks];bp=[bs.auc(m==2) for m in masks];rows=[]
    for w in spec['weights']:
        p=expit((1-w)*z+w*zr).astype('float32');pa=paired_auc(y,p,base)
        d=np.array([roc_auc_score(y[folds==k],p[folds==k])-roc_auc_score(y[folds==k],base[folds==k]) for k in range(5)])
        se=float(d.std(ddof=1)/np.sqrt(5));sc=RankedAUC(y,p)
        sim=summarize([sc.auc(m==1)-b for m,b in zip(masks,bu)],[sc.auc(m==2)-b for m,b in zip(masks,bp)],pa['delta'])
        passed=bool(d.mean()>=1.5e-5 and d.mean()>=2.5*se and (d>0).sum()==5 and pa['one_sided_p']<=.01 and sim['private_p05']>=0)
        row={'name':f'own_raw_w{w:.2f}','weight':w,'pooled':pa,'paired_fold_deltas':d.tolist(),'mean_fold_delta':float(d.mean()),'fold_se':se,'fold_t':float(d.mean()/se),'positive_folds':int((d>0).sum()),'private80':sim,'logit_corr':float(np.corrcoef(z,logit(p))[0,1]),'spearman':float(spearmanr(base,p).statistic),'prediction_sha256':arr_sha256(p),'primary_gate_pass':passed,'verdict':'PRIMARY_PASS_REQUIRES_SHADOW_AND_TEST' if passed else 'NO_ADMISSION'}
        rows.append(row);print(json.dumps(row),flush=True)
    save_json({'status':'COMPLETE','git':git_commit(),'source_sha256':file_sha256(__file__),'prereg_sha256':file_sha256(pr),'raw_context_proofs':proofs,'raw_oof_sha256':arr_sha256(raw),'champion_sha256':arr_sha256(base),'candidates':rows,'GPU_hours':0,'CPU_wall_seconds':time.monotonic()-started,'new_submissions':0,'limitation':'Adaptive mechanism discovery after public-vector diagnostic; frozen primary and shadow remain necessary. Raw1 differs from public raw8, so a null result does not disprove raw8.'},output)

if __name__=='__main__':main()
