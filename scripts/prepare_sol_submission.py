"""Build once from certified vectors after an explicit scientific decision."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import REPORTS,SUBMISSIONS,arr_sha256,file_sha256,git_commit,load_cached_parquet,save_json
from src.submission.make import build
from src.validation.compare import logit,spearman
from scripts.assemble_sol_foundation_test import verify_primary,checked_probability
from scripts.audit_sol_state import reconstruct_v5,pair_diagnostic


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary-report',default='reports/sol_route_aux10_primary_final.json')
    ap.add_argument('--test-certificate',default='reports/sol_route_aux10_primary_final_test_certificate.json')
    ap.add_argument('--scorecard',default='reports/sol_route_aux10_recovery_scorecard.json')
    ap.add_argument('--decision',required=True)
    ap.add_argument('--name',default='v6_sol_tabpfn_route_aux10')
    args=ap.parse_args()
    if not args.name or Path(args.name).name!=args.name or args.name in ('.','..'):
        raise ValueError('Submission name must be a filename stem')
    output=REPORTS/(args.name+'_pre_result.json')
    if output.exists() or (SUBMISSIONS/(args.name+'.csv')).exists():raise FileExistsError('Preserve banked submission and manifest')
    decision=json.loads(Path(args.decision).read_text(encoding='utf-8'))
    score=json.loads(Path(args.scorecard).read_text(encoding='utf-8'))
    assert decision['verdict']=='SUBMIT_FROZEN_CERTIFIED_CANDIDATE'
    assert decision['scorecard_sha256']==file_sha256(args.scorecard)
    assert decision['candidate_name']==args.name
    assert score['status']=='COMPLETE_FIXED_PORTFOLIO_ROBUSTNESS_EVIDENCE_REQUIRES_SCIENTIFIC_DECISION'
    assert score['primary_report_sha256']==file_sha256(args.primary_report)
    assert score['test_certificate_sha256']==file_sha256(args.test_certificate)
    assert all(file_sha256(p)==h for p,h in score['source_sha256'].items())
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();test_ids=te.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    data={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    primary,vectors,folds=verify_primary(args.primary_report,ids,y,data)
    certificate=json.loads(Path(args.test_certificate).read_text(encoding='utf-8'))
    assert certificate['status']=='CERTIFIED_FIXED_OOF_TEST_REQUIRES_ROBUSTNESS_SCORECARD'
    assert certificate['primary_report_sha256']==file_sha256(args.primary_report)
    assert certificate['oof_sha256']==score['oof_sha256'] and certificate['test_sha256']==score['test_sha256']
    assert certificate['weights']=={'route':.5,'equal_logit_auxiliary10':.5}
    assert certificate['data_sha256']==data
    root=Path(certificate['test_artifact_root'])
    prediction=checked_probability(root/'candidate.npy',root/'test_ids.npy',certificate['test_sha256']['candidate'],test_ids)
    legacy3,legacy5,_=reconstruct_v5(y,folds)
    banked={}
    for name,reference in [('v3_final',legacy3),('v4_fulldata',legacy3),('v5_aux_cross',legacy5)]:
        path=SUBMISSIONS/(name+'.csv');old=pd.read_csv(path)
        assert list(old)==['id','satisfaction'] and np.array_equal(old.id,test_ids)
        p=old.satisfaction.to_numpy()
        assert p.shape==prediction.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        banked[name]={'submission_sha256':file_sha256(path),
            'test_logit_correlation':float(np.corrcoef(logit(prediction),logit(p))[0,1]),
            'test_spearman':spearman(prediction,p),
            'legacy_oof_auc_diagnostic':float(roc_auc_score(y,reference)),
            'oof_delta_diagnostic':float(roc_auc_score(y,vectors['candidate'])-roc_auc_score(y,reference)),
            'oof_spearman_diagnostic':spearman(vectors['candidate'],reference),
            'pair_comparison_diagnostic':pair_diagnostic(y,vectors['candidate'],reference),
            'eligibility':'Legacy contracts defective; v4 uses v3 OOF proxy. This is diagnostic only.'}
    notes='Frozen50/50 route/strict auxiliary10; primary5 verified; predeclared two-shadow-fold and seed diagnostic; route test probability-average of5 primary FIT contexts; no public tuning'
    path=build(prediction,args.name,notes=notes,oof_auc=primary['pooled_oof_auc'],
               members=['route_primary_context_probability_average','strict_auxiliary10_equal_logit'])
    csv=pd.read_csv(path)
    assert list(csv)==['id','satisfaction'] and np.array_equal(csv.id,test_ids)
    assert np.allclose(csv.satisfaction.to_numpy(),prediction.astype('float64'),rtol=0,atol=1e-15)
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'git':git_commit(),'candidate':args.name,
        'status':'PRE_RESULT_CERTIFIED_SUBMISSION_READY_NOT_YET_UPLOADED','submission_path':str(path),
        'submission_sha256':file_sha256(path),'test_prediction_sha256':arr_sha256(prediction),
        'test_ids_sha256':arr_sha256(test_ids),'test_rows':len(test_ids),
        'primary_report_sha256':file_sha256(args.primary_report),'test_certificate_sha256':file_sha256(args.test_certificate),
        'scorecard_sha256':file_sha256(args.scorecard),'decision_sha256':file_sha256(args.decision),
        'oof_auc':primary['pooled_oof_auc'],'folds':primary['folds'],
        'reason':decision['reason'],'intended_role':decision['intended_role'],
        'independent_confirmation_scope':score['independent_confirmation_scope'],
        'test_policy':certificate['foundation_test_policy'],'banked_comparisons':banked,
        'schema_preflight':'PASS; exact columns, complete299844 rows, ordered IDs, finite probabilities, preserved frozen prediction',
        'public_result_interpretation':'Measures transfer on about20percent of test; neither proves private ranking nor justifies tuning weights from tiny public differences.'}
    save_json(manifest,output)
    print(json.dumps({'candidate':args.name,'rows':len(test_ids),'csv_sha256':manifest['submission_sha256'],'status':manifest['status']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
