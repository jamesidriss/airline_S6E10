"""Refit one frozen auxiliary role after verified full primary admission."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,load_cached_parquet
from src.validation.folds import get_scheme
from scripts.replay_sol_classical import frozen_roles,resolved_params,complete_role
from scripts.evaluate_sol_foundation import classical


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary-report',default='reports/sol_route_aux10_primary.json')
    ap.add_argument('--tag',default='sol_clean_aux10_primary')
    ap.add_argument('--member',required=True)
    args=ap.parse_args()
    report=json.loads(Path(args.primary_report).read_text(encoding='utf-8'))
    assert report['scope']=='full primary OOF' and report['scheme']=='primary'
    assert report['admission_gate_against_clean_auxiliary_stack']['admit']
    roles=[r for r in frozen_roles() if r['aux_arm']]
    role=next(r for r in roles if r['member']==args.member)
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    folds=get_scheme('primary',y,ids).folds
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    assert report['data_sha256']==data_hash and report['fold_sha256']==arr_sha256(folds)
    assert report['ordered_train_ids_sha256']==arr_sha256(ids)
    assert sorted(r['fold'] for r in report['folds'])==list(range(5))
    for k in range(5):
        _,proof=classical(k,args.tag,[role],ids,folds,y,data_hash)
        expected=next(r for r in report['folds'] if r['fold']==k)
        expected=next(p for p in expected['classical_certificates'] if p['member']==role['member'])
        assert proof[0]==expected
    first=json.loads((REPORTS/args.tag/f'{role["member"]}_f0.json').read_text(encoding='utf-8'))
    complete_role(role,tr,te,y,ids,folds,ARTIFACTS/args.tag,REPORTS/args.tag,
        resolved_params(role),True,first['contract']['source_sha256'],data_hash)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
