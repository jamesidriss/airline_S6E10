"""Fixed OOF-only stress diagnostics; no test certificate or submission eligibility."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json
from src.validation.compare import logit
from src.validation.private_sim import RankedAUC, summarize
from scripts.phase16_common import bank
from scripts.phase17_primary_io import load_primary
from scripts.private_lb_simulator import make_partition, segments
from scripts.audit_sol_state import reconstruct_v5

DEFINITIONS_SHA='7003ac1c9a0594d3e28f152cc9b556704ef9f1dc1545a1486dd76b164ed64472'


def verify_definitions(definitions):
    assert len(definitions)==820
    assert hashlib.sha256(json.dumps(definitions,sort_keys=True).encode()).hexdigest()==DEFINITIONS_SHA


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary-tag',required=True);ap.add_argument('--tag',required=True)
    args=ap.parse_args();assert Path(args.tag).name==args.tag
    output=Path('reports')/(args.tag+'.json');root=ARTIFACTS/args.tag
    assert not output.exists() and not root.exists(), 'Preserve prior diagnostics'
    tr,_,y,folds,vectors,_,proof=bank();ids=tr.id.to_numpy()
    primary,p=load_primary(args.primary_tag,ids)
    assert primary['bank']==proof and primary['scope_sha256']==file_sha256('research/phase17_scope_20261009.json')
    candidates={'v6':vectors['candidate'],'unchanged_aux10':vectors['aux10'],'original_route':vectors['route'],
                args.primary_tag:p['candidate'],args.primary_tag+'_route_component':p['route']}
    old_path=Path('reports/sol_private_sim_recovery.json');old=json.loads(old_path.read_text())
    assert old['portfolio_contract_sha256']==proof['primary_report_sha256']
    assert old['fold_sha256']==arr_sha256(folds) and old['ids_sha256']==arr_sha256(ids)
    assert all(file_sha256(s)==h for s,h in old['source_sha256'].items())
    definitions=json.loads(Path(old['split_and_draw_artifact']).read_text())['definitions'];verify_definitions(definitions)
    assert old['split_definitions_sha256']==DEFINITIONS_SHA
    _,legacy,_=reconstruct_v5(y,folds);stress=segments(tr,legacy)
    scorers={n:RankedAUC(y,logit(v)) for n,v in candidates.items()}
    full={n:float(roc_auc_score(y,v)) for n,v in candidates.items()}
    assert all(abs(scorers[n].auc(np.ones(len(y)))-full[n])<1e-14 for n in candidates)
    draws={n:[] for n in candidates}
    for d in definitions:
        assert d['population']==299844
        mask=make_partition(y,folds,d['seed'],d['mode'],d['population'],stress.get(d['segment']),fold=d['public_fold'])
        assert arr_sha256(mask)==d['mask_sha256']
        public,private=mask==1,mask==2
        assert int(public.sum())==d['public_rows'] and int(private.sum())==d['private_rows']
        pb,pr=scorers['v6'].auc(public),scorers['v6'].auc(private)
        for name,scorer in scorers.items():
            draws[name].append({'mode':d['mode'],'segment':d['segment'],
                'public_delta':scorer.auc(public)-pb,'private_delta':scorer.auc(private)-pr})
        if (d['run']+1)%100==0:print(f'OOF-only fixed stress {d["run"]+1}/820',flush=True)
    summary={}
    for name,rows in draws.items():
        delta=full[name]-full['v6']
        def describe(chosen):return summarize([r['public_delta'] for r in chosen],[r['private_delta'] for r in chosen],delta)
        summary[name]={'oof_auc':full[name],'pooled_delta_vs_v6':delta,'all_820':describe(rows),
            'modes':{m:describe([r for r in rows if r['mode']==m]) for m in ('random_stratified','fold_aware','segment_stressed')},
            'stress_segments':{k:describe([r for r in rows if r['segment']==k]) for k in stress},
            'test_correlation_vs_v6':None,'eligibility':'OOF diagnostic only; independent confirmation and test certificate still required'}
    root.mkdir();save_json({'definitions':definitions,'draws':draws},root/'draws.json')
    result={'utc':datetime.now(timezone.utc).isoformat(),'git':git_commit(),'bank':proof,
        'status':'COMPLETE_OOF_ONLY_820_STRESS_NOT_SUBMISSION_ELIGIBLE','primary_tag':args.primary_tag,
        'primary_report_sha256':file_sha256(Path('reports')/(args.primary_tag+'_primary.json')),
        'reference':'v6','total_draws':820,'every_original_mask_hash_matched':True,
        'split_definitions_sha256':DEFINITIONS_SHA,'old_report_sha256':file_sha256(old_path),'candidates':summary,
        'prediction_sha256':{n:arr_sha256(v) for n,v in candidates.items()},
        'draws_artifact':str(root/'draws.json'),'draws_file_sha256':file_sha256(root/'draws.json'),
        'source_sha256':{s:file_sha256(s) for s in ('scripts/diagnose_phase17_oof_private.py','scripts/private_lb_simulator.py',
            'src/validation/private_sim.py','scripts/phase17_primary_io.py','scripts/audit_sol_state.py')},
        'test_certified':False,'independent_confirmation_passed':False,'submission_eligible':False,
        'limitation':'Conditional resampling of existing OOF only; no private labels, refitting uncertainty or unknown shift. Fixed legacyv5 defines confidence partitions, never an eligible model score.'}
    save_json(result,output);print(json.dumps(summary[args.primary_tag]['all_820']))


if __name__=='__main__':main()
