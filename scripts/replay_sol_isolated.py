"""Run fixed auxiliary roles sequentially, one fresh process per fold.

Avoid carrying native allocator memory between folds. The child runner performs
every feature/ID/source check; this orchestrator never changes model settings.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',default='reports/sol_clean_aux10/frozen_roles.json')
    ap.add_argument('--folds',default='0,1')
    ap.add_argument('--tag',default='sol_clean_aux10_isolated')
    ap.add_argument('--resume-from')
    ap.add_argument('--confirmation-shadow',action='store_true')
    args=ap.parse_args()
    plan=json.loads((ROOT/args.plan).read_text(encoding='utf-8'))
    roles=[r for r in plan['roles'] if r['aux_arm']]
    assert len(roles)==10 and len({r['member'] for r in roles})==10
    assert all(r['scheme']=='primary' and r['view']=='full' and r['es_seed_policy']=='1' for r in roles)
    ks=list(map(int,args.folds.split(','))); assert len(set(ks))==len(ks) and set(ks)<=set(range(5))
    folder=ROOT/'reports'/args.tag; folder.mkdir(parents=True,exist_ok=True)
    log=folder/'isolated_workers.log'
    if log.exists():
        raise FileExistsError('Preserve the existing worker log; use a new tag and --resume-from')
    completed=[]
    with log.open('x',encoding='utf-8') as output:
        for role in roles:
            for k in ks:
                command=[sys.executable,'-u','scripts/replay_sol_classical.py',
                    '--members',role['member'],'--folds',str(k),'--tag',args.tag]
                if args.resume_from:
                    command+=['--resume-from',args.resume_from]
                if args.confirmation_shadow:
                    command+=['--confirmation-shadow']
                progress={'utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNING',
                    'member':role['member'],'fold':k,'completed':completed,'command':command}
                output.write(json.dumps(progress)+'\n'); output.flush()
                child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                progress['pid']=child.pid
                (folder/'isolated_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                print(f'{role["member"]} fold{k}: worker{child.pid}',flush=True)
                try:
                    code=child.wait()
                except KeyboardInterrupt:
                    child.terminate(); child.wait(timeout=30)
                    raise
                if code:
                    progress.update(status='STOPPED_REVIEW_CHILD_REPORT',exit_code=code)
                    (folder/'isolated_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                    return code
                completed.append({'member':role['member'],'fold':k})
    (folder/'isolated_progress.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'status':'COMPLETED','completed':completed,'scope':'Only requested folds; no full OOF claim'},indent=2),encoding='utf-8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
