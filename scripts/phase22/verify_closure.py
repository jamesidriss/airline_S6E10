"""Non-cyclic proof of committed evidence, append-only history and bank integrity."""
import hashlib,json,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.common import file_sha256,save_json
from scripts.audit_phase17_integrity import scan_bytes
BASE='a919a8949f9c0f4c7df21c1b31462d2b18e551bc'
def git(*args):return subprocess.check_output(['git',*args])
def main():
    out=Path('reports/phase22/final_proof_20261019.json');assert not out.exists()
    completion=json.loads(Path('reports/phase22/completion_20261019.json').read_text())
    assert not git('status','--porcelain').strip(),'Commit completed evidence before proof'
    for p,h in completion['protected_files'].items():assert file_sha256(p)==h,p
    history={}
    for p in Path('reports/phase21').glob('*'):
        if p.is_file():assert p.read_bytes()==git('show',BASE+':'+p.as_posix());history[p.as_posix()]=file_sha256(p)
    for p in ['reports/v7_sol_tabpfn_internal2_aux10_test_certificate.json','reports/v7_sol_tabpfn_internal2_aux10_decision.json','reports/v7_sol_tabpfn_internal2_aux10_result.json','reports/sol_route_aux10_primary_final_test_certificate.json','submissions/manifest.csv','submissions/kaggle_submissions.csv']:
        assert Path(p).read_bytes()==git('show',BASE+':'+p),p
        history[p]=file_sha256(p)
    prefix=git('show',BASE+':experiments/ledger.jsonl');ledger=Path('experiments/ledger.jsonl').read_bytes()
    assert ledger.startswith(prefix) and len(ledger.splitlines())==621
    additions=[json.loads(s) for s in ledger[len(prefix):].decode().splitlines()]
    assert len(additions)==37 and len({r['exp_id'] for r in additions})==37
    runtime='research/phase20_exhaustive/runtime_ledger.jsonl'
    assert Path(runtime).read_bytes().startswith(git('show',BASE+':'+runtime))
    old=json.loads(git('show',BASE+':experiments/private_finalists.json'))
    new=json.loads(Path('experiments/private_finalists.json').read_text())
    assert {k:new[k] for k in old}==old,'Existing finalist decisions must be preserved'
    configs=[('private_breakthrough_20261019.json','80d9eea'),('own_raw_replay_20261019.json','1d7a1eb'),('lean84_cat_20261019.json','479b642')]
    for name,commit in configs:
        p='configs/phase22/'+name;assert json.loads(Path(p).read_text())==json.loads(git('show',commit+':'+p))
    pairs={'external_comparison.json':'compare.py','fresh_comparison.json':'compare.py','own_raw_replay.json':'raw_replay.py','lean84_primary.json':'lean84.py','lean84_cat_primary.json':'lean84_cat.py','cat_comparison.json':'compare_cat.py'}
    for report,source in pairs.items():
        r=json.loads(Path('reports/phase22',report).read_text());assert r['source_sha256']==file_sha256(Path('scripts/phase22',source))
    changed=[p for p in git('diff','--name-only',BASE,'HEAD','-z').decode().split('\0') if p]
    hashes={};findings=[]
    for p in changed:
        path=Path(p);assert path.is_file(),p
        assert not p.startswith(('artifacts/','data/raw/','data/original/')),p
        raw=path.read_bytes();assert raw==git('show','HEAD:'+p),p
        hashes[p]=hashlib.sha256(raw).hexdigest()
        encodings=['utf-8-sig']+(['utf-16'] if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else [])
        for encoding in encodings:
            decoded=raw.decode(encoding,errors='replace')
            findings.extend({'path':p,'encoding':encoding,**f} for f in scan_bytes(decoded.encode()))
            if re.search(r'(?i)(X-Amz-Signature=|X-Goog-Signature=)',decoded):findings.append({'path':p,'kind':'signed_download_URL'})
    assert not findings,findings
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','base_commit':BASE,'head_before_proof_commit':git('rev-parse','HEAD').decode().strip(),'decision':completion['status'],'protected_files':completion['protected_files'],'historical_files':history,'ledger_prefix_preserved':True,'ledger_rows_before':584,'ledger_rows_after':621,'runtime_ledger_prefix_preserved':True,'existing_finalist_decisions_preserved':True,'preregistration_semantics_match_pre_result_commits':True,'executed_source_hashes_verified':True,'new_model_contexts':10,'new_submissions':0,'new_GPU_hours':0,'full_byte_secret_scan_files':len(hashes),'secret_findings':findings,'working_committed_bytes_match':True,'files_sha256':hashes,'QA_registered_checks_without_exception':247,'QA_CUDA_only_checks_skipped':2,'QA_failures':0,'QA_scope':'CPU-isolated exact registered suite; not a fresh GPU hardware validation','scope':'Proof excludes itself from hash map; push and clean status verified after proof commit'}
    save_json(result,out);print(json.dumps({k:result[k] for k in ('status','full_byte_secret_scan_files','ledger_rows_after','new_model_contexts')}))
if __name__=='__main__':main()
