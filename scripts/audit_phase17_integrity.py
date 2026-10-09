"""Audit historical evidence, append-only history, banked models, and committed bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import file_sha256, git_commit, save_json
from scripts.phase16_common import bank

BASELINE = '673a6c8783d83dfbbfd680bfbc7036de07568bc3'


def git_bytes(*args):
    return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL)


def scan_bytes(raw):
    text = raw.decode('utf-8', errors='replace')
    patterns = {'openai_token': r'\bsk-[A-Za-z0-9_-]{24,}',
                'huggingface_token': r'\bhf_[A-Za-z0-9]{24,}',
                'kaggle_token': r'\bKGAT_[A-Za-z0-9_-]{20,}',
                'aws_access_key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
                'github_token': r'\bgh[pousr]_[A-Za-z0-9]{30,}',
                'credential_field': r'"(?:api[_-]?key|token|password)"\s*:\s*"[A-Za-z0-9+/=_-]{24,}"',
                'kaggle_legacy_key': r'"key"\s*:\s*"[0-9a-fA-F]{32}"',
                'private_key': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'}
    return [{'kind': name, 'line': text.count('\n', 0, m.start())+1}
            for name, pattern in patterns.items() for m in re.finditer(pattern, text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    output = Path('reports')/(args.tag+'.json')
    if output.exists():
        raise FileExistsError('Preserve previous integrity evidence')
    recovery = json.loads(Path('reports/phase17_recovery_20261009.json').read_text())
    old_path = Path('reports/phase16_final_proof_index_20261009.json')
    assert file_sha256(old_path) == recovery['phase16_proof_sha256']
    old = json.loads(old_path.read_text()); dynamic = set(recovery['archived_dynamic_paths']); verified = {}
    for path, expected in old['evidence_file_sha256'].items():
        actual = Path('research/phase17_baseline_snapshots')/path.replace('/', '__') if path in dynamic else Path(path)
        assert file_sha256(actual) == expected, f'Historical evidence mismatch: {path}'
        verified[path] = {'verified_path': str(actual), 'sha256': expected}
    snapshots = Path('research/phase17_baseline_snapshots')
    for path in dynamic:
        assert (snapshots/path.replace('/', '__')).read_bytes() == git_bytes('show', BASELINE+':'+path)
    baseline_ledger = (snapshots/'experiments__ledger.jsonl').read_bytes()
    ledger = Path('experiments/ledger.jsonl').read_bytes()
    assert ledger.startswith(baseline_ledger), 'Historical ledger bytes changed'
    new_rows = [json.loads(row) for row in ledger[len(baseline_ledger):].decode('utf-8').splitlines() if row]
    assert len({r['exp_id'] for r in new_rows}) == len(new_rows), 'Duplicate Phase17 experiment IDs'
    _, _, _, _, _, _, proof = bank()
    assert proof == recovery['bank']
    tracked = git_bytes('ls-files', '-z').decode().split('\0')
    untracked = git_bytes('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    findings = []
    for path in sorted(set(tracked+untracked)-{''}):
        p = Path(path)
        if p.is_file() and p.stat().st_size < 32*1024**2 and p.suffix.lower() not in ('.png', '.jpg', '.pdf', '.zip'):
            findings.extend({'path': path, **f} for f in scan_bytes(p.read_bytes()))
    changed = git_bytes('diff', '--name-only', BASELINE, 'HEAD', '-z').decode().split('\0')
    committed_findings = []; byte_mismatches = []; checked = 0
    for path in changed:
        if not path or not Path(path).is_file():
            continue
        raw = git_bytes('show', 'HEAD:'+path); checked += 1
        if raw != Path(path).read_bytes():
            byte_mismatches.append(path)
        committed_findings.extend({'path': path, **f} for f in scan_bytes(raw))
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'git': git_commit(), 'baseline_commit': BASELINE,
        'status': 'PHASE17_INTEGRITY_FAIL' if findings or committed_findings or byte_mismatches else 'PHASE17_INTEGRITY_PASS', 'bank': proof, 'historical_files_verified': len(verified),
        'historical_evidence': verified, 'all_dynamic_snapshots_match_baseline_git': True,
        'ledger_prefix_bitexact': True, 'new_ledger_rows': len(new_rows),
        'new_ledger_prefix_sha256': hashlib.sha256(ledger).hexdigest(),
        'secret_scan_files': len(set(tracked+untracked)-{''}), 'secret_findings': findings,
        'changed_git_blobs_scanned': checked, 'git_blob_secret_findings': committed_findings,
        'committed_working_byte_mismatches': byte_mismatches,
        'source_sha256': file_sha256(__file__),
        'scope': 'Current tracked/untracked text and every changed committed blob; excluded local credentials and bulk data are never read or printed'}
    save_json(result, output)
    assert not findings and not committed_findings, 'Secret pattern found; locations preserved without values'
    assert not byte_mismatches, 'Working/committed bytes differ; commit before final integrity audit'
    print(json.dumps({k: result[k] for k in ('status', 'historical_files_verified', 'new_ledger_rows', 'changed_git_blobs_scanned')}))


if __name__ == '__main__':
    main()
