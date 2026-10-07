"""Recover reserve from exact aliases and an unused, regenerable static view.

No raw data, original data, folds, predictions, model weights or submissions are
touched. Exact aliases are checked through streamed NPY payload hashes and names.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (ROOT / 'artifacts' / 'features').resolve()


def payload_hashes(path):
    result = {}
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == {'tr.npy', 'te.npy'}
        for name in sorted(archive.namelist()):
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            result[name] = digest.hexdigest()
    return result


def checked(name):
    path = (FEATURES / name).resolve()
    if path.parent != FEATURES or not path.is_file():
        raise ValueError(f'Cache is missing or escapes the named feature directory: {name}')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    rows = []
    for alias, canonical in [('static_full_te21.npz', 'static_full.npz'),
                             ('static_full_ogsurf_te.npz', 'static_full_ogsurf.npz')]:
        if not (FEATURES / alias).exists():
            continue
        a, c = checked(alias), checked(canonical)
        ah, ch = payload_hashes(a), payload_hashes(c)
        assert ah == ch, 'Static payloads are not exact duplicates'
        assert json.loads(a.with_suffix('.json').read_text()) == json.loads(c.with_suffix('.json').read_text())
        rows.append({'path': str(a), 'canonical': str(c), 'bytes': a.stat().st_size,
                     'payload_sha256': ah, 'reason': 'exact static alias; canonical cache retained'})
    closed = FEATURES / 'static_full_enrich.npz'
    if closed.exists():
        for report in (ROOT / 'reports').glob('finalist*.json'):
            members = json.loads(report.read_text(encoding='utf-8')).get('members', [])
            assert not any(m.get('featureset') == 'full_enrich' for m in members), 'View belongs to a saved finalist'
        p = checked(closed.name)
        rows.append({'path': str(p), 'bytes': p.stat().st_size, 'payload_sha256': payload_hashes(p),
                     'reason': 'regenerable label-free full_enrich cache; absent from every saved finalist member manifest',
                     'regeneration': "python -c \"from src.common import load_cached_parquet; from src.features.view import ViewBuilder; a,b=load_cached_parquet(); ViewBuilder(a,b,'full_enrich').build_static()\""})
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'EXECUTED' if args.execute else 'PLAN_ONLY',
              'files': rows, 'total_bytes': sum(r['bytes'] for r in rows),
              'preserved': 'all canonical/used feature caches, raw/original data, folds, weights, predictions and submissions'}
    output = ROOT / 'reports' / 'sol_cache_reserve2.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if args.execute:
        # All hashes and boundaries are validated before the first deletion.
        for row in rows:
            p = Path(row['path'])
            assert p.resolve().parent == FEATURES and p.stat().st_size == row['bytes']
            p.unlink()
    print(json.dumps({'files': len(rows), 'bytes': result['total_bytes'], 'status': result['status']}))


if __name__ == '__main__':
    main()
