"""Release regenerable feature caches absent from saved finalist recipes."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from compact_static_cache_reserve import payload_hashes

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (ROOT / 'artifacts' / 'features').resolve()
VIEWS = ('full_ogm_ogs', 'core3_ogsurf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    rows = []
    for view in VIEWS:
        for report in (ROOT / 'reports').glob('finalist*.json'):
            members = json.loads(report.read_text(encoding='utf-8')).get('members', [])
            assert not any(m.get('featureset') == view for m in members), 'Saved finalist uses this cache'
        path = (FEATURES / f'static_{view}.npz').resolve()
        assert path.parent == FEATURES, 'Cache escapes the named feature directory'
        if path.exists():
            rows.append({'path': str(path), 'bytes': path.stat().st_size,
                'payload_sha256': payload_hashes(path), 'view': view,
                'reason': 'regenerable label-free static view, absent from saved finalist recipes',
                'regeneration': f"python -c \"from src.common import load_cached_parquet; from src.features.view import ViewBuilder; a,b=load_cached_parquet(); ViewBuilder(a,b,'{view}').build_static()\""})
    output = ROOT / 'reports' / 'sol_cache_reserve3.json'
    if output.exists():
        raise FileExistsError('Existing cleanup record is preserved')
    record = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'EXECUTED' if args.execute else 'PLAN_ONLY', 'files': rows,
        'total_bytes': sum(r['bytes'] for r in rows),
        'preserved': 'raw/original data, folds, model weights, predictions, submissions and used static views'}
    output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    if args.execute:
        for row in rows:
            path = Path(row['path']).resolve()
            assert path.parent == FEATURES and path.stat().st_size == row['bytes']
            path.unlink()
    print(json.dumps({'files': len(rows), 'bytes': record['total_bytes'], 'status': record['status']}))


if __name__ == '__main__':
    main()
