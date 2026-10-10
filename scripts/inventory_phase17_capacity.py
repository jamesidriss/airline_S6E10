"""Read-only targeted storage inventory; no cleanup or model changes."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
import psutil

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
SIZE = KERNEL.GetCompressedFileSizeW
SIZE.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
SIZE.restype = ctypes.c_ulong
ATTR = KERNEL.GetFileAttributesW
ATTR.argtypes = [ctypes.c_wchar_p]
ATTR.restype = ctypes.c_ulong


def allocated(path):
    ctypes.set_last_error(0)
    high = ctypes.c_ulong()
    low = SIZE(str(path), ctypes.byref(high))
    if low == 0xffffffff and ctypes.get_last_error():
        raise ctypes.WinError()
    return low + (high.value << 32)


def inventory(directory, purpose, protected=False):
    start = time.monotonic()
    result = {'path': str(directory), 'owner_purpose': purpose, 'protected': protected,
              'exists': directory.exists(), 'logical_bytes': 0, 'allocated_bytes': 0,
              'files': 0, 'errors': [], 'top_files': [], 'groups': {}, 'truncated': False}
    if not result['exists']:
        return result
    stack = [directory]
    largest = []
    while stack:
        folder = stack.pop()
        if time.monotonic()-start > 90 or result['files'] > 250000:
            result['truncated'] = True
            break
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    # Avoid reparse points, credential stores, other project trees.
                    if entry.name.lower() in {'.kaggle', '.git', 'credentials', 'secrets'}:
                        continue
                    if entry.is_symlink() or ATTR(str(path)) & 0x400:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(path)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    stat = entry.stat(follow_symlinks=False)
                    physical = allocated(path)
                    relative = path.relative_to(directory)
                    group = relative.parts[0] if len(relative.parts) > 1 else '<root files>'
                    totals = result['groups'].setdefault(group, {'logical_bytes': 0, 'allocated_bytes': 0, 'files': 0})
                    for value in (result, totals):
                        value['logical_bytes'] += stat.st_size
                        value['allocated_bytes'] += physical
                        value['files'] += 1
                    if stat.st_size >= 16*1024**2:
                        largest.append({'path': str(path), 'logical_bytes': stat.st_size,
                                        'allocated_bytes': physical, 'modified_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                        'ntfs_compressed_flag': bool(ATTR(str(path)) & 0x800)})
        except (OSError, PermissionError) as error:
            result['errors'].append({'path': str(folder), 'error': type(error).__name__})
    result['top_files'] = sorted(largest, key=lambda row: row['allocated_bytes'], reverse=True)[:80]
    result['seconds'] = time.monotonic()-start
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='reports/phase17_capacity_inventory.json')
    args = ap.parse_args()
    output = ROOT/args.output
    if output.exists():
        raise FileExistsError('Preserve the prior inventory; choose a new output')
    excluded = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    workers = []
    for process in psutil.process_iter(['name', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            if process.pid not in excluded and (process.info['name'] or '').lower().startswith('python'):
                workers.append({'pid': process.pid, 'executable': command[0] if command else None,
                                'script': next((s for s in command[1:] if s.endswith('.py')), None)})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    user_local = Path(os.environ['USERPROFILE'])/'AppData/Local'
    specs = [(ROOT/'artifacts', 'Campaign artifacts: predictions/checkpoints protected; generated feature caches distinguishable'),
             (ROOT/'data/cache', 'Registered folds protected; generated input parquet caches', True),
             (ROOT/'checkpoints', 'Saved model weights; absolute protection', True),
             (user_local/'Temp', 'Host shared Windows/Python temporary files; owner unproven, no deletion authorized'),
             (Path(tempfile.gettempdir()), 'Tool-process temporary directory, potentially sandbox-redirected; no deletion selected'),
             (user_local/'pip/Cache', 'Host shared pip download/wheel cache; regenerable, owner not campaign-specific'),
             (Path(os.environ['USERPROFILE'])/'.cache/huggingface', 'Shared pretrained-model downloads; checkpoint protection', True),
             (Path(os.environ['USERPROFILE'])/'.cache/torch', 'Shared torch downloads/build caches; checkpoint protection', True),
             (user_local/'torch_extensions', 'Shared torch compiled extensions; no deletion selected')]
    scans = []
    before = shutil.disk_usage(ROOT)
    for spec in specs:
        row = inventory(*spec)
        scans.append(row)
        print(json.dumps({'path': row['path'], 'allocated_gib': row['allocated_bytes']/2**30,
                          'logical_gib': row['logical_bytes']/2**30, 'files': row['files'], 'truncated': row['truncated']}), flush=True)
    volumes = []
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = shutil.disk_usage(partition.mountpoint)
            volumes.append({'mountpoint': partition.mountpoint, 'filesystem': partition.fstype,
                            'total_bytes': usage.total, 'free_bytes': usage.free})
        except OSError:
            pass
    candidates = []
    for scan in scans:
        for row in scan['top_files']:
            p = Path(row['path'])
            generated = any(p.is_relative_to(ROOT/f'artifacts/{s}') for s in ('features', 'aux_distribution', 'sol_ssl12')) and p.suffix in ('.npz', '.npy')
            candidates.append({**row, 'owner_purpose': scan['owner_purpose'], 'actively_in_use': 'not established per file; no campaign model worker' if not workers else 'unresolved; active Python workers present',
                               'reproducibility': 'Generated cache; source/manifests must be verified before operation' if generated else 'Not established or protected model/prediction payload',
                               'proposed_operation': 'Consider reversible compression only' if generated else 'No operation selected',
                               'risk': 'Require logical SHA equality, narrow scope and no active model use' if generated else 'Protected or uncertain ownership; do not delete/modify',
                               'expected_recoverable_bytes': None})
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'READ_ONLY_TARGETED_INVENTORY_COMPLETE',
              'initial_c_free_bytes': before.free, 'final_c_free_bytes': shutil.disk_usage(ROOT).free,
              'volumes': volumes, 'python_workers_excluding_self_and_ancestors': workers,
              'host_available_bytes': psutil.virtual_memory().available, 'scans': scans,
              'recovery_candidates': candidates, 'operations_performed': [],
              'scope': 'Targeted project/cache/temp metadata only; no personal-file trees, checkpoint contents, secret stores, OS-managed file operations or whole-disk recursion',
              'limitations': 'Allocated-size sum may double-count hard links. Metadata inventory alone does not establish ownership, open handles or recoverable free space. Shared temp and system storage cleanup remain unselected.'}
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
