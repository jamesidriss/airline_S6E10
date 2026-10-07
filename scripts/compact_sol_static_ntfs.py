"""Lossless NTFS compression of exact named static caches, with byte hashes."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FEATURES=(ROOT/'artifacts'/'features').resolve()
NAMES=('static_full_ogm.npz','static_full_ogsurf.npz','static_raw_trans.npz')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        while block:=handle.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def allocated(path):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    function=kernel.GetCompressedFileSizeW
    function.argtypes=(ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong))
    function.restype=ctypes.c_ulong
    high=ctypes.c_ulong()
    ctypes.set_last_error(0)
    low=function(str(path),ctypes.byref(high))
    if low==0xFFFFFFFF and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    return (high.value<<32)|low


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--algorithm',choices=('ntfs','lzx'),default='ntfs')
    args=parser.parse_args()
    output=ROOT/'reports'/('sol_cache_ntfs_compression.json' if args.algorithm=='ntfs' else 'sol_cache_lzx_compression.json')
    if output.exists():
        raise FileExistsError('Preserve the existing compression report')
    rows=[]; before=shutil.disk_usage(ROOT).free
    names=NAMES if args.algorithm=='ntfs' else NAMES+('static_full.npz','static_core3.npz')
    for name in names:
        path=(FEATURES/name).resolve()
        assert path.parent==FEATURES and path.is_file()
        row={'path':str(path),'logical_bytes':path.stat().st_size,
            'sha256_before':digest(path),'allocated_before':allocated(path),
            'reversal_command':f'compact.exe /u {"/exe " if args.algorithm=="lzx" else ""}"{path}"'}
        if args.execute:
            options=['/c','/i','/a']+(['/f','/exe:lzx'] if args.algorithm=='lzx' else [])
            operation=subprocess.run(['compact.exe',*options,str(path)],capture_output=True)
            assert operation.returncode==0, 'Native compression failed; retain original files'
            row.update(sha256_after=digest(path),allocated_after=allocated(path))
            assert row['sha256_before']==row['sha256_after'], 'Cache content changed during filesystem compression'
            assert path.stat().st_size==row['logical_bytes']
        rows.append(row)
        print(f'{name}: byte content verified',flush=True)
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'EXECUTED' if args.execute else 'PLAN_ONLY',
        'algorithm':args.algorithm,
        'files':rows,'disk_free_before':before,'disk_free_after':shutil.disk_usage(ROOT).free,
        'allocated_bytes_saved':sum(r['allocated_before']-r.get('allocated_after',r['allocated_before']) for r in rows),
        'preserved':'Every file byte, predictions, model weights, folds, raw/original data and submissions; no files deleted'}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','allocated_bytes_saved','disk_free_before','disk_free_after')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
