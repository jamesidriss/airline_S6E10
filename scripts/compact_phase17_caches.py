"""Reversible NTFS compression of generated caches, with logical-byte proofs."""
import ctypes
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, file_sha256, save_json


def allocated_bytes(path):
    fn=ctypes.WinDLL('kernel32',use_last_error=True).GetCompressedFileSizeW
    fn.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)]
    fn.restype=ctypes.c_ulong
    high=ctypes.c_ulong();low=fn(str(path),ctypes.byref(high))
    if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError()
    return low+(high.value<<32)


def main():
    output=ROOT/'reports/phase17_disk_capacity_compaction_20261009.json'
    assert not output.exists(), 'Preserve the intervention evidence'
    directories=[ROOT/'artifacts/features',ROOT/'artifacts/aux_distribution',ROOT/'artifacts/sol_ssl12']
    inventory=subprocess.run(['rg','--files','--hidden','--no-ignore',*[str(p) for p in directories]],
        check=True,capture_output=True,text=True)
    paths=[]
    for name in inventory.stdout.splitlines():
        p=Path(name).resolve()
        assert any(p.is_relative_to(d.resolve()) for d in directories)
        if p.suffix in ('.npy','.npz') and p.stat().st_size>=16*1024**2:paths.append(p)
    paths.sort(key=lambda p:p.stat().st_size,reverse=True)
    result={'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':file_sha256(__file__),
        'scope':'Only generated cache files under the three explicit artifact directories; no deletion, system configuration, checkpoint, source, raw data or finalist prediction mutation',
        'operation':'compact.exe /C /F /EXE:LZX; reversible with compact.exe /U /EXE',
        'status':'IN_PROGRESS','disk_before_bytes':shutil.disk_usage(ROOT).free,'files':[]}
    save_json(result,output);started=time.monotonic()
    for p in paths:
        before=file_sha256(p);physical=allocated_bytes(p)
        r=subprocess.run(['compact.exe','/C','/F','/EXE:LZX',str(p)],capture_output=True,text=True)
        after=file_sha256(p)
        row={'path':str(p.relative_to(ROOT)),'logical_bytes':p.stat().st_size,'before_sha256':before,
            'after_sha256':after,'allocated_before_bytes':physical,'allocated_after_bytes':allocated_bytes(p),
            'returncode':r.returncode,'tool_output':r.stdout[-1500:],'tool_error':r.stderr[-1500:]}
        result['files'].append(row);result['disk_after_bytes']=shutil.disk_usage(ROOT).free
        result['seconds']=time.monotonic()-started;save_json(result,output)
        assert before==after, 'Logical bytes changed: stop before any model job'
        assert r.returncode==0, 'Compression failed: preserve evidence and stop'
        print(json.dumps({k:row[k] for k in ('path','allocated_before_bytes','allocated_after_bytes')}),flush=True)
    result['status']='PASS_ALL_LOGICAL_BYTES_IDENTICAL';save_json(result,output)
    print(json.dumps({'status':result['status'],'disk_after_gib':result['disk_after_bytes']/1024**3,'seconds':result['seconds']}))


if __name__=='__main__':main()
