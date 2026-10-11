"""Run an unchanged Phase17 qualification driver with observational telemetry."""
from __future__ import annotations
import argparse
import gc
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import threading
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'run_sol_tabpfn_confirm.py', 'run_phase17_confirmation.py'}


def c_free_gib():
    return shutil.disk_usage(ROOT).free/2**30


def preflight():
    if c_free_gib() < 40:
        raise RuntimeError('RESCUE_PREFLIGHT: at least40GiB free at rest is required; original20GiB guard stays unchanged')
    if psutil.virtual_memory().available < 4*2**30:
        raise RuntimeError('RESCUE_PREFLIGHT: original4GiB host prefit guard is not met')
    own = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    for process in psutil.process_iter(['name', 'cmdline']):
        try:
            if process.pid in own or not (process.info['name'] or '').lower().startswith('python'):
                continue
            command = process.info['cmdline'] or []
            if any(Path(a).name in ALLOWED | {'run_phase17_tabpfn.py', 'resume_phase17_context.py',
                                             'replay_sol_classical.py'} for a in command):
                if Path(process.cwd()).resolve() == ROOT:
                    raise RuntimeError(f'RESCUE_PREFLIGHT: separate campaign worker{process.pid} is active')
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass


def temp_snapshot(path):
    """Only immediate metadata, capped; never recurse through other projects."""
    try:
        count = total = 0
        newest = []
        with os.scandir(path) as entries:
            for entry in entries:
                count += 1
                if count > 1500:
                    break
                stat = entry.stat(follow_symlinks=False)
                if entry.is_file(follow_symlinks=False):
                    total += stat.st_size
                newest.append((stat.st_mtime, entry.name, stat.st_size if entry.is_file(follow_symlinks=False) else None))
        return {'path': str(path), 'immediate_file_bytes': total, 'entries_observed': count,
                'capped': count > 1500, 'newest_entries': sorted(newest, reverse=True)[:8],
                'scope': 'Immediate metadata only, not total recursive directory size'}
    except OSError as error:
        return {'path': str(path), 'unavailable': type(error).__name__}


def observation(process, pagefiles=None):
    info = process.memory_info()._asdict()
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'pid': process.pid,
           'c_free_bytes': shutil.disk_usage(ROOT).free,
           'host_available_bytes': psutil.virtual_memory().available,
           'process_memory_native_fields_bytes': info,
           'process_private_bytes': info.get('private'),
           'process_commit_charge_bytes': info.get('pagefile'),
           'process_rss_bytes': info['rss'], 'process_virtual_bytes': info['vms']}
    if pagefiles is not None:
        try:
            row['pagefiles'] = [{'name': str(p.Name), 'allocated_bytes': int(p.AllocatedBaseSize)*2**20,
                                'used_bytes': int(p.CurrentUsage)*2**20, 'peak_used_bytes': int(p.PeakUsage)*2**20}
                               for p in pagefiles.ExecQuery('SELECT Name,AllocatedBaseSize,CurrentUsage,PeakUsage FROM Win32_PageFileUsage')]
        except Exception as error:
            row['pagefile_query_unavailable'] = type(error).__name__
    torch = sys.modules.get('torch')
    # Torch can be visible in sys.modules before its import has finished.
    # Observe only an already initialized CUDA runtime; never initialize it here.
    cuda = getattr(torch, 'cuda', None)
    initialized = getattr(cuda, 'is_initialized', None)
    row['gpu_allocated_bytes'] = row['gpu_reserved_bytes'] = None
    if callable(initialized):
        try:
            if initialized():
                row['gpu_allocated_bytes'] = int(cuda.memory_allocated(0))
                row['gpu_reserved_bytes'] = int(cuda.memory_reserved(0))
        except Exception as error:
            row['gpu_query_unavailable'] = type(error).__name__
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--telemetry-tag', required=True)
    ap.add_argument('--interval', type=int, default=15, choices=range(15, 31))
    ap.add_argument('driver', choices=sorted(ALLOWED))
    args, driver_args = ap.parse_known_args()
    assert Path.cwd().resolve() == ROOT
    # The rescue authorizes only the frozen two-estimator qualification path.
    if args.driver == 'run_phase17_confirmation.py':
        position = driver_args.index('--primary-tag')
        primary = json.loads((ROOT/'reports'/f'{driver_args[position+1]}_primary.json').read_text())
        assert primary['internal_estimators'] == 2
    assert '--seed-diagnostic' not in driver_args and '--head-views' not in driver_args
    preflight()
    folder = ROOT/'reports'/args.telemetry_tag
    assert not folder.exists(), 'Use a new telemetry tag; preserve prior observations'
    folder.mkdir()
    driver = ROOT/'scripts'/args.driver
    contract = {'utc': datetime.now(timezone.utc).isoformat(), 'worker_pid': os.getpid(),
                'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'driver_path': str(driver), 'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
                'driver_args': driver_args, 'interval_seconds': args.interval,
                'resource_guards_modified': False, 'ML_recipe_modified': False,
                'source': 'Frozen driver executed with runpy; only read-only resource samples added',
                'inflight_extra_disk_stop_bytes': 20*2**30}
    (folder/'contract.json').write_text(json.dumps(contract, indent=2)+'\n', encoding='utf-8')
    stop = threading.Event()
    started = time.monotonic()
    process = psutil.Process()

    def monitor():
        pagefiles = None
        com = None
        try:
            import pythoncom
            import win32com.client
            com = pythoncom
            com.CoInitialize()
            pagefiles = win32com.client.GetObject('winmgmts:{impersonationLevel=impersonate}!\\\\.\\root\\cimv2')
        except Exception as error:
            (folder/'pagefile_query_unavailable.json').write_text(json.dumps({'error_type': type(error).__name__})+'\n')
        try:
            with (folder/'observations.jsonl').open('x', encoding='utf-8') as stream:
                while not stop.is_set():
                    row = observation(process, pagefiles)
                    row['elapsed_seconds'] = time.monotonic()-started
                    row['temporary_directories'] = [temp_snapshot(Path(os.environ.get('TEMP', 'C:/Windows/Temp')))]
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    if row['c_free_bytes'] < 20*2**30:
                        abort = {**contract, **row, 'status': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE',
                                 'reason': 'Original20GiB disk reserve breached; only this worker stops, no model verdict'}
                        (folder/'resource_abort.json').write_text(json.dumps(abort, indent=2)+'\n', encoding='utf-8')
                        os._exit(86)
                    stop.wait(args.interval)
        except Exception as error:
            # Missing observations must not leave an apparently monitored job.
            failure = {**contract, 'status': 'INVALID_OBSERVER_NO_MODEL_VERDICT',
                       'utc': datetime.now(timezone.utc).isoformat(),
                       'error_type': type(error).__name__, 'error': str(error)}
            (folder/'observer_abort.json').write_text(json.dumps(failure, indent=2)+'\n', encoding='utf-8')
            os._exit(86)
        finally:
            if com is not None:
                pagefiles = None
                gc.collect()
                com.CoUninitialize()

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    summary = {**contract, 'status': 'RUNNING'}
    try:
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT/'scripts'))
        sys.argv = [str(driver), *driver_args]
        runpy.run_path(str(driver), run_name='__main__')
        summary['status'] = 'FROZEN_DRIVER_COMPLETE'
    except SystemExit as error:
        summary['status'] = 'FROZEN_DRIVER_COMPLETE' if error.code in (None, 0) else 'FROZEN_DRIVER_FAILED'
        summary['exit_code'] = error.code
        raise
    except Exception as error:
        summary['status'] = 'INVALID_RESOURCE_NO_MODEL_VERDICT' if type(error).__name__ in ('ResourcePreflightError', 'OutOfMemoryError') else 'FROZEN_DRIVER_EXCEPTION_NO_MODEL_VERDICT'
        summary['error_type'] = type(error).__name__
        summary['error'] = str(error)
        raise
    finally:
        stop.set(); watcher.join(timeout=3)
        summary['seconds'] = time.monotonic()-started
        summary['final_resources'] = observation(process)
        (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
