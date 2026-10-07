"""Periodic telemetry and resource limits for our own expensive inference jobs."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import threading
import time
import psutil


@contextmanager
def inference_guard(directory, contract, max_seconds=2700):
    directory = Path(directory)
    if shutil.disk_usage(directory).free < 20 * 1024**3:
        raise RuntimeError('Full inference requires at least 20 GiB free disk reserve')
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise RuntimeError('Full inference requires at least 8 GiB available host RAM')
    stop = threading.Event()
    start = time.monotonic()

    def monitor():
        low_memory = 0
        while not stop.is_set():
            memory = psutil.virtual_memory()
            disk = shutil.disk_usage(directory).free
            elapsed = time.monotonic() - start
            row = {'elapsed_seconds': elapsed, 'host_available_bytes': memory.available,
                   'disk_free_bytes': disk, 'process_rss_bytes': psutil.Process().memory_info().rss}
            target = directory / 'resources.json'
            temporary = directory / 'resources.partial.json'
            temporary.write_text(json.dumps(row, indent=2), encoding='utf-8')
            temporary.replace(target)
            low_memory = low_memory + 1 if memory.available < 1024**3 else 0
            if low_memory >= 2 or disk < 8 * 1024**3 or elapsed > max_seconds:
                error = {**contract, **row, 'status': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE',
                         'reason': 'Host/disk reserve or predeclared 45-minute inference limit breached; own worker terminated.'}
                (directory / 'resource_abort.json').write_text(json.dumps(error, indent=2, default=str), encoding='utf-8')
                print('Resource guard terminated this inference worker; no completed CV result.', flush=True)
                os._exit(86)
            stop.wait(15)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)
