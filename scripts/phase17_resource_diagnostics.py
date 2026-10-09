"""Record guard-entry evidence without modifying the immutable resource guard."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import psutil
from src.common import save_json
from src.models.resource_guard import inference_guard


def snapshot(directory):
    path = Path(directory).resolve()
    usage = shutil.disk_usage(path)
    return {'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
            'cwd': str(Path.cwd()), 'absolute_guard_path': str(path),
            'disk_free_bytes': usage.free, 'disk_total_bytes': usage.total,
            'host_available_bytes': psutil.virtual_memory().available,
            'process_rss_bytes': psutil.Process().memory_info().rss}


@contextmanager
def diagnosed_guard(directory, contract, **limits):
    directory = Path(directory)
    entry = snapshot(directory)
    save_json({'entry': entry, 'limits': limits}, directory/'guard_entry.json')
    try:
        with inference_guard(directory, contract, **limits):
            yield
    except Exception as error:
        save_json({'entry': entry, 'limits': limits, 'exception': repr(error),
                   'after_exception': snapshot(directory)}, directory/'guard_exception.json')
        raise
