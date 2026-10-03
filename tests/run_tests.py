"""Dependency-free test runner: `python tests/run_tests.py`"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import tests.test_pipeline as T

    fns = [getattr(T, n) for n in sorted(dir(T)) if n.startswith("test_")]
    ok = fail = 0
    for fn in fns:
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {fn.__name__}  ({time.time()-t0:.1f}s)")
            ok += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            fail += 1
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())