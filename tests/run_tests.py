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
    import tests.test_tabr_retrieval as TR
    import tests.test_no_undefined_names as TN
    import tests.test_tabr_wiring as TW
    import tests.test_index_space as TI
    import tests.test_te_all21 as TE

    fns = [(mod.__name__, getattr(mod, n)) for mod in (T, TR, TW, TN, TI, TE)
           for n in sorted(dir(mod)) if n.startswith("test_")]
    ok = fail = 0
    for modname, fn in fns:
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {modname}.{fn.__name__}  ({time.time()-t0:.1f}s)")
            ok += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {modname}.{fn.__name__}")
            traceback.print_exc()
            fail += 1
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())