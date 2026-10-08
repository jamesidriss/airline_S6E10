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
    import tests.test_view_composition as TV
    import tests.test_sol_contracts as TS
    import tests.test_private_sim as TP
    import tests.test_masked_encoder as TM
    import tests.test_sol_research as TQ
    import tests.test_multitask as TC
    import tests.test_kaggle_io as TK
    import tests.test_sol_phase15 as T15

    # NOTE: test_stochastic_protocol.py has its own __main__ runner with per-assertion reporting, so
    # it is invoked as a subprocess rather than imported here -- importing it would only pick up its
    # module-level helpers. It is registered explicitly so the aggregate count stays honest; a test
    # file that silently stops being run is worse than no test file.
    mods = (T, TR, TW, TN, TI, TE, TV, TS, TP, TM, TQ, TC, TK, T15)
    fns = [(mod.__name__, getattr(mod, n)) for mod in mods
           for n in sorted(dir(mod)) if n.startswith("test_")]
    ok = fail = 0

    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "test_stochastic_protocol.py")],
                       capture_output=True, text=True, cwd=str(ROOT))
    tail = [ln for ln in r.stdout.splitlines() if "passed" in ln]
    print(f"  {'PASS' if r.returncode == 0 else 'FAIL'}  "
          f"tests.test_stochastic_protocol  ({tail[-1].strip() if tail else 'no summary'})")
    if r.returncode == 0:
        ok += 1
    else:
        fail += 1
        print(r.stdout[-3000:])
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
