"""Phase 24 -- verify the champion composition identity from stored artifacts (read-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, file_sha256  # noqa: E402


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def main() -> None:
    route = np.load("artifacts/phase17_B1/primary/route_oof.npy").astype("float64")
    aux = np.load("artifacts/sol_route_aux10_primary_final/aux10_oof.npy").astype("float64")
    port = np.load("artifacts/phase17_B1/primary/portfolio_oof.npy").astype("float64")
    v7 = np.load("artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy").astype("float64")
    comp = 1.0 / (1.0 + np.exp(-(logit(route) + logit(aux)) / 2.0))
    out = {
        "route_sha256": arr_sha256(route.astype("float32")),
        "aux10_sha256": arr_sha256(aux.astype("float32")),
        "portfolio_sha256": arr_sha256(port.astype("float32")),
        "v7_sha256": file_sha256("artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy"),
        "portfolio_vs_v7_max_abs": float(np.max(np.abs(port - v7))),
        "portfolio_vs_v7_bit_equal": bool(np.array_equal(port.astype("float32"), v7.astype("float32"))),
        "recomposed_vs_v7_max_abs": float(np.max(np.abs(comp - v7))),
        "recomposed_vs_portfolio_max_abs": float(np.max(np.abs(comp - port))),
        "route_range": [float(route.min()), float(route.max())],
        "aux_range": [float(aux.min()), float(aux.max())],
    }
    # test side identity
    tr_ = np.load("artifacts/v7_sol_tabpfn_internal2_aux10_test/route.npy").astype("float64")
    ta = np.load("artifacts/v7_sol_tabpfn_internal2_aux10_test/aux10.npy").astype("float64")
    tc = np.load("artifacts/v7_sol_tabpfn_internal2_aux10_test/candidate.npy").astype("float64")
    tcomp = 1.0 / (1.0 + np.exp(-(logit(tr_) + logit(ta)) / 2.0))
    out["test_recomposed_vs_candidate_max_abs"] = float(np.max(np.abs(tcomp - tc)))
    out["test_candidate_sha256"] = arr_sha256(tc.astype("float32"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
