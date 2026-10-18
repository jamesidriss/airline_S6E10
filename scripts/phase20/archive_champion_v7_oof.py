"""Archive champion v7 OOF + test vectors into the prediction store (zero GPU).

The per-fold champion vectors already exist in artifacts/phase17_B1/f{0..4}/
(portfolio.npy = champion v7 blend, prediction.npy = route) and the test vector
in artifacts/v7_sol_tabpfn_internal2_aux10_test/; all are hash-certified in
reports/phase17_B1_primary.json and the v7 test certificate.  They were simply
never registered in the prediction store (phase17 did not use it).  This script
assembles, verifies against every recorded hash, and registers them so the
champion-stability audit, finalist lock, and deferred matched-inference work can
load the champion like any other experiment.

No new predictive computation: every vector is read from certified artifacts.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import arr_sha256, file_sha256, load_cached_parquet  # noqa: E402
from src.submission.store import load_oof, load_test, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = "v7_sol_tabpfn_internal2_aux10"
TEST_SHA = "2b744d0df36a0b1143a92e1db18661ccd94fa881490784308624aaa0a0fc95af"


def main() -> None:
    rep = json.loads((ROOT / "reports/phase17_B1_primary.json").read_text(encoding="utf-8"))
    cert = json.loads(
        (ROOT / "reports/v7_sol_tabpfn_internal2_aux10_test_certificate.json").read_text(encoding="utf-8")
    )
    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")

    # ---- assemble champion v7 OOF + route across the five primary folds ----
    oof = np.zeros(len(tr), dtype=np.float32)
    route = np.zeros(len(tr), dtype=np.float32)
    covered = np.zeros(len(tr), dtype=bool)
    per_fold = {}
    for k in range(5):
        d = ROOT / "artifacts/phase17_B1" / f"f{k}"
        ids = np.load(d / "apply_ids.npy")
        port = np.asarray(np.load(d / "portfolio.npy"), dtype=np.float32)
        rte = np.asarray(np.load(d / "prediction.npy"), dtype=np.float32)
        assert not covered[ids].any(), f"fold {k} apply_ids overlap earlier folds"
        assert len(ids) == len(port) == len(rte)
        oof[ids] = port
        route[ids] = rte
        covered[ids] = True
        fr = rep["folds"][k]
        assert fr["fold"] == k
        for key, arr in (("portfolio", port), ("route", rte)):
            rec = fr.get(f"{key}_sha256")
            if rec is not None:
                assert arr_sha256(arr) == rec, f"fold {k} {key} sha mismatch"
        per_fold[str(k)] = {
            "rows": int(len(ids)),
            "portfolio_auc": fr["portfolio_auc"],
            "route_auc": fr["route_auc"],
        }
    assert covered.all(), "OOF coverage incomplete"

    # ---- top-level hash + AUC verification ----
    assert arr_sha256(oof) == rep["portfolio_sha256"], "assembled OOF hash != report"
    assert arr_sha256(route) == rep["route_sha256"], "assembled route hash != report"
    auc = float(roc_auc_score(y, oof))
    published = float(rep["full_pooled_oof_auc"])
    assert abs(auc - published) < 1e-12, "OOF AUC does not reproduce the published value"
    route_auc = float(roc_auc_score(y, route))
    print(f"OOF auc={auc!r} (published {published!r}) route_auc={route_auc!r}")

    # ---- route OOF note: the phase17 route vector is a separate pipeline from
    # the store's sol_route_aux10_primary_final (phase16 route final); both are
    # valid and are kept distinct.  No equality is asserted here.
    stored_route = load_oof("sol_route_aux10_primary_final")
    stored_route_auc = float(roc_auc_score(y, stored_route))
    print(f"stored phase16 route auc={stored_route_auc!r} (kept distinct)")

    # ---- champion test vector ----
    troot = Path(cert["test_artifact_root"])
    test = np.asarray(np.load(troot / "candidate.npy"), dtype=np.float32)
    test_route = np.asarray(np.load(troot / "route.npy"), dtype=np.float32)
    test_aux = np.asarray(np.load(troot / "aux10.npy"), dtype=np.float32)
    test_ids = np.load(troot / "test_ids.npy")
    assert len(test) == len(te) == len(test_ids)
    assert arr_sha256(test) == TEST_SHA == cert["test_sha256"]["candidate"], "test hash mismatch"
    assert arr_sha256(test_route) == cert["test_sha256"]["route"]
    assert arr_sha256(test_aux) == cert["test_sha256"]["aux10"]
    assert arr_sha256(test_ids) == cert["test_ids_sha256"]

    # ---- register in the prediction store ----
    entry = save(
        EXP,
        oof,
        test,
        fold_scheme="primary",
        meta={
            "kind": "champion archival (certified phase17 artifacts; no new run)",
            "source_report": "reports/phase17_B1_primary.json",
            "source_report_sha256": file_sha256(ROOT / "reports/phase17_B1_primary.json"),
            "source_test_certificate": "reports/v7_sol_tabpfn_internal2_aux10_test_certificate.json",
            "source_test_certificate_sha256": file_sha256(
                ROOT / "reports/v7_sol_tabpfn_internal2_aux10_test_certificate.json"
            ),
            "oof_artifacts": "artifacts/phase17_B1/f{0..4}/{apply_ids,portfolio,prediction}.npy",
            "test_artifacts": str(troot),
            "per_fold": per_fold,
            "oof_auc": auc,
            "route_oof_auc": route_auc,
            "test_sha256": TEST_SHA,
            "archived_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("store entry:", json.dumps({k: v for k, v in entry.items() if k != "meta"}, indent=1))

    # ---- read-back verification ----
    o2 = load_oof(EXP)
    t2 = load_test(EXP)
    assert np.array_equal(o2, oof) and np.array_equal(t2, test)
    print("read-back OK; champion v7 OOF + test now in the prediction store")


if __name__ == "__main__":
    main()
