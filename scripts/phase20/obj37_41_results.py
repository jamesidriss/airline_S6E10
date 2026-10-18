"""Aggregate OBJ37-41 arm results (control + 5 objectives) into one verified artifact.

For every arm this recomputes the standalone AUC from the saved prediction array,
verifies the array SHA against the report, and assembles the mandate's required
comparison set: delta vs the BCE control, delta vs the TRUE v7 champion
(v7_sol_tabpfn_internal2_aux10), the frozen gate verdict (route_ref port_delta
>= +5e-5 AND spearman < 0.995), correlations, resource usage, and whether a
champion-admission evaluation (5-fold promotion) is justified.

CPU-only, reads saved arrays.  Usage:
  ./.venv/Scripts/python.exe scripts/phase20/obj37_41_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, load_cached_parquet  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

ARMS = ["bce_control", "supcon", "auc_surrogate", "hard_mining", "gce", "ema"]
ART = ROOT / "artifacts" / "phase20" / "obj37_41_objectives"
REP = ROOT / "reports" / "phase20" / "obj37_41_objectives"
FOLDS = np.load(ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy")
CHAMPION = "v7_sol_tabpfn_internal2_aux10"
CONTROL_REFERENCE = 0.9576563038026185
GATE_MIN_PORT_DELTA = 5e-5
GATE_MAX_SPEARMAN = 0.995
OUT = ROOT / "reports" / "endgame" / "obj37_41_results.json"


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> int:
    tr, _te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    m = FOLDS == 0
    y0 = y[m]
    champ0 = load_oof(CHAMPION)[m]
    route0 = load_oof("sol_route_aux10_primary_final")[m]
    champ_auc0 = float(roc_auc_score(y0, champ0))

    arms: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
    for arm in ARMS:
        rpath = REP / f"{arm}_fold0.json"
        ppath = ART / f"{arm}_fold0.npy"
        if not rpath.is_file() or not ppath.is_file():
            arms[arm] = {"status": "PENDING" if not rpath.is_file() else "ARRAY_MISSING"}
            continue
        rep = json.loads(rpath.read_text(encoding="utf-8"))
        pred = np.asarray(np.load(ppath), dtype=np.float32)
        sha = arr_sha256(pred)
        sha_ok = sha == rep.get("prediction_sha256")
        auc = float(roc_auc_score(y0, pred))
        blend_champ = (0.5 * (logit(champ0) + logit(pred)))
        fr = rep.get("frozen_rule", {})
        arms[arm] = {
            "status": rep.get("status"),
            "git": rep.get("git"),
            "prereg_sha256": rep.get("prereg_sha256"),
            "standalone_auc_recomputed": auc,
            "standalone_auc_reported": rep.get("metrics", {}).get("standalone_auc"),
            "auc_delta_recompute": abs(auc - float(rep.get("metrics", {}).get("standalone_auc", auc))),
            "delta_vs_control": None,  # filled below
            "delta_vs_v7_standalone": auc - champ_auc0,
            "blend_v7_auc": float(roc_auc_score(y0, blend_champ)),
            "blend_delta_vs_v7": float(roc_auc_score(y0, blend_champ)) - champ_auc0,
            "spearman_vs_v7": float(spearmanr(pred, champ0).statistic),
            "logit_pearson_vs_v7": float(np.corrcoef(logit(pred), logit(champ0))[0, 1]),
            "spearman_vs_route_ref": fr.get("spearman_vs_route_ref"),
            "frozen_gate": {
                "port_delta": fr.get("port_delta"),
                "spearman_vs_route_ref": fr.get("spearman_vs_route_ref"),
                "promising": fr.get("promising"),
                "rule": "port_delta >= +5e-5 AND spearman_vs_route_ref < 0.995",
            },
            "control_parity_ok": rep.get("control_parity_ok"),
            "eval_rows": rep.get("eval_rows"),
            "fit_rows": rep.get("fit_rows"),
            "train_rows": rep.get("train_rows"),
            "val_rows": rep.get("val_rows"),
            "features": len(rep.get("features", [])),
            "arch_cfg": rep.get("arch_cfg"),
            "train_cfg": rep.get("train_cfg"),
            "arm_params": rep.get("arm_params"),
            "prediction_sha256": sha,
            "prediction_sha256_matches_report": sha_ok,
            "total_seconds": rep.get("total_seconds"),
            "peak_vram_gb": rep.get("peak_vram_gb"),
        }
        preds[arm] = pred

    if "bce_control" in preds:
        ctrl_auc = arms["bce_control"]["standalone_auc_recomputed"]
        for arm, rec in arms.items():
            if rec.get("status") == "PENDING" or rec.get("standalone_auc_recomputed") is None:
                continue
            rec["delta_vs_control"] = rec["standalone_auc_recomputed"] - ctrl_auc
            rec["parity_vs_reference"] = (
                abs(rec["standalone_auc_recomputed"] - CONTROL_REFERENCE) <= 1e-4 if arm == "bce_control"
                else None)

    # cross-arm spearman (decorrelation structure among objectives)
    done = [a for a in ARMS if a in preds]
    cross = {}
    for i, a in enumerate(done):
        for b in done[i + 1:]:
            cross[f"{a}|{b}"] = round(float(spearmanr(preds[a], preds[b]).statistic), 6)

    # promotion decision per arm (frozen gate only decides)
    for arm, rec in arms.items():
        if rec.get("status") == "PENDING":
            rec["promotion"] = "not run"
            continue
        gate = rec["frozen_gate"]["promising"]
        rec["champion_admission_justified"] = bool(gate)
        rec["promotion"] = ("promote to 5-fold OOF (frozen gate passed)" if gate
                            else "close: frozen gate failed (screen is terminal)")

    out = {
        "experiment": "OBJ37-41 objective variants (fold-0 screens)",
        "prereg": "configs/phase20/obj37_41_objectives_prereg.json",
        "prereg_sha256": "94561fcea7ebb4315aac24a9928240ef6d464be9c13a4d35930be92c83a112b2",
        "eval_fold": 0,
        "eval_rows": int(m.sum()),
        "baseline": {
            "control_reference_auc": CONTROL_REFERENCE,
            "true_champion": CHAMPION,
            "true_champion_fold0_auc": champ_auc0,
            "route_ref_fold0_auc": float(roc_auc_score(y0, route0)),
            "note": "the frozen gate uses the route_ref proxy; champion-admission would use the true v7",
        },
        "arms": arms,
        "cross_arm_spearman": cross,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"true v7 fold-0 AUC {champ_auc0:.6f}; control reference {CONTROL_REFERENCE:.12f}")
    for arm in ARMS:
        rec = arms[arm]
        if rec.get("status") == "PENDING":
            print(f"  {arm:14s} PENDING")
            continue
        print(f"  {arm:14s} auc={rec['standalone_auc_recomputed']:.6f} "
              f"d_ctrl={rec['delta_vs_control']:+.2e} d_v7={rec['delta_vs_v7_standalone']:+.2e} "
              f"blend_d_v7={rec['blend_delta_vs_v7']:+.2e} spear_v7={rec['spearman_vs_v7']:.4f} "
              f"gate={rec['frozen_gate']['promising']} sha_ok={rec['prediction_sha256_matches_report']} "
              f"({rec['total_seconds']:.0f}s)")
    print(f"-> {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())