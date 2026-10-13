"""H18-A stage S1 — paired comparison of treatment vs control arms.

Reads the two completed arm reports (reports/phase18/h18a_s1_f{fold}_{arm}.json)
plus their saved prediction arrays, verifies eval-row identity, and computes the
paired delta with an exact pair-credit decomposition. Discovery gate:
treatment - control > 0 on the eval fold AND both arms COMPLETE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, arr_sha256, save_json  # noqa: E402
from scripts.phase16_pair_metrics import pair_rescue_damage  # noqa: E402

OUT = ROOT / "reports" / "phase18"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=range(10), required=True)
    args = ap.parse_args()
    rep = {}
    preds = {}
    for arm in ("control", "treatment"):
        r = json.loads((OUT / f"h18a_s1_f{args.fold}_{arm}.json").read_text(encoding="utf-8"))
        if r["status"] != "COMPLETE":
            raise SystemExit(f"arm {arm} status {r['status']} — no comparison")
        p = np.load(ARTIFACTS / "phase18" / f"h18a_s1_f{args.fold}_{arm}" / "prediction.npy")
        if arr_sha256(p) != r["prediction_sha256"]:
            raise SystemExit(f"arm {arm} prediction hash mismatch — refuse to compare")
        rep[arm], preds[arm] = r, p

    if rep["control"]["eval_ids_sha256"] != rep["treatment"]["eval_ids_sha256"]:
        raise SystemExit("eval rows differ between arms — not a matched comparison")
    eval_ids = np.load(ARTIFACTS / "phase18" / f"h18a_s1_f{args.fold}_control" / "eval_ids.npy")
    y = pd.read_parquet(ROOT / "data/cache/train.parquet",
                        columns=["id", "satisfaction"])
    y["row"] = np.arange(len(y))
    m = y.set_index("id").reindex(eval_ids)
    assert m["satisfaction"].notna().all()
    yt = m["satisfaction"].to_numpy()

    auc_c = roc_auc_score(yt, preds["control"])
    auc_t = roc_auc_score(yt, preds["treatment"])
    delta = auc_t - auc_c
    prd = pair_rescue_damage(yt, preds["treatment"], preds["control"])
    total = int(yt.sum()) * int((1 - yt).sum())
    assert abs(prd["net_auc_gain"] - delta) < 1e-12

    result = {
        "hypothesis_id": "H18-A", "stage": "S1_matched_discovery",
        "eval_fold_block10": args.fold,
        "eval_rows": int(len(eval_ids)),
        "auc_control": auc_c, "auc_treatment": auc_t, "delta": delta,
        "pair_rescue_damage": prd,
        "discovery_pass": bool(delta > 0),
        "control_context_rows": rep["control"]["context_rows"],
        "treatment_context_rows": rep["treatment"]["context_rows"],
        "only_variable_verified": rep["control"]["params"] == rep["treatment"]["params"]
        and rep["control"]["model_seed"] == rep["treatment"]["model_seed"],
        "report_sha256": {a: arr_sha256(preds[a]) for a in ("control", "treatment")},
    }
    save_json(result, OUT / f"h18a_s1_f{args.fold}_comparison.json")
    print(json.dumps({k: result[k] for k in ("eval_fold_block10", "auc_control",
                                             "auc_treatment", "delta", "discovery_pass")}, indent=2))


if __name__ == "__main__":
    main()
