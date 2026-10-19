"""Phase-23 frozen evaluation of OUR OWN native raw8 TabPFN-3.5 member.

Read-only: consumes the downloaded Kaggle artefacts, reproduces the frozen
statistics of scripts/phase22/compare.py (same estimators, same simulation
partitions) on the new candidate list, and writes
reports/phase23/raw8_comparison.json exactly once.

Refuses to run if the output exists or the frozen prereg hash drifts.
Makes no submissions and touches no protected file.

    .\.venv\Scripts\python.exe scripts\phase23\compare_raw8.py
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.submission.store import load_oof  # noqa: E402
from src.validation.compare import logit  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.validation.paired_auc import holm_adjust, paired_auc  # noqa: E402
from src.validation.private_sim import RankedAUC, summarize  # noqa: E402
from scripts.private_lb_simulator import make_partition  # noqa: E402

PROTOCOL = ROOT / "configs/phase23/raw8_kaggle_prereg.json"
PROTOCOL_SHA = "b9aec0ce67aa0cef33f6f8e9a546d85af8f5a9aa7aebc6667e038a65a87f8fe0"
OUT = ROOT / "reports/phase23/raw8_comparison.json"
N_TRAIN = 699635
N_TEST = 299844
W_GRID = (0.10, 0.25, 0.50)


def load_vector(path: Path, n: int, what: str):
    p = np.load(path, allow_pickle=False)
    assert p.shape == (n,), f"{what}: bad shape {p.shape}"
    assert np.isfinite(p).all(), f"{what}: non-finite values"
    assert ((p >= 0) & (p <= 1)).all(), f"{what}: outside [0,1]"
    return p.astype(np.float64)


def unit_audit(units_dir: Path):
    rows = []
    if units_dir.exists():
        for f in sorted(units_dir.glob("*_meta.json")):
            meta = json.loads(f.read_text(encoding="utf-8"))
            entry = {
                "tag": f.name.replace("_meta.json", ""),
                "fold": int(meta["fold"]),
                "member": int(meta["member"]),
                "variant": int(meta.get("variant", -1)),
                "t_fit": meta.get("t_fit"),
                "t_val": meta.get("t_val"),
                "t_test": meta.get("t_test"),
                "val_logits_sha256": meta.get("val_logits_sha256"),
                "test_logits_sha256": meta.get("test_logits_sha256"),
                "meta_sha256": file_sha256(f),
            }
            rows.append(entry)
    per_fold = {}
    for e in rows:
        per_fold.setdefault(e["fold"], []).append(e["member"])
    counts = {k: len(v) for k, v in sorted(per_fold.items())}
    test_present = sum(
        1
        for e in rows
        if (units_dir / f"{e['tag']}_test_logits.npy").exists()
    )
    return rows, counts, test_present


def evaluate_row(name, p, y, folds, base, zbase, baseline_folds, masks, pub0, priv0):
    pair = paired_auc(y, p, base)
    d = np.array(
        [
            roc_auc_score(y[folds == k], p[folds == k]) - baseline_folds[k]
            for k in range(5)
        ]
    )
    se = float(d.std(ddof=1) / np.sqrt(5))
    score = RankedAUC(y, p)
    pub = [score.auc(m == 1) - b for m, b in zip(masks, pub0)]
    priv = [score.auc(m == 2) - b for m, b in zip(masks, priv0)]
    row = {
        "name": name,
        "pooled": pair,
        "paired_fold_deltas": d.tolist(),
        "mean_fold_delta": float(d.mean()),
        "fold_se": se,
        "fold_t": float(d.mean() / se) if se else None,
        "positive_folds": int((d > 0).sum()),
        "private80": summarize(pub, priv, pair["delta"]),
        "standalone_auc": float(roc_auc_score(y, p)),
        "logit_corr": float(np.corrcoef(zbase, logit(p))[0, 1]),
        "spearman": float(spearmanr(base, p).statistic),
        "prediction_sha256": arr_sha256(p),
    }
    print(
        json.dumps(
            {
                "name": name,
                "delta": pair["delta"],
                "mean_fold_delta": row["mean_fold_delta"],
                "positive_folds": row["positive_folds"],
                "standalone_auc": row["standalone_auc"],
            }
        ),
        flush=True,
    )
    return row


def main():
    started = time.monotonic()
    assert file_sha256(PROTOCOL) == PROTOCOL_SHA, "prereg hash drift"
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert not OUT.exists(), f"refuse to overwrite {OUT}"

    tr, te = load_cached_parquet()
    y = tr.satisfaction.to_numpy(dtype="int8")
    ids = tr.id.to_numpy()
    assert len(y) == N_TRAIN and len(te) == N_TEST
    folds = get_scheme("primary", y, ids).folds
    base = load_oof(spec["champion"])
    assert arr_sha256(base) == "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552", "champion drift"
    zbase = logit(base)
    baseline_auc = float(roc_auc_score(y, base))

    raw8 = load_vector(ROOT / spec["inputs"]["raw8_oof"], N_TRAIN, "raw8_oof")
    route = load_vector(ROOT / spec["inputs"]["route2_oof"], N_TRAIN, "route2_oof")
    aux = load_vector(ROOT / spec["inputs"]["aux10_oof"], N_TRAIN, "aux10_oof")
    test_path = ROOT / spec["inputs"]["raw8_test"]
    test_vec = None
    if test_path.exists():
        test_vec = load_vector(test_path, N_TEST, "raw8_test")

    units = ROOT / spec["inputs"]["raw8_unit_metas"]
    rows_meta, per_fold_counts, test_present = unit_audit(units)
    complete = all(per_fold_counts.get(k, 0) == 8 for k in range(5))
    has_nan = bool(np.isnan(raw8).any())

    composition = float(np.max(np.abs(expit((logit(route) + logit(aux)) / 2).astype("float32") - base)))

    report = {
        "status": "COMPLETE",
        "protocol": spec["protocol"],
        "prereg_sha256": PROTOCOL_SHA,
        "source_sha256": file_sha256(Path(__file__)),
        "git": git_commit(),
        "utc_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_auc": baseline_auc,
        "baseline_sha256": arr_sha256(base),
        "raw8": {
            "oof_sha256": arr_sha256(raw8),
            "test_sha256": arr_sha256(test_vec) if test_vec is not None else None,
            "standalone_auc": float(roc_auc_score(y, raw8)),
            "fold_auc": [float(roc_auc_score(y[folds == k], raw8[folds == k])) for k in range(5)],
            "route2_fold_auc": [float(roc_auc_score(y[folds == k], route[folds == k])) for k in range(5)],
            "logit_corr_vs_v7": float(np.corrcoef(zbase, logit(raw8))[0, 1]),
            "spearman_vs_v7": float(spearmanr(base, raw8).statistic),
            "fold_member_counts": {str(k): per_fold_counts.get(k, 0) for k in range(5)},
            "units": rows_meta,
            "test_units_present": test_present,
        },
        "composition_check": {
            "max_abs_delta_route_aux_vs_v7": composition,
            "tolerance": 1e-5,
            "pass": composition <= 1e-5,
        },
        "simulations": spec["simulations"],
        "external_reference": spec["external_reference"],
        "limitations": [
            "Conditional saved-OOF diagnostics on the immutable primary folds; repeated model selection and fit variation excluded.",
            "The candidate is a new member produced by our own Kaggle run; its only admissible use is as an ensemble member, never as a standalone submission claim.",
        ],
    }

    if has_nan or not complete:
        report["status"] = "OUTCOME_PARTIAL"
        report["reason"] = (
            "NaN rows in raw8_oof" if has_nan else f"incomplete fold member counts {per_fold_counts}"
        )
        save_json(report, OUT)
        print(json.dumps({"status": report["status"], "reason": report["reason"]}))
        return 0

    cfg = spec["simulations"]
    masks = [
        make_partition(y, folds, cfg["seed"] + i * 104729, "random_stratified", cfg["population"])
        for i in range(cfg["repeats"])
    ]
    assert all(
        int((m == 1).sum()) == cfg["public_rows"] and int((m == 2).sum()) == cfg["private_rows"]
        for m in masks
    )
    scorer = RankedAUC(y, base)
    pub0 = [scorer.auc(m == 1) for m in masks]
    priv0 = [scorer.auc(m == 2) for m in masks]
    baseline_folds = [roc_auc_score(y[folds == k], base[folds == k]) for k in range(5)]

    z8 = logit(raw8)
    candidates = [
        (f"raw8_w{w:.2f}", expit((1 - w) * zbase + w * z8).astype("float32"))
        for w in W_GRID
    ]
    if report["composition_check"]["pass"]:
        candidates.append(("port_raw8", expit((z8 + logit(aux)) / 2).astype("float32")))
    else:
        report["composition_check"]["note"] = "port_raw8 dropped (composition check failed)"

    rows = [
        evaluate_row(name, p, y, folds, base, zbase, baseline_folds, masks, pub0, priv0)
        for name, p in candidates
    ]
    nominations = []
    pvals = holm_adjust([r["pooled"]["one_sided_p"] for r in rows])
    for r, pval in zip(rows, pvals):
        r["holm_p"] = float(pval)
        passed = (
            r["mean_fold_delta"] >= 1.5e-5
            and r["mean_fold_delta"] >= 2.5 * r["fold_se"]
            and r["positive_folds"] == 5
            and pval <= 0.05
            and r["private80"]["private_p05"] >= 0
        )
        r["numerical_gate_pass"] = bool(passed)
        r["verdict"] = "NOMINATE_MATCHED_SHADOW" if passed else "NO_ADMISSION"
        if passed:
            nominations.append(r["name"])
    report["candidates"] = rows
    report["nominations"] = nominations
    report["mask_sha256"] = [arr_sha256(m) for m in masks]
    report["CPU_wall_seconds"] = time.monotonic() - started
    report["GPU_hours"] = 0
    report["new_submissions"] = 0
    save_json(report, OUT)
    print(json.dumps({"status": "COMPLETE", "nominations": nominations,
                      "seconds": round(time.monotonic() - started, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
