"""Phase-23 frozen shadow-fold evaluation of the native raw8 member.

Consumes the raw8 shadow Kaggle outputs plus the CERTIFIED phase-17 shadow arms
(route member set + auxiliary10 mean logits), reproduces the certified
composition formula, and writes reports/phase23/raw8_shadow_comparison.json once.

Frozen by configs/phase23/raw8_shadow_prereg.json (sha 05b7235e...); refuses to
overwrite; makes no submissions; recomputes every statistic from durable
per-member logits.

    .\\.venv\\Scripts\\python.exe scripts\\phase23\\compare_raw8_shadow.py
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
from src.validation.compare import logit  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.validation.paired_auc import paired_auc  # noqa: E402

PROTOCOL = ROOT / "configs/phase23/raw8_shadow_prereg.json"
PROTOCOL_SHA = "05b7235ed55e6d8a9d891c3b9d6ab73695c38573156bb06a826d9aa896d3df29"
OUT = ROOT / "reports/phase23/raw8_shadow_comparison.json"
N_TRAIN = 699635
FOLDS = (2, 3)
N_MEMBERS_W = 8
N_MEMBERS_R = 2
REUSE_TOL = 2e-6
AGREE_TOL = 1e-6
CONFIRM_DELTA_MIN = 1.0e-5
CONFIRM_SE_MULT = 2.5

UNITS = ROOT / "artifacts/phase23/raw8_shadow/units"
ASSEMBLED = ROOT / "artifacts/phase23/raw8_shadow/shadow_raw8_oof.npy"
ROUTE_DIR = ROOT / "artifacts/phase17_shadow_B1"
AUX_DIR = ROOT / "artifacts/phase17_shadow_aux10"


def load_member_prob_mean(fold: int, n_members: int, label: str):
    probs, logits, metas = [], [], []
    for m in range(n_members):
        vp = UNITS / f"f{fold}_m{m}_val_logits.npy"
        mp = UNITS / f"f{fold}_m{m}_meta.json"
        if not (vp.exists() and mp.exists()):
            raise SystemExit(f"missing unit f{fold} m{m} ({label})")
        lg = np.load(vp, allow_pickle=False).astype(np.float64)
        assert lg.ndim == 2 and lg.shape[1] == 2, f"bad unit logits shape {lg.shape}"
        metas.append(json.loads(mp.read_text(encoding="utf-8")))
        logits.append(lg[:, 1] - lg[:, 0])
        probs.append(expit(lg[:, 1] - lg[:, 0]))
    return np.mean(probs, axis=0), np.mean(logits, axis=0), metas


def main() -> int:
    started = time.monotonic()
    assert file_sha256(PROTOCOL) == PROTOCOL_SHA, "prereg hash drift"
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert not OUT.exists(), f"refuse to overwrite {OUT}"

    tr, _te = load_cached_parquet()
    y = tr.satisfaction.to_numpy(dtype="int8")
    ids = tr.id.to_numpy()
    assert len(y) == N_TRAIN
    shadow = get_scheme("shadow", y, ids).folds
    assert arr_sha256(shadow) == spec["design"]["folds"]["fold_hash_sha256"], "shadow fold drift"

    report = {
        "status": "COMPLETE",
        "protocol": spec["protocol"],
        "prereg_sha256": PROTOCOL_SHA,
        "source_sha256": file_sha256(Path(__file__)),
        "git": git_commit(),
        "utc_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "folds": list(FOLDS),
        "gates": {
            "confirm_delta_min": CONFIRM_DELTA_MIN,
            "confirm_se_multiple": CONFIRM_SE_MULT,
            "reuse_tolerance": REUSE_TOL,
            "agreement_tolerance": AGREE_TOL,
        },
    }

    y_parts, rows_parts = [], []
    r_parts, w_parts, aux_parts, cert_parts = [], [], [], []
    per_fold = {}
    reuse_check, agree_check = {}, {}

    for f in FOLDS:
        rows = np.flatnonzero(shadow == f)
        apply_ids = np.load(ROUTE_DIR / f"f{f}/apply_ids.npy")
        if not np.array_equal(rows, apply_ids):
            raise SystemExit(f"row alignment mismatch on shadow fold {f}")
        aux = np.load(AUX_DIR / f"aux10_mean_logits_f{f}.npy").astype(np.float64)
        r_logits = np.stack([
            (lambda a: a[0, :, 1] - a[0, :, 0])(np.load(ROUTE_DIR / f"f{f}/member_{m}_raw_logits.npy"))
            for m in range(N_MEMBERS_R)
        ])
        r_prob = np.mean(expit(r_logits), axis=0)
        port_r = expit(0.5 * logit(r_prob) + 0.5 * aux)
        certified = np.load(ROUTE_DIR / f"f{f}/portfolio.npy").astype(np.float64)
        reuse_check[f"f{f}"] = float(np.max(np.abs(port_r - certified)))

        w_prob, w_logit_mean, metas = load_member_prob_mean(f, N_MEMBERS_W, "W")
        if len(rows) != len(w_prob):
            raise SystemExit(f"unit length mismatch on shadow fold {f}")
        agree = None
        if ASSEMBLED.exists():
            assembled = np.load(ASSEMBLED, allow_pickle=False)
            if assembled.shape == (N_TRAIN,):
                agree = float(np.max(np.abs(w_prob - assembled[rows].astype(np.float64))))
        agree_check[f"f{f}"] = agree

        port_w = expit(0.5 * logit(w_prob) + 0.5 * aux)
        per_fold[f"f{f}"] = {
            "n_rows": int(len(rows)),
            "auc_route_member": float(roc_auc_score(y[rows], r_prob)),
            "auc_raw8_member": float(roc_auc_score(y[rows], w_prob)),
            "auc_portfolio_route": float(roc_auc_score(y[rows], port_r)),
            "auc_portfolio_raw8": float(roc_auc_score(y[rows], port_w)),
            "member_delta": float(roc_auc_score(y[rows], w_prob) - roc_auc_score(y[rows], r_prob)),
            "portfolio_delta": float(roc_auc_score(y[rows], port_w) - roc_auc_score(y[rows], port_r)),
            "unit_fit_seconds": [m.get("t_fit") for m in metas],
            "unit_variants": [m.get("variant") for m in metas],
        }
        y_parts.append(y[rows]); rows_parts.append(rows)
        r_parts.append(r_prob); w_parts.append(w_prob); aux_parts.append(aux)
        cert_parts.append(certified)

    y_u = np.concatenate(y_parts)
    r_u = np.concatenate(r_parts)
    w_u = np.concatenate(w_parts)
    aux_u = np.concatenate(aux_parts)
    cert_u = np.concatenate(cert_parts)
    port_r_u = expit(0.5 * logit(r_u) + 0.5 * aux_u)
    port_w_u = expit(0.5 * logit(w_u) + 0.5 * aux_u)

    e1 = paired_auc(y_u, w_u.astype(np.float32), r_u.astype(np.float32))
    e2 = paired_auc(y_u, port_w_u.astype(np.float32), port_r_u.astype(np.float32))

    report.update(
        reuse_check=reuse_check,
        agreement_check=agree_check,
        per_fold=per_fold,
        endpoint_E1_member={
            "pooled": e1,
            "auc_raw8": float(roc_auc_score(y_u, w_u)),
            "auc_route": float(roc_auc_score(y_u, r_u)),
            "logit_corr": float(np.corrcoef(logit(r_u), logit(w_u))[0, 1]),
            "spearman": float(spearmanr(r_u, w_u).statistic),
        },
        endpoint_E2_portfolio={
            "pooled": e2,
            "auc_raw8_portfolio": float(roc_auc_score(y_u, port_w_u)),
            "auc_route_portfolio": float(roc_auc_score(y_u, cert_u)),
        },
        baseline_check_portfolio_route={
            "recon_vs_certified_max": float(np.max(np.abs(port_r_u - cert_u))),
        },
        vector_sha256={
            "arm_W_prob_mean": arr_sha256(w_u.astype(np.float32)),
            "arm_R_prob_mean": arr_sha256(r_u.astype(np.float32)),
            "aux_mean_logits": arr_sha256(aux_u),
            "portfolio_W": arr_sha256(port_w_u.astype(np.float32)),
            "portfolio_R_certified": arr_sha256(cert_u.astype(np.float32)),
        },
        limitations=[
            "Two predeclared shadow folds only (279,854 rows); conditional saved-model uncertainty.",
            "The certified route/aux shadow arms were produced by the phase-17 local run; reused unmodified and hash-checked.",
        ],
    )

    invalid = [k for k, v in reuse_check.items() if v > REUSE_TOL]
    if invalid:
        report["status"] = "OUTCOME_INVALID"
        report["reason"] = f"reuse validity gate failed on {invalid}: {reuse_check}"
        save_json(report, OUT)
        print(json.dumps({"status": report["status"], "reuse": reuse_check}, indent=1))
        return 1

    e2_delta = e2["delta"]
    e2_se = e2["paired_se"]
    both_positive = all(v["portfolio_delta"] > 0 for v in per_fold.values())
    confirmed = bool(
        e2_delta >= CONFIRM_DELTA_MIN
        and e2_delta >= CONFIRM_SE_MULT * e2_se
        and both_positive
    )
    report["decision"] = {
        "confirmed": confirmed,
        "rule": "E2 delta >= 1.0e-5 and >= 2.5*SE and both folds positive",
        "e2_delta": e2_delta,
        "e2_se": e2_se,
        "both_folds_positive": both_positive,
        "verdict": "SHADOW_CONFIRMED" if confirmed else "SHADOW_NOT_CONFIRMED",
    }
    report["CPU_wall_seconds"] = time.monotonic() - started
    report["GPU_hours"] = 0
    report["new_submissions"] = 0
    save_json(report, OUT)
    print(json.dumps({
        "status": report["status"],
        "E1": {"delta": e1["delta"], "se": e1["paired_se"]},
        "E2": {"delta": e2_delta, "se": e2_se},
        "per_fold_portfolio_delta": {k: v["portfolio_delta"] for k, v in per_fold.items()},
        "verdict": report["decision"]["verdict"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
