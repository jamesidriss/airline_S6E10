"""Hash-addressed OOS auxiliary distributions, with ordered row-id provenance."""
from __future__ import annotations

import json
import time
import shutil

import numpy as np

from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, save_json
from scripts.run_phase13 import RATINGS, build_aux


def probability_cache(Xfit, Xapply, names, fit_ids, apply_ids, fold_hash, seed=1,
                      rounds=250, inner_folds=3, label=""):
    fi, ai = np.asarray(fit_ids), np.asarray(apply_ids)
    if len(fi) != len(Xfit) or len(ai) != len(Xapply) or len(np.unique(fi)) != len(fi):
        raise ValueError("aux row-id alignment failure")
    if np.intersect1d(fi, ai).size:
        raise ValueError("aux fit/apply rows overlap")
    pos = [names.index(c) for c in RATINGS] + [names.index(c) for c in (
        "Gender", "Customer Type", "Type of Travel", "Class", "Age", "Flight Distance",
        "Departure Delay in Minutes", "Arrival Delay in Minutes")]
    contract = {"fit_ids_sha256": arr_sha256(fi), "apply_ids_sha256": arr_sha256(ai),
                "fit_raw_sha256": arr_sha256(Xfit[:, pos]), "apply_raw_sha256": arr_sha256(Xapply[:, pos]),
                "fold_sha256": fold_hash, "seed": seed, "rounds": rounds, "inner_folds": inner_folds,
                "builder_sha256": file_sha256("scripts/run_phase13.py"), "raw_names": [names[j] for j in pos]}
    from hashlib import sha256
    fingerprint = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    dest = ARTIFACTS / "aux_distribution" / fingerprint
    dest.mkdir(parents=True, exist_ok=True)
    meta = dest / "manifest.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        pf, pa = np.load(dest / "fit.npy", mmap_mode="r"), np.load(dest / "apply.npy", mmap_mode="r")
        assert arr_sha256(pf) == m["fit_probability_sha256"]
        assert arr_sha256(pa) == m["apply_probability_sha256"]
        print(f"{label}: aux probability cache {fingerprint[:12]}", flush=True)
        return pf, pa, m
    required_bytes = (len(fi) + len(ai)) * 13 * 6 * 8 + 2 * 1024**3
    if shutil.disk_usage(dest).free < required_bytes:
        raise RuntimeError('Insufficient disk reserve for a complete auxiliary probability cache')
    start = time.monotonic()

    def progress(j, info):
        save_json({"contract": contract, "git": git_commit(), "completed_ratings": j + 1,
                   "last_rating": info, "elapsed_seconds": time.monotonic() - start}, dest / "progress.json")
        print(f"{label}: aux {j + 1}/13 {info['rating']} {info['seconds']:.1f}s", flush=True)
        if shutil.disk_usage(dest).free < required_bytes:
            raise RuntimeError('Disk reserve fell below auxiliary cache safety margin; no result claimed')

    pf, pa, info = build_aux(Xfit, Xapply, names, seed, rounds=rounds,
                              inner_folds=inner_folds, return_probabilities=True, progress=progress)
    # Never expose a truncated array as a completed cache component.
    for name, array in (('fit', pf), ('apply', pa)):
        temporary = dest / f'{name}.partial.npy'
        np.save(temporary, array)
        temporary.replace(dest / f'{name}.npy')
    np.save(dest / "fit_ids.npy", fi)
    np.save(dest / "apply_ids.npy", ai)
    m = {"contract": contract, "git": git_commit(), "fingerprint": fingerprint,
         "fit_probability_sha256": arr_sha256(pf), "apply_probability_sha256": arr_sha256(pa),
         "ratings": info, "seconds": time.monotonic() - start,
         "provenance": "auxiliary rating targets only; inner OOS fit, outer-fit-only apply model",
         "path": str(dest)}
    save_json(m, meta)
    return pf, pa, m
