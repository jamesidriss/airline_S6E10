"""Phase 24 -- OOF -> public transfer table over every submitted candidate.

Reads the remote submission list (reports/phase24/_kaggle_submissions.csv, fetched with
`kaggle competitions submissions -c playground-series-s6e10 --csv`) with the stdlib csv
reader, because the description field contains embedded newlines that defeat the fast
pandas parser. Joins on the CSV file stem to the local manifest and reports the OOF ->
public shift per submission plus the model-specific spread the finalist decision depends on.

    .\\.venv\\Scripts\\python.exe reports/phase24/oof_public_transfer.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REMOTE = ROOT / "reports/phase24/kaggle_submissions_20261020.csv"
MANIFEST = ROOT / "submissions/manifest.csv"
OUT = ROOT / "reports/phase24/oof_public_transfer.json"
PUBLIC_ROWS = 59969
PRIVATE_ROWS = 239875
# v5 predates the manifest's coverage; its OOF is recorded in the ledger entry
# p14_S1_V5_AUX_CROSS_PUBLIC_0_96103_TRANSFER_CONFIRMED.
FALLBACK_OOF = {"v5_aux_cross": 0.961523}


def read_remote() -> pd.DataFrame:
    with REMOTE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    df["stem"] = df["fileName"].astype(str).str.replace(r"\.csv$", "", regex=True)
    df["ref"] = df["ref"].astype(int)
    df["public"] = pd.to_numeric(df["publicScore"], errors="coerce")
    df["complete"] = df["status"].astype(str).str.contains("COMPLETE")
    return df


def main() -> None:
    remote = read_remote()
    manifest = pd.read_csv(MANIFEST).drop_duplicates("name", keep="last")
    j = remote.merge(manifest[["name", "oof_auc"]], left_on="stem", right_on="name", how="left")
    j = j.dropna(subset=["public"])
    j["oof_auc"] = j["oof_auc"].fillna(j["stem"].map(FALLBACK_OOF))
    j = j.dropna(subset=["oof_auc"]).sort_values("oof_auc")
    j["shift"] = j["public"] - j["oof_auc"]

    assert j["ref"].is_unique, "duplicate submission refs after join"
    shifts = j["shift"].to_numpy(float)
    common = float(shifts.mean())
    resid = shifts - common

    oof = j["oof_auc"].to_numpy(float)
    pub = j["public"].to_numpy(float)
    slope, intercept = np.polyfit(oof, pub, 1)
    r = float(np.corrcoef(oof, pub)[0, 1])

    # paired-model transfer scale: residual sd around the common shift, and its private-size
    # counterpart if the residual is dominated by row sampling (sqrt of row-count ratio).
    resid_sd = float(resid.std(ddof=1))
    scale = np.sqrt(PUBLIC_ROWS / PRIVATE_ROWS)
    pair_sd_public = float(np.sqrt(2) * resid_sd)
    pair_sd_private = float(pair_sd_public * scale)

    v7 = j[j["stem"] == "v7_sol_tabpfn_internal2_aux10"].iloc[0]
    v6 = j[j["stem"] == "v6_sol_tabpfn_route_aux10"].iloc[0]
    delta_oof = float(v7.oof_auc - v6.oof_auc)
    delta_public = float(v7.public - v6.public)

    def p_v6_beats(expected_delta: float) -> float:
        from math import erf, sqrt

        z = expected_delta / pair_sd_private
        return float(0.5 * (1.0 + erf(-z / sqrt(2))))

    out = {
        "n_submissions": int(len(j)),
        "all_complete": bool(remote["complete"].all()),
        "common_shift_mean": common,
        "shift_sd": float(shifts.std(ddof=1)),
        "model_specific_resid_sd_public": resid_sd,
        "paired_pair_sd_public": pair_sd_public,
        "paired_pair_sd_private_if_sampling_dominated": pair_sd_private,
        "oof_on_public_regression": {"slope": float(slope), "intercept": float(intercept), "pearson_r": r},
        "v7_vs_v6": {
            "delta_oof": delta_oof,
            "delta_public": delta_public,
            "public_minus_oof_advantage_for_v6": float(delta_oof - delta_public),
            "p_v6_beats_v7_oof_only": p_v6_beats(delta_oof),
            "p_v6_beats_v7_half_shrunk": p_v6_beats(delta_oof - 0.5 * (delta_oof - delta_public)),
            "p_v6_beats_v7_public_differential_real": p_v6_beats(delta_public),
        },
        "rows": [
            {
                "ref": int(row.ref),
                "file": row.stem,
                "oof": float(row.oof_auc),
                "public": float(row.public),
                "shift": float(row.shift),
                "shift_minus_common": float(row.shift - common),
            }
            for row in j.itertuples()
        ],
        "reading": (
            "The common component is the public split being harder than the OOF contexts; it cancels in "
            "paired model comparisons. The model-specific residual sd measures how far a submission moves "
            "relative to the cohort once the split is fixed. The probability of v6 beating v7 on private "
            "is reported under three transfer models: OOF only, half-shrunk public differential, and the "
            "public differential taken at face value."
        ),
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{'ref':>9} {'file':46} {'OOF':>10} {'public':>9} {'shift':>10} {'resid':>10}")
    for row in j.itertuples():
        print(f"{int(row.ref):9d} {row.stem:46} {row.oof_auc:10.6f} {row.public:9.5f} "
              f"{row.shift:+10.2e} {row.shift - common:+10.2e}")
    print()
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"refuse to overwrite {OUT}")
    main()
