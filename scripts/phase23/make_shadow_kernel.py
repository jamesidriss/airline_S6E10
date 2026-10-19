"""Generate the parameterized variants of the raw8 kernel (separate Kaggle ids).

Variants (all generated from research/phase23/kaggle/raw8/main.py by frozen,
source-drift-guarded line replacements):

  raw8_shadow        arm W of the shadow confirmation:  raw geometry, 8 estimators,
                     shadow folds (seed 777001), two predeclared folds {2, 4}.
  raw8_shadow_route  arm R (control) of the shadow A/B: route geometry,
                     2 estimators, same shadow folds {2, 4}.
  raw8_route_repro   port-validation run on the PRIMARY fold 0 only: route
                     geometry, 2 estimators, to be compared against the
                     certified artifacts/phase17_B1/primary/route_oof.npy slice.

The shadow fold hash was verified against the immutable registry
(`get_scheme('shadow', ...)`, StratifiedKFold(5, shuffle, seed=777001)).

    .\\.venv\\Scripts\\python.exe scripts\\phase23\\make_shadow_kernel.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "research/phase23/kaggle/raw8"
SRC_MAIN = SRC / "main.py"
SRC_META = SRC / "kernel-metadata.json"

PRIMARY_FOLD_SHA = "b41f1d7ccf2fc504a1631bd8b84ba59b62b825895871949159a89f3f00a9f23b"
SHADOW_FOLD_SHA = "5d181e32357e67b2ad8381c33549ef905bb45b969de7b8e09a3e4e7cb24af758"

VARIANTS = {
    "raw8_shadow": {
        "tag": "shadow_",
        "fold_seed": 777001,
        "folds_sha": SHADOW_FOLD_SHA,
        "route": False,
        "n_est": 8,
        "run_folds": (2, 3),
        "kernel_id": "jamesripoll/s6e10-phase23-raw8-shadow-tabpfn35",
        "title": "s6e10 phase23 raw8 shadow tabpfn35",
    },
    "raw8_shadow_route": {
        "tag": "shadowR_",
        "fold_seed": 777001,
        "folds_sha": SHADOW_FOLD_SHA,
        "route": True,
        "n_est": 2,
        "run_folds": (2, 3),
        "kernel_id": "jamesripoll/s6e10-phase23-raw8-shadow-route-tabpfn35",
        "title": "s6e10 phase23 raw8 shadow route tabpfn35",
    },
    "raw8_route_repro": {
        "tag": "rrepro_",
        "fold_seed": 20261010,
        "folds_sha": PRIMARY_FOLD_SHA,
        "route": True,
        "n_est": 2,
        "run_folds": (0,),
        "kernel_id": "jamesripoll/s6e10-phase23-raw8-route-repro-tabpfn35",
        "title": "s6e10 phase23 raw8 route repro tabpfn35",
    },
}


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"source drift: {what}: expected 1 occurrence, found {text.count(old)}")
    return text.replace(old, new)


def gen_variant(name: str, spec: dict, source: str, meta: dict) -> dict:
    text = source
    text = replace_once(
        text,
        "PROBE = True                      # flipped to False for the production push",
        "PROBE = False                     # production (generated variant)",
        "PROBE",
    )
    text = replace_once(
        text,
        "TAG = ''                          # output prefix; the shadow variant sets 'shadow_'",
        f"TAG = '{spec['tag']}'" + " " * max(1, 34 - len(spec["tag"])) + "# generated variant",
        "TAG",
    )
    text = replace_once(
        text,
        "ROUTE_GEOMETRY = False            # True = 21 raw + Flight Distance coarse10 (route view)",
        "ROUTE_GEOMETRY = " + ("True " if spec["route"] else "False")
        + "           # True = 21 raw + Flight Distance coarse10 (route view)",
        "ROUTE_GEOMETRY",
    )
    text = replace_once(
        text,
        "RUN_FOLDS = (0, 1, 2, 3, 4)       # production folds (the reproduction probe uses (0,))",
        f"RUN_FOLDS = {tuple(spec['run_folds'])!r:<20} # generated variant",
        "RUN_FOLDS",
    )
    text = replace_once(
        text,
        "FOLD_SEED = 20261010              # immutable primary folds (shadow variant: 777001)",
        f"FOLD_SEED = {spec['fold_seed']:<22} # generated variant",
        "FOLD_SEED",
    )
    text = replace_once(
        text,
        f"EXPECT_FOLDS_SHA = '{PRIMARY_FOLD_SHA}'",
        f"EXPECT_FOLDS_SHA = '{spec['folds_sha']}'",
        "EXPECT_FOLDS_SHA",
    )
    if spec["n_est"] != 8:
        text = replace_once(
            text,
            "N_EST = 8                         # the hypothesis: eight internal estimators",
            f"N_EST = {spec['n_est']:<25} # generated variant",
            "N_EST",
        )
    # keep the docstring's mode note truthful
    text = text.replace(
        "PROBE = False  -- production: 40 units (5 folds x 8 members), then assemble.",
        f"PROBE = False  -- production: {len(spec['run_folds'])} fold(s) x {spec['n_est']} "
        f"member(s), route={spec['route']}, then assemble.",
    )

    dst = ROOT / "research/phase23/kaggle" / name
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "main.py").write_text(text, encoding="utf-8")
    out_meta = dict(meta)
    out_meta["id"] = spec["kernel_id"]
    out_meta["title"] = spec["title"]
    (dst / "kernel-metadata.json").write_text(json.dumps(out_meta, indent=2) + "\n",
                                              encoding="utf-8")
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": name,
        "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
        "source_main": "research/phase23/kaggle/raw8/main.py",
        "source_main_sha256": sha_text(source),
        "variant_main_sha256": sha_text(text),
        "primary_fold_sha256": PRIMARY_FOLD_SHA,
        "shadow_fold_sha256": SHADOW_FOLD_SHA,
        "note": ("Generated file; push with: python -m kaggle kernels push -p "
                 f"research/phase23/kaggle/{name} -t 39600"),
    }
    (dst / (manifest["variant"] + "_gen_manifest.json")).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return {"variant": name, "dir": str(dst.relative_to(ROOT)),
            "main_sha256": manifest["variant_main_sha256"][:16]}


def main() -> int:
    source = SRC_MAIN.read_text(encoding="utf-8")
    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    rows = [gen_variant(name, spec, source, meta) for name, spec in VARIANTS.items()]
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
