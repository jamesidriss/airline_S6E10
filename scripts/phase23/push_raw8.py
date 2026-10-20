"""Guarded push of the raw8 production kernel (and its variants).

Why this exists: a wrong push costs ~7-19 GPU-hours of a 30 h/week free quota and,
worse, could produce a result that looks certified but was produced by the wrong
configuration. This script asserts every frozen constant of the requested variant,
verifies the kernel metadata, records a push manifest, and only then (with --push)
calls the Kaggle CLI.

    # dry run (default): print the plan, write the manifest, push nothing
    .\\.venv\\Scripts\\python.exe scripts\\phase23\\push_raw8.py --variant raw8

    # production push (flips PROBE=False in research/phase23/kaggle/raw8/main.py)
    .\\.venv\\Scripts\\python.exe scripts\\phase23\\push_raw8.py --variant raw8 --push --timeout-s 39600

    # shadow arm / control / repro variants (already generated with PROBE=False)
    .\\.venv\\Scripts\\python.exe scripts\\phase23\\push_raw8.py --variant raw8_shadow --push
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import file_sha256  # noqa: E402

PREREG_SHA = "b9aec0ce67aa0cef33f6f8e9a546d85af8f5a9aa7aebc6667e038a65a87f8fe0"
SHADOW_PREREG_SHA = "05b7235ed55e6d8a9d891c3b9d6ab73695c38573156bb06a826d9aa896d3df29"
PRIMARY_FOLD_SHA = "b41f1d7ccf2fc504a1631bd8b84ba59b62b825895871949159a89f3f00a9f23b"
SHADOW_FOLD_SHA = "5d181e32357e67b2ad8381c33549ef905bb45b969de7b8e09a3e4e7cb24af758"

VARIANTS = {
    "raw8": {
        "dir": "research/phase23/kaggle/raw8",
        "kernel_id": "jamesripoll/s6e10-phase23-raw8-tabpfn35",
        "tag": "",
        "proto": "PROBE = False",
        "route": False,
        "folds": (0, 1, 2, 3, 4),
        "fold_seed": 20261010,
        "folds_sha": PRIMARY_FOLD_SHA,
        "n_est": 8,
        "prereg": PREREG_SHA,
    },
    "raw8_shadow": {
        "dir": "research/phase23/kaggle/raw8_shadow",
        "kernel_id": "jamesripoll/s6e10-phase23-raw8-shadow-tabpfn35",
        "tag": "shadow_",
        "proto": "PROBE = False",
        "route": False,
        "folds": (2, 3),
        "fold_seed": 777001,
        "folds_sha": SHADOW_FOLD_SHA,
        "n_est": 8,
        "prereg": SHADOW_PREREG_SHA,
    },
    "raw8_shadow_route": {
        "dir": "research/phase23/kaggle/raw8_shadow_route",
        "kernel_id": "jamesripoll/s6e10-phase23-raw8-shadow-route-tabpfn35",
        "tag": "shadowR_",
        "proto": "PROBE = False",
        "route": True,
        "folds": (2, 3),
        "fold_seed": 777001,
        "folds_sha": SHADOW_FOLD_SHA,
        "n_est": 2,
        "prereg": SHADOW_PREREG_SHA,
    },
    "raw8_route_repro": {
        "dir": "research/phase23/kaggle/raw8_route_repro",
        "kernel_id": "jamesripoll/s6e10-phase23-raw8-route-repro-tabpfn35",
        "tag": "rrepro_",
        "proto": "PROBE = False",
        "route": True,
        "folds": (0,),
        "fold_seed": 20261010,
        "folds_sha": PRIMARY_FOLD_SHA,
        "n_est": 2,
        "prereg": PREREG_SHA,
    },
}


WANT = {"PROBE", "TAG", "ROUTE_GEOMETRY", "RUN_FOLDS", "FOLD_SEED",
        "EXPECT_FOLDS_SHA", "N_EST"}


def parse_constants(text: str) -> dict:
    """Read module-level constants with ast (immune to docstrings/comments)."""
    tree = ast.parse(text)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in WANT:
            out[node.targets[0].id] = (ast.literal_eval(node.value), node.lineno)
    missing = WANT - set(out)
    if missing:
        raise SystemExit(f"constants not found in kernel source: {sorted(missing)}")
    return out


def flip_probe(text: str) -> str:
    """Set PROBE=False on its assignment line only (docstring untouched)."""
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "PROBE":
            if ast.literal_eval(node.value) is False:
                return text
            lines = text.splitlines()
            lines[node.lineno - 1] = "PROBE = False  # flipped by push_raw8.py (guarded)"
            body = "\n".join(lines)
            return body + ("\n" if text.endswith("\n") else "")
    raise SystemExit("PROBE assignment not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--push", action="store_true", help="actually call the Kaggle CLI")
    ap.add_argument("--timeout-s", type=int, default=39600)
    ap.add_argument("--allow-probe-flip", action="store_true",
                    help="required to edit PROBE=True -> False in the kernel source")
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    kdir = ROOT / spec["dir"]
    main_py = kdir / "main.py"
    meta_py = kdir / "kernel-metadata.json"
    if not main_py.exists() or not meta_py.exists():
        raise SystemExit(f"missing kernel files under {kdir}")

    assert file_sha256(ROOT / "configs/phase23/raw8_kaggle_prereg.json") == PREREG_SHA, \
        "primary prereg hash drift"
    assert file_sha256(ROOT / "configs/phase23/raw8_shadow_prereg.json") == SHADOW_PREREG_SHA, \
        "shadow prereg hash drift"

    text = main_py.read_text(encoding="utf-8")
    const = parse_constants(text)
    if const["PROBE"][0] is True:
        if not args.allow_probe_flip:
            raise SystemExit("kernel has PROBE=True; pass --allow-probe-flip to switch to production")
        text = flip_probe(text)
        main_py.write_text(text, encoding="utf-8")
        const = parse_constants(text)

    expected = {
        "PROBE": False,
        "TAG": spec["tag"],
        "ROUTE_GEOMETRY": spec["route"],
        "RUN_FOLDS": tuple(spec["folds"]),
        "FOLD_SEED": spec["fold_seed"],
        "EXPECT_FOLDS_SHA": spec["folds_sha"],
        "N_EST": spec["n_est"],
    }
    bad = {k: {"found": const[k][0], "expected": v}
           for k, v in expected.items() if const[k][0] != v}
    if bad:
        raise SystemExit(f"configuration mismatch for {args.variant}: {bad}")

    meta = json.loads(meta_py.read_text(encoding="utf-8"))
    assert meta["id"] == spec["kernel_id"], f"metadata id {meta['id']} != {spec['kernel_id']}"
    assert meta.get("enable_gpu") is True, "GPU must be enabled"
    assert meta.get("enable_internet") is True, "internet must be enabled"
    assert meta["code_file"] == "main.py"
    assert "playground-series-s6e10" in json.dumps(meta.get("dataset_sources", "")) or \
        "playground-series-s6e10" in json.dumps(meta.get("competition_sources", "")), \
        "competition source missing"

    plan = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": args.variant,
        "kernel_id": spec["kernel_id"],
        "kernel_dir": spec["dir"],
        "main_sha256": file_sha256(main_py),
        "meta_sha256": file_sha256(meta_py),
        "prereg_sha256": spec["prereg"],
        "constants": {k: v[0] for k, v in const.items()},
        "timeout_s": args.timeout_s,
        "pushed": bool(args.push),
    }
    out = ROOT / "reports/phase23" / f"push_plan_{args.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=1))

    if not args.push:
        print("dry run: nothing pushed (add --push)")
        return 0

    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir),
           "--timeout", str(args.timeout_s)]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1200:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1200:])
        return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
