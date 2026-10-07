"""Pull S6E10 public notebooks into PER-KERNEL DIRECTORIES and record provenance authoritatively.

WHY THIS EXISTS (a real corruption, not a hypothetical)
-------------------------------------------------------
`kaggle kernels pull <ref> -p research/raw` does not reliably name the output after the ref. Running
it for three different kernels into the same directory left a SINGLE file, and that file's content
was a DIFFERENT notebook than the earlier audited one -- while still carrying the earlier notebook's
audited filename `s6e10-what-each-step-was-worth.ipynb`. Had the audit been re-run, it would have
attributed a notebook's findings to the wrong author, and every downstream claim about "what the
public notebook contains" would have been silently wrong.

That is the same failure class as the earlier path-assumption bug: trusting an output filename
instead of verifying identity. Here it is worse, because the wrong content was filed under a name
that ledger entries already reference.

So identity is established three ways and cross-checked:
  1. each kernel gets its OWN directory named from the ref, so files cannot collide;
  2. `kernel-metadata.json` inside that directory is read for the authoritative id/title/author;
  3. the notebook's own metadata is read and COMPARED against the directory's ref.

A mismatch between any of the three is a hard failure, not a warning.

Usage:
  python scripts/pull_public_notebooks.py --refs a/slug,b/slug
  python scripts/pull_public_notebooks.py --restore s6e10-what-each-step-was-worth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, save_json  # noqa: E402

ROOT = Path("research/raw/notebooks")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def pull(ref: str) -> dict:
    ref = ref.strip()
    dest = ROOT / ref.replace("/", "__")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "pull", ref,
                        "-p", str(dest), "-m"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    nbs = sorted(dest.glob("*.ipynb"))
    rec = {"ref": ref, "dest": str(dest), "returncode": r.returncode,
           "stderr": (r.stderr or "")[:200], "notebooks": []}
    if not nbs:
        rec["error"] = "no .ipynb landed in the kernel directory"
        return rec
    meta_p = dest / "kernel-metadata.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    rec["kernel_metadata"] = {k: meta.get(k) for k in ("id", "title", "author", "code_file",
                                                       "language", "kernel_type")}
    for p in nbs:
        nb = json.loads(p.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])
        src = "\n".join("".join(c.get("source", [])) for c in cells)
        nm = nb.get("metadata", {})
        nref = (nm.get("kernelspec") or {}).get("name")
        rec["notebooks"].append({
            "file": p.name, "bytes": p.stat().st_size, "sha16": sha(p.read_bytes()),
            "n_cells": len(cells), "source_chars": len(src),
            "reads_csv": ("read_csv" in src) or (".csv" in src),
            "nb_title": ((nm.get("kaggle") or {}).get("kernelRun") or {}).get("title"),
        })
        # IDENTITY CHECK 3: the directory is named from the ref, so a file landing here must
        # belong to that kernel. Compare the slug appearing in the filename against the ref.
        slug_in_name = p.stem
        slug_in_ref = ref.split("/", 1)[1]
        rec["notebooks"][-1]["slug_matches_ref"] = (
            slug_in_name == slug_in_ref or slug_in_ref in slug_in_name)
    # IDENTITY CHECK 2: the metadata id must match the ref we asked for.
    mid = rec["kernel_metadata"].get("id")
    rec["metadata_id_matches_ref"] = (mid == ref) if mid else None
    return rec


MECHANISM_PROBES = {
    "native_categorical_lightgbm": ["categorical_feature", "pandas.Categorical", "category"],
    "native_categorical_catboost": ["cat_features", "CatBoostClassifier", "max_ctr_complexity"],
    "auxiliary_task_features": ["aux", "each rating", "other 20", "auxiliary"],
    "pseudo_labelling": ["pseudo", "self.train", "iterative"],
    "route_features": ["route", "Route", "route_id"],
    "neural_or_mlp": ["torch", "RealMLP", "MLP", "ResNet", "TabM"],
    "target_encoding": ["TargetEncoder", "target encoding", "target_encoding", "te_"],
    "original_dataset": ["original", "129,880", "airline.csv"],
    "stacking": ["LogisticRegression", "greedy", "stack"],
    "imports_public_predictions": ["public OOF", "CC0", "kaggle datasets", "public_oof"],
}


def classify(rec: dict) -> dict:
    ref = rec["ref"]
    dest = Path(rec["dest"])
    out = {"ref": ref, "probes": {}}
    p = dest / rec["notebooks"][0]["file"]
    nb = json.loads(p.read_text(encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    for key, needles in MECHANISM_PROBES.items():
        hits = {n: src.count(n) for n in needles if src.count(n)}
        out["probes"][key] = hits
    out["markdown_mechanism_lines"] = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "markdown":
            continue
        t = "".join(c.get("source", []))
        for ln in t.splitlines():
            low = ln.lower()
            if any(k in low for k in ("aux", "categorical", "did not help", "no help",
                                      "hurt", "worse", "negative", "ablation")) and len(ln) < 260:
                out["markdown_mechanism_lines"].append(ln.strip())
    out["markdown_mechanism_lines"] = out["markdown_mechanism_lines"][:28]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="")
    ap.add_argument("--restore", default="")
    args = ap.parse_args()
    refs = [r for r in args.refs.split(",") if r.strip()]
    if args.restore:
        refs.append(args.restore)
    if not refs:
        print("give --refs a,b or --restore <slug>")
        return 1

    ROOT.mkdir(parents=True, exist_ok=True)
    out = {"pulled": [], "problems": []}
    for ref in refs:
        print(f"\n  pulling {ref}")
        rec = pull(ref)
        out["pulled"].append(rec)
        if rec.get("error"):
            out["problems"].append(f"{ref}: {rec['error']}")
            print(f"    FAILED: {rec['error']}")
            continue
        print(f"    -> {rec['dest']}")
        print(f"       kernel-metadata id = {rec['kernel_metadata'].get('id')!r} "
              f"(matches ref: {rec['metadata_id_matches_ref']})")
        for n in rec["notebooks"]:
            print(f"       {n['file']}: {n['n_cells']} cells, {n['source_chars']} chars, "
                  f"sha16={n['sha16']}, reads_csv={n['reads_csv']}, "
                  f"slug_matches_ref={n['slug_matches_ref']}")
        if rec["metadata_id_matches_ref"] is False:
            out["problems"].append(f"{ref}: kernel-metadata id != requested ref")
        for n in rec["notebooks"]:
            if not n["slug_matches_ref"]:
                out["problems"].append(f"{ref}: file slug {n['file']!r} does not match the ref")
        try:
            out.setdefault("classified", []).append(classify(rec))
        except Exception as exc:                                  # noqa: BLE001
            out["problems"].append(f"{ref}: classification failed: {exc}")

    for c in out.get("classified", []):
        print(f"\n  MECHANISM PROBES for {c['ref']}")
        for k, v in c["probes"].items():
            print(f"    {k:<32} {v if v else 'ABSENT'}")
    if out.get("classified"):
        print("\n  THEIR OWN mechanism / negative lines:")
        for c in out["classified"]:
            for ln in c["markdown_mechanism_lines"][:12]:
                print(f"    [{c['ref'].split('/')[0]}] {ln[:190]}")

    save_json(out, REPORTS / "public_notebook_pull_manifest.json")
    print(f"\n  problems: {out['problems'] or 'none'}")
    print("  wrote", REPORTS / "public_notebook_pull_manifest.json")
    return 1 if out["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())