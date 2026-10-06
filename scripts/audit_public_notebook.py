"""Audit the public notebook "S6E10 | LB 0.96134 | What Each Step Was Worth".

What this does and does not do
------------------------------
DOES: retrieve the notebook's SOURCE and metadata, and classify every claimed "step worth X" into a
category that says how much evidential weight the claim can carry.

DOES NOT: download or import its submission CSV or any prediction file, and does not treat its public
score as evidence of anything. A public LB number is a measurement of one 59,969-row split against a
metric with a paired noise floor of about +/-2e-4 between near-identical submissions, so a claimed
"+3e-4 from this step" is a claim about a quantity smaller than or comparable to the board's own
resolution unless the author shows local validation.

Classification scheme
----------------------
  A  honest local validation      -- cross-validated on a scheme that does not consult the eval fold
  B  public-LB-only observation   -- attributed from the leaderboard, no local CV behind it
  C  imported prediction          -- contributes a prediction file rather than a trained model
  D  leakage / validation ambiguity -- the validation cannot be trusted as written
  E  mechanism already tested here -- we have measured it and know the answer
  F  genuinely new mechanism      -- untested here and worth reproducing under our folds

Usage: python scripts/audit_public_notebook.py [--notebooks id1,id2]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, ROOT, save_json  # noqa: E402

TARGET_TITLE = "LB 0.96134"

# What we have already measured, so a claimed step can be classified E rather than re-tested.
ALREADY_TESTED = {
    "extra_trees / extremely randomised trees": "E -- +44.8e-5 under GBDT (Phase 9 ctl_det), and the "
        "single largest lever measured in this campaign",
    "dart": "E -- REJECTED, -42.8e-5 and -48.7e-5 (Phase 9)",
    "lightgbm rf / random forest mode": "E -- REJECTED, -306.8e-5 (Phase 9)",
    "catboost ordered": "E -- REJECTED, -71.7e-5 (Phase 10B C1)",
    "catboost native categorical": "E for the marginal-ensemble question -- +6.64e-5 at MODEL level "
        "(C2-C0, 4/5 folds) but +0.00e-5 added to the v3 ensemble (Phase 10B)",
    "target encoding": "E -- existing TE beats all-21 TE by 13.3e-5; all-21 and conditional variants "
        "both REJECTED (Phase 8)",
    "soft targets / label smoothing": "E -- REJECTED",
    "pairwise / ranking objective": "E -- pure pairwise LightGBM REJECTED",
    "augmentation": "E -- REJECTED, all families",
    "original-row append": "E -- REJECTED, -4.2e-4 at lambda=0.2, monotone",
    "external teacher": "E -- phase 1-2 result, -4.2e-4 at lambda=0.2",
    "group calibration": "E -- REJECTED, +0.8e-5 with the control moving 0.0e-5",
    "pseudo-labelling": "E -- REJECTED, -152.5e-5 at frac 0.25",
    "bagging fraction sweep": "E -- REJECTED, signs flip",
    "more members / seed proliferation": "E -- saturated; near-clones buy 1e-6..5e-6 against a "
        "+1.5e-5 gate",
    "full-data refit": "E -- v4 submitted, public +2e-5, unresolved at +/-2e-4 board precision",
    "k-fold averaging": "E -- 5->10 folds +1.0e-4 measured; already in v3 via block10 members",
    "realmlp ensemble size": "E -- n_ens 8->32 +2.0e-4, already applied",
}


def classify(step: str) -> tuple[str, str]:
    s = step.lower()
    for k, v in ALREADY_TESTED.items():
        if k in s:
            return "E", v
    if any(w in s for w in ("import", "read_csv", "kaggle input", "submission", "pseudo file",
                            "external csv", "blend of submissions")):
        return "C", "contributes a prediction file rather than a trained model; its 'value' is the "\
                    "author's own blend, not a mechanism"
    if any(w in s for w in ("public", "lb", "leaderboard", "score went", "0.961")):
        return "B", "attributed from the leaderboard; the paired noise floor between near-identical "\
                    "submissions is about +/-2e-4, so sub-1e-4 attributions are not resolvable"
    if any(w in s for w in ("cv", "cross-val", "validation", "oof", "fold")):
        return "A", "claims local validation; the scheme must be read before the claim is accepted"
    if any(w in s for w in ("leak", "test set", "fit on all", "full data", "transduct")):
        return "D", "validation scheme is ambiguous or leaks the evaluation fold"
    return "F", "not recognised as something already measured here"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="playground-series-s6e10")
    args = ap.parse_args()
    t0 = time.time()

    print("PUBLIC NOTEBOOK AUDIT")
    print("=" * 96)
    print(f"  looking for: {TARGET_TITLE!r} in competition {args.competition}")
    print("  policy: retrieve SOURCE and metadata only. No prediction CSV or submission file is")
    print("  downloaded or imported, and no public score is accepted as evidence.\n")

    raw = ROOT / "research" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out: dict = {"target": TARGET_TITLE, "competition": args.competition,
                 "policy": "source and metadata only; no prediction or submission file imported",
                 "attempts": []}

    # ---- try the Kaggle CLI for notebook metadata ----
    # Invoke as `python -m kaggle`, NOT the bare `kaggle` executable. The bare one resolves to a
    # different interpreter's script (Python314 AppData) whose output is decoded with the cp1252
    # default and raises UnicodeDecodeError on any non-ASCII byte. Using the venv module keeps the
    # decoding explicit and the version known (2.2.4).
    nb_ids: list[str] = []
    for cmd in (["kernels", "list", "-s", args.competition, "--sort-by", "voteCount"],
                ["kernels", "list", "--competition", args.competition, "--sort-by", "voteCount"],
                ["kernels", "list", "-s", args.competition, "--page-size", "100"]):
        argv = [sys.executable, "-m", "kaggle"] + cmd
        try:
            r = subprocess.run(argv, capture_output=True, timeout=240, encoding="utf-8",
                               errors="replace")
            out["attempts"].append({"cmd": " ".join(argv[2:]), "rc": r.returncode,
                                    "stdout_head": (r.stdout or "")[:2000],
                                    "stderr_head": (r.stderr or "")[:400]})
            if r.returncode == 0 and (r.stdout or "").strip():
                print(f"  '{' '.join(cmd)}' -> rc=0, {len(r.stdout.splitlines())} lines")
                for line in r.stdout.splitlines():
                    if "0.96134" in line or "What Each Step" in line:
                        m = re.match(r"\s*([0-9a-f]{8,})\s", line)
                        if m:
                            nb_ids.append(m.group(1))
                        print(f"    MATCH: {line.strip()[:160]}")
                if not nb_ids:
                    print("    (listing succeeded but the target title was not in it; recording")
                    print("     what IS listed so the search is reproducible and the scope is clear)")
                    listing = [{"ref": ln.split()[0], "title": ln.split()[1] if len(ln.split()) > 1
                                else ""} for ln in r.stdout.splitlines()
                               if ln.strip() and not ln.strip().startswith(("ref", "---"))]
                    for e in listing[:12]:
                        print(f"      {e['ref']:<58} {e['title'][:70]}")
                    out["listing"] = listing
                break
            else:
                print(f"  '{' '.join(cmd)}' -> rc={r.returncode} "
                      f"{(r.stderr or '').strip()[:120]}")
        except Exception as exc:                                 # noqa: BLE001
            out["attempts"].append({"cmd": " ".join(argv[2:]),
                                    "error": f"{type(exc).__name__}: {exc}"})
            print(f"  '{' '.join(cmd)}' -> {type(exc).__name__}: {str(exc)[:100]}")

    # ---- the competition listing is NOT exhaustive: search the GLOBAL index BY TITLE ----
    # The first attempt only listed kernels attached to the competition, which does not contain the
    # target title. `kernels list --search` queries the global kernel index, where a notebook whose
    # slug is unrelated to the competition still appears. Tried widest-to-narrowest so a partial
    # match is not missed, and each attempt is recorded even on failure so the search is auditable.
    if not nb_ids:
        for term in ("What Each Step Was Worth", "0.96134", "S6E10", "Playground Series S6E10"):
            argv = [sys.executable, "-m", "kaggle", "kernels", "list", "--search", term,
                    "--sort-by", "voteCount"]
            try:
                r = subprocess.run(argv, capture_output=True, timeout=240, encoding="utf-8",
                                   errors="replace")
                out["attempts"].append({"cmd": " ".join(argv[2:]), "rc": r.returncode,
                                        "stdout_head": (r.stdout or "")[:2500],
                                        "stderr_head": (r.stderr or "")[:300]})
                if r.returncode != 0:
                    print(f"  search {term!r} -> rc={r.returncode} "
                          f"{(r.stderr or '').strip()[:110]}")
                    continue
                lines = (r.stdout or "").splitlines()
                print(f"  search {term!r} -> rc=0, {len(lines)} lines")
                for line in lines:
                    low = line.lower()
                    if "0.96134" in low or "what each step" in low:
                        m = re.match(r"\s*([0-9a-zA-Z_\-]+/[0-9a-zA-Z_\-]+)\s", line)
                        if m:
                            nb_ids.append(m.group(1))
                        print(f"    EXACT MATCH: {line.strip()[:170]}")
                if not nb_ids and term == "S6E10":
                    print("    (no exact title match; S6E10-related kernels seen:)")
                    for line in lines[:14]:
                        print(f"      {line.strip()[:150]}")
                if nb_ids:
                    break
            except Exception as exc:                             # noqa: BLE001
                out["attempts"].append({"cmd": " ".join(argv[2:]),
                                        "error": f"{type(exc).__name__}: {exc}"})
                print(f"  search {term!r} -> {type(exc).__name__}")

    print(f"\n  notebook refs recovered: {nb_ids if nb_ids else 'none'}")
    out["notebook_ids"] = nb_ids

    # ---- reuse any previously cached notebook research ----
    for p in sorted(raw.glob("*notebook*")) + sorted(raw.glob("*kernel*")):
        print(f"  cached artifact: {p.name} ({p.stat().st_size} bytes)")
        out["attempts"].append({"cached": p.name, "bytes": p.stat().st_size})

    if not nb_ids:
        # Nothing matched, so the target notebook is NOT RETRIEVED and no claim is made about it.
        # Auditing a notebook we could not read would mean inventing its contents, which is worse
        # than recording the gap. The only defensible output is: not retrieved, no claims.
        print("\n  RESULT: the target notebook was NOT located in this environment.")
        print("  The kernel listing succeeded but did not contain the title, so NO source was")
        print("  retrieved and NO claim is made about its contents, models, validation scheme or")
        print("  claimed gains. Recorded as OUTSTANDING -- an audit that invents its subject is")
        print("  worse than no audit.")
        out["verdict"] = ("NOT RETRIEVED -- outstanding. No claim is made about the notebook's "
                          "contents, models, validation scheme or claimed gains.")
        out["steps_classified"] = []
        out["known_mechanisms_already_measured"] = ALREADY_TESTED
        out["what_a_reproduction_would_require"] = (
            "If the source is obtained later, every claimed 'step worth X' must be classified A-F "
            "with ALREADY_TESTED as the reference for category E. Categories B (public-LB-only) and "
            "C (imported predictions) cannot license any local change: the paired public-LB noise "
            "floor between near-identical submissions is about +/-2e-4, so a sub-1e-4 attribution "
            "is below the board's resolution. Only A, D and F are worth reproducing.")
    else:
        steps = []
        nb_source, nb_cells_all = "", []
        for nid in nb_ids:
            dest = raw / f"kernel_{nid}.ipynb"
            r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "pull", nid,
                                "-p", str(raw), "-m"], capture_output=True, timeout=300,
                               encoding="utf-8", errors="replace")
            out["attempts"].append({"cmd": f"kernels pull {nid}", "rc": r.returncode,
                                    "stderr_head": (r.stderr or "")[:300]})
            if not dest.exists():
                # `kaggle kernels pull` names the file after the notebook SLUG, not the id, so the
                # expected path is often wrong even on a successful pull. Locate whatever landed in
                # research/raw and use that; a wrong path would otherwise leave `src` empty and make
                # every targeted probe report ABSENT -- which is exactly the false negative I hit.
                landed = sorted(raw.glob("*.ipynb"), key=lambda q: q.stat().st_mtime, reverse=True)
                dest = landed[0] if landed else dest
            print(f"  pull {nid}: rc={r.returncode} "
                  f"{'-> ' + dest.name if dest.exists() else (r.stderr or '').strip()[:120]}")
            if not dest.exists():
                print(f"    nothing landed in {raw} after the pull")
                continue
            nb = json.loads(dest.read_text(encoding="utf-8"))
            cells = nb.get("cells", [])
            src = "\n".join("".join(c.get("source", [])) for c in cells)
            nb_source, nb_cells_all = src, cells
            out["attempts"].append({"notebook": nid, "n_cells": len(cells),
                                    "source_chars": len(src),
                                    "metadata": {k: v for k, v in nb.get("metadata", {}).items()
                                                 if k in ("kernelspec", "language_info")}})
            print(f"    {len(cells)} cells, {len(src)} chars of source")
            reads_preds = bool(re.search(r"read_csv|submission|\.csv['\"]", src))
            print(f"    reads a CSV / submission file: {reads_preds}")
            out["reads_prediction_csv"] = reads_preds
            # markdown headings are where an attribution-style notebook states its claims
            claims = []
            for c in cells:
                if c.get("cell_type") == "markdown":
                    for line in "".join(c.get("source", [])).splitlines():
                        ls = line.strip()
                        if re.search(r"(step|worth|\+?\d+\.\d{3,}|LB)", ls) and len(ls) < 220:
                            claims.append(ls)
            print(f"    candidate claim lines: {len(claims)}")
            for cl in claims[:40]:
                cat, why = classify(cl)
                steps.append({"notebook": nid, "claim": cl, "category": cat, "why": why})
                print(f"      [{cat}] {cl[:120]}")
            out["steps_classified"] = steps
        counts: dict[str, int] = {}
        for s in steps:
            counts[s["category"]] = counts.get(s["category"], 0) + 1
        out["category_counts"] = counts

        # ---- a targeted read of the things that actually matter for our decision ----
        # The generic claim-line scan above is a coarse first pass. These are the specific questions
        # Phase 11R needs answered, and each is answered by QUOTING the notebook rather than by
        # pattern-matching a line, because a single word decides whether a step is reproducible here.
        probes = {
            "native_categorical_catboost":
                ("cat_features", "boosting_type", "CatBoostClassifier", "Ordered"),
            "imports_prediction_files": ("read_csv", "CC0", "public OOF", "kaggle datasets"),
            "ten_fold_members": ("10-fold", "10 fold", "n_splits=10", "StratifiedKFold(n_splits=10"),
            "target_encoding_flight_distance": ("Flight Distance", "_fd_bin", "te_fd"),
            "original_data_teacher": ("original", "129,880", "teacher"),
            "stacker": ("LogisticRegression", "combiner CV", "greedy"),
        }
        findings = {}
        # `src` and `cells` are defined inside the per-notebook loop, so accumulate them into outer
        # variables. Reading them here directly raises NameError because they are loop-local.
        nb_src, nb_cells = nb_source, nb_cells_all
        for key, needles in probes.items():
            hits = {n: nb_src.count(n) for n in needles}
            findings[key] = {"counts": hits,
                             "present": {n: c for n, c in hits.items() if c > 0}}
        out["targeted_findings"] = findings
        print("\n  TARGETED FINDINGS from the retrieved source")
        for key, f in findings.items():
            print(f"    {key:<34} {f['present'] if f['present'] else 'ABSENT'}")

        # CatBoost is the decisive one for us: if the notebook claims a native-categorical result we
        # must know whether it actually passed cat_features, because that is the mechanism Phase 10B
        # measured and Phase 11R is extending.
        nb_cat = ("CatBoost" in nb_src)
        nb_native = findings["native_categorical_catboost"]["present"]
        print(f"\n    mentions CatBoost: {nb_cat}   passes native categoricals: "
              f"{'YES' if nb_native else 'NO -- no cat_features / no CatBoostClassifier in the source'}")
        out["catboost_native_categorical_used"] = bool(nb_native)

        # The notebook's own "what did not help" table is the highest-value content for us, because
        # it reports a NEGATIVE for exactly the mechanism we tested.
        neg = []
        for c in nb_cells_all:
            if c.get("cell_type") != "markdown":
                continue
            t = "".join(c.get("source", []))
            if "did not help" in t.lower() or "numerics as native categoricals" in t.lower():
                for ln in t.splitlines():
                    if "native categorical" in ln.lower():
                        neg.append(ln.strip())
        out["notebook_negative_on_native_cat"] = neg
        for ln in neg:
            print(f"    NOTEBOOK SAYS: {ln[:200]}")
        if neg:
            print("    -> the notebook reports a NEGATIVE for native categoricals INSTEAD OF target")
            print("       encoding. That is the OPPOSITE substitution from ours: we measured +6.64e-5")
            print("       for native categoricals ON TOP of our existing TE block, and the notebook")
            print("       measured numerics-as-categories INSTEAD of TE. Different interventions;")
            print("       neither refutes the other.")

        out["verdict"] = ("RETRIEVED AND CLASSIFIED. See targeted_findings for the mechanism-level "
                          "answers. Category A items are reproducible in principle; B and C cannot "
                          "license a local change; the notebook's own negative table is recorded "
                          "because it concerns a mechanism we tested, in a different substitution.")

    out["seconds"] = round(time.time() - t0, 1)
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()[:12]
    save_json(out, REPORTS / "public_notebook_audit.json")
    print(f"\nVERDICT: {out['verdict']}")
    print("wrote", REPORTS / "public_notebook_audit.json")


if __name__ == "__main__":
    main()
