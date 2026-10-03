"""Audit which real source files are missing from git because of an over-broad .gitignore rule.

This exists because a bare ``features/`` pattern (intended for ``artifacts/features/``) silently
excluded ``src/features/`` -- i.e. the entire feature-engineering module was never committed, so
the GitHub repository could not reproduce the solution. Rules 2.8.b requires the winning model's
code to be delivered, so a silently missing source directory is a competition-level defect.

Reports every file present on disk but absent from the index, classified as
  SOURCE-MISSING  -> under a source directory; must be committed
  ARTIFACT-OK     -> a cache / prediction / model artefact; expected to be ignored
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "configs", "tests")
ARTIFACT_HINTS = ("artifacts", "data/raw", "data/original", ".venv", "__pycache__",
                  ".pytest_cache", ".git")

# Directory patterns that are INTENTIONALLY matched at any depth: build/env caches, which can
# legitimately appear under a package directory and never correspond to source we need.
INTENTIONAL_ANY_DEPTH = {".venv/", "venv/", "__pycache__/", ".pytest_cache/",
                         ".ipynb_checkpoints/"}


def git_ls_files() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return set(out.stdout.split())


def main() -> int:
    tracked = git_ls_files()
    print("=" * 100)
    print("UNTRACKED-BUT-PRESENT AUDIT")
    print("=" * 100)

    source_missing, artifact_ok = [], []
    for sd in SOURCE_DIRS:
        d = ROOT / sd
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            if "__pycache__" in rel or rel in tracked:
                continue
            source_missing.append(rel)
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel in tracked or rel in source_missing:
            continue
        if any(h in rel for h in ARTIFACT_HINTS):
            artifact_ok.append(rel)

    print(f"\nSOURCE files present but NOT tracked ({len(source_missing)}):")
    if source_missing:
        for r in source_missing:
            print(f"  MISSING  {r}  ({(ROOT / r).stat().st_size} bytes)")
    else:
        print("  none")

    print(f"\nArtefact files ignored (expected), showing first 12 of {len(artifact_ok)}:")
    for r in artifact_ok[:12]:
        print(f"  ok       {r}")

    print("\n.gitignore rules that could over-match (unanchored directory names):")
    gi = (ROOT / ".gitignore")
    risky = 0
    for i, line in enumerate(gi.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        # over-broad if it does NOT start with '/', has no internal '/', and contains no glob.
        # A glob like '*.egg-info/' can only ever match a path ending in that literal, so it
        # cannot swallow a source directory.
        if s.endswith("/") and not s.startswith("/") and s.count("/") == 1 and "*" not in s:
            if s in INTENTIONAL_ANY_DEPTH:
                print(f"  line {i}: '{s}'  ok (intentional env/cache rule, any depth)")
            else:
                risky += 1
                print(f"  line {i}: '{s}'  <-- RISKY UNANCHORED: matches at any depth, "
                      f"e.g. src/{s}")
    print(f"  risky patterns: {risky}")

    return 1 if (source_missing or risky) else 0


if __name__ == "__main__":
    sys.exit(main())