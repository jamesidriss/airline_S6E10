"""Secret scanner for the repository tree and reachable git history.

Design rule: this tool NEVER prints a candidate secret value. It reports only
    (detector, severity, path, line-or-commit, redacted fingerprint length)
so that a human can go and rotate the credential without the value ever entering
a terminal scrollback, a CI log, or this repository.

Detectors
---------
google_api_key   AIza[0-9A-Za-z_-]{35}
google_oauth     ya29.[0-9A-Za-z_-]+
aws_access_key   AKIA|ASIA[0-9A-Z]{16}
aws_secret_key   aws_secret_access_key assignment
github_token     gh[pousr]_[A-Za-z0-9]{36,}   |  github_pat_[A-Za-z0-9_]{50,}
kaggle_json      a "key":"<40 hex>" inside a file named kaggle.json
bearer_header    Authorization: Bearer <token>   |  "authorization" with a token value
private_key      -----BEGIN [A-Z ]*PRIVATE KEY-----
dotenv           a .env file with non-comment assignments
cookie_header    Cookie: / Set-Cookie: with a long value
generic_secret   password|secret|token|api_key assignment with a long literal

Usage:
  python scripts/secret_scan.py                 # scan working tree
  python scripts/secret_scan.py --history       # scan every reachable commit
  python scripts/secret_scan.py --json out.json
Exit code 1 if any HIGH severity finding exists.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (name, severity, compiled pattern)
DETECTORS: list[tuple[str, str, re.Pattern]] = [
    ("google_api_key", "HIGH", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("google_oauth_token", "HIGH", re.compile(r"ya29\.[0-9A-Za-z_-]{20,}")),
    ("aws_access_key_id", "HIGH", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret_access_key", "HIGH",
     re.compile(r"aws_secret_access_key\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{40}")),
    ("github_token", "HIGH",
     re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("private_key_block", "HIGH",
     re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("bearer_token", "HIGH",
     re.compile(r"(?:Authorization\s*:\s*Bearer|[\"']authorization[\"']\s*[:=]\s*[\"']Bearer)"
                r"\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("kaggle_api_token", "HIGH", re.compile(r"[\"']?api_token[\"']?\s*[:=]\s*[\"'][A-Za-z0-9]{32,}[\"']")),
    ("cookie_header", "MEDIUM",
     re.compile(r"(?:Cookie|Set-Cookie)\s*:\s*[^\n]{20,}")),
    ("generic_secret_assignment", "MEDIUM",
     re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
                r"\b\s*[=:]\s*[\"']?[A-Za-z0-9/+_.-]{16,}[\"']?")),
    ("kaggle_json_key", "HIGH", re.compile(r"\"key\"\s*:\s*\"[0-9a-fA-F]{32,40}\"")),
]

TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".ini", ".cfg",
                 ".toml", ".ps1", ".sh", ".env", ".gitignore", ".lock", ""}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}

# The scanner's own detector regexes necessarily contain the literal patterns they look for, so
# scanning itself produces guaranteed self-matches. Excluded by path, with the reason recorded.
SELF_EXEMPT = {
    "scripts/secret_scan.py",  # contains the detector patterns themselves
}


class Finding(dict):
    pass


def fingerprint(value: str) -> str:
    """Length + a non-reversible tag, so two findings can be compared without exposing values."""
    import hashlib

    return f"len={len(value)} sha256:{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def scan_text(text: str, path: str, where: str) -> list[Finding]:
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        if len(line) > 20000:  # skip minified blobs
            line = line[:20000]
        for name, sev, pat in DETECTORS:
            for m in pat.finditer(line):
                out.append(Finding(detector=name, severity=sev, path=path,
                                   where=f"{where}:{i}" if where else str(i),
                                   match_len=len(m.group(0)),
                                   fingerprint=fingerprint(m.group(0))))
    return out


def scan_tree() -> list[Finding]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel in SELF_EXEMPT:
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name not in (".env", ".gitignore"):
            continue
        if p.stat().st_size > 8_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        out.extend(scan_text(text, rel, ""))
        if p.name == "kaggle.json" or p.name == ".env":
            out.append(Finding(detector="sensitive_filename", severity="HIGH", path=rel,
                               where="file", match_len=0, fingerprint="n/a"))
    return out


def scan_history() -> list[Finding]:
    """Scan every blob reachable from every ref. Reports commit + path only."""
    out: list[Finding] = []
    try:
        revs = subprocess.run(["git", "rev-list", "--all"], cwd=ROOT, capture_output=True,
                             text=True, timeout=300).stdout.split()
    except Exception as exc:  # noqa: BLE001
        return [Finding(detector="history_scan_error", severity="LOW", path="-", where="-",
                        match_len=0, fingerprint=str(exc)[:40])]
    seen_blob: dict[str, str] = {}
    for rev in revs:
        try:
            files = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev], cwd=ROOT,
                                   capture_output=True, text=True, timeout=120).stdout.split()
        except Exception:  # noqa: BLE001
            continue
        for f in files:
            if f.split("/")[0] in SKIP_DIRS:
                continue
            if Path(f).suffix.lower() not in TEXT_SUFFIXES and Path(f).name not in (".env",):
                continue
            h = subprocess.run(["git", "rev-parse", f"{rev}:{f}"], cwd=ROOT, capture_output=True,
                               text=True, timeout=60).stdout.strip()
            if not h:
                continue
            if h in seen_blob:
                continue
            seen_blob[h] = f
            try:
                blob = subprocess.run(["git", "cat-file", "blob", h], cwd=ROOT,
                                      capture_output=True, timeout=60).stdout.decode(
                    "utf-8", "ignore")
            except Exception:  # noqa: BLE001
                continue
            out.extend(scan_text(blob, f, rev[:12]))
    # sensitive filenames anywhere in history
    for rev in revs:
        files = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev], cwd=ROOT,
                               capture_output=True, text=True, timeout=120).stdout.split()
        for f in set(files):
            if Path(f).name in ("kaggle.json", ".env", "id_rsa", ".netrc"):
                out.append(Finding(detector="sensitive_filename", severity="HIGH", path=f,
                                   where=rev[:12], match_len=0, fingerprint="n/a"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    findings = scan_history() if args.history else scan_tree()
    high = [f for f in findings if f["severity"] == "HIGH"]
    med = [f for f in findings if f["severity"] == "MEDIUM"]

    print("=" * 96)
    print(f"SECRET SCAN: {'GIT HISTORY (all reachable commits)' if args.history else 'WORKING TREE'}")
    print("=" * 96)
    print(f"  findings: {len(findings)}  HIGH: {len(high)}  MEDIUM: {len(med)}")
    print("  (values are never printed; a redacted fingerprint identifies each)")
    print()
    for f in sorted(findings, key=lambda d: (d["severity"] != "HIGH", d["path"])):
        print(f"  [{f['severity']:<6}] {f['detector']:<26} {f['path']}  {f['where']}  "
              f"{f['fingerprint']}")
    if args.json:
        Path(args.json).write_text(json.dumps(findings, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 1 if high else 0


if __name__ == "__main__":
    sys.exit(main())