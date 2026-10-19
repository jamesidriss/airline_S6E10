"""Phase-20 resumable job scheduler (mandate §20) + runtime ledger (§8).

Durable state file: research/phase20_exhaustive/scheduler.json
Append-only runtime log: research/phase20_exhaustive/runtime_ledger.jsonl

States: PENDING, RUNNING, DONE, FAILED, INVALID, RESOURCE_BLOCKED,
        LICENSE_BLOCKED, DEPENDENCY_BLOCKED, PROMOTED, CLOSED_NEGATIVE.

Rules enforced here:
  * never rerun an identical fingerprint (cmd + prereg sha + runner bytes sha);
    a changed runner/prereg legitimately produces a new fingerprint.
  * never two heavy GPU jobs: a pid-checked lock file; additionally refuses a
    GPU job when nvidia-smi utilization >= 80% (some other process is busy).
  * a job with no prereg sha refuses to run unless explicitly --no-prereg
    (read-only diagnostics only).
  * DONE requires manifest verification when the job declares `expect`
    (report exists, status DONE, metrics non-null, prediction sha matches,
    row count matches) -- a FAILED run can never be recorded DONE.

Usage:
  scheduler.py init                 # seed from the catalogue (idempotent)
  scheduler.py list [--state S] [--kind K]
  scheduler.py show J001
  scheduler.py set-cmd J001 -- cmd...   # register/replace the invocation
  scheduler.py prereg J001 path.json    # record prereg + sha (recomputes fingerprint)
  scheduler.py block J001 LICENSE_BLOCKED --why "..."
  scheduler.py next [--kind cpu|gpu]    # highest-rank runnable job
  scheduler.py run J001 [--no-prereg] [--timeout N] [--why "..."]
  scheduler.py resume                  # RUNNING records whose pid died -> FAILED
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256  # noqa: E402

SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"
RUNTIME = ROOT / "research" / "phase20_exhaustive" / "runtime_ledger.jsonl"
CATALOGUE = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
GPU_LOCK = ROOT / "artifacts" / "phase20" / ".gpu.lock"

STATES = ("PENDING", "RUNNING", "DONE", "FAILED", "INVALID", "RESOURCE_BLOCKED",
          "LICENSE_BLOCKED", "DEPENDENCY_BLOCKED", "PROMOTED", "CLOSED_NEGATIVE")
TERMINAL = ("DONE", "INVALID", "PROMOTED", "CLOSED_NEGATIVE", "RESOURCE_BLOCKED",
            "LICENSE_BLOCKED", "DEPENDENCY_BLOCKED")

# Catalogue status -> scheduler state. Verdict nuance lives in result.verdict so
# no historical row is overstated or understated by the state alone.
STATUS_MAP = {
    "COMPLETE_NEGATIVE": "CLOSED_NEGATIVE",
    "COMPLETE_CERTIFIED": "PROMOTED",
    "HISTORICAL_EQUIVALENT": "DONE",
    "PARTIAL_VALID": "DONE",
    "LICENSE_BLOCKED": "LICENSE_BLOCKED",
    "NOT_STARTED": "PENDING",
}
# blocking_reason -> DEPENDENCY_BLOCKED (the trigger condition cannot fire).
DEP_REASONS = {
    "blocked-on-base", "blocked-on-arch", "blocked-on-acc", "no-candidate",
    "no-strong-candidate", "not-warranted", "blocked-identifiability",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load() -> dict:
    if not SCHED.exists():
        raise SystemExit("scheduler.json missing -- run `scheduler.py init` first")
    return json.loads(SCHED.read_text(encoding="utf-8"))


def save(doc: dict) -> None:
    doc["updated_utc"] = utcnow()
    SCHED.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def runtime_append(entry: dict) -> None:
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fingerprint(job: dict) -> str | None:
    """sha256(cmd + prereg sha + runner bytes). None until both cmd+prereg exist."""
    if not job.get("cmd") or not (job.get("prereg") or {}).get("sha256"):
        return None
    # The executable can be python.exe; fingerprint the actual .py runner.
    source = next((Path(arg) for arg in job["cmd"] if str(arg).lower().endswith(".py")), None)
    runner = (source if source.is_absolute() else ROOT / source) if source else None
    runner_sha = sha256_file(runner) if runner and runner.is_file() else "MISSING"
    blob = json.dumps([job["cmd"], job["prereg"]["sha256"], runner_sha])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def gpu_busy() -> tuple[bool, str]:
    """True if another process holds a live lock or the GPU is heavily used."""
    if GPU_LOCK.exists():
        try:
            lock = json.loads(GPU_LOCK.read_text(encoding="utf-8"))
            pid = int(lock.get("pid", -1))
        except Exception:
            pid = -1
        if pid > 0:
            try:
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                STILL_ACTIVE = 259
                k32 = ctypes.windll.kernel32
                h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if h:
                    code = ctypes.c_ulong(0)
                    k32.GetExitCodeProcess(h, ctypes.byref(code))
                    k32.CloseHandle(h)
                    if code.value == STILL_ACTIVE:
                        return True, f"lock held by live pid {pid} ({lock.get('job')})"
            except Exception:
                return True, f"lock present for pid {pid}; cannot verify -- refusing"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        util = int(float(out[0])) if out else 0
        if util >= 80:
            return True, f"nvidia-smi reports {util}% utilization"
    except Exception:
        pass  # no nvidia-smi: rely on the lock only
    return False, ""


def _budget_path() -> Path:
    """Resolved lazily so sandboxed tests (patched ROOT) never see the real file."""
    return ROOT / "research" / "phase20_exhaustive" / "campaign_budget.json"


def budget_check(is_gpu: bool) -> tuple[bool, str]:
    """Hard GPU ceiling guard (compute_budget_extension_22h.md sec.7).

    Fail-open when no budget file exists (sandboxed test runs, fresh checkouts);
    fail-closed whenever a cap is declared and the balance is exhausted.  Also
    advances the consumed estimate by a finished GPU job's wall time.
    """
    path = _budget_path()
    if not is_gpu or not path.is_file():
        return True, "no active GPU budget file; run allowed"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        ext = doc.get("active_gpu_budget") or {}
        cap = ext.get("cap_gpu_hours")
        used = ext.get("consumed_gpu_hours_estimate")
    except Exception:
        return False, "GPU budget file unreadable; refusing unaccounted compute"
    if ext.get("hardware_stop"):
        return False, "GPU hardware stop active; repeated driver faults require resolution before GPU compute"
    if ext.get("unresolved_gpu_runtime"):
        return False, "GPU runtime accounting unresolved; reconcile interrupted job before new GPU compute"
    if cap is None or used is None:
        return True, "budget file declares no active cap; run allowed"
    remaining = float(cap) - float(used)
    if remaining <= 0:
        return False, f"GPU budget exhausted: {used:.2f}h consumed of {cap}h authorized"
    return True, f"GPU budget OK: {remaining:.2f}h of {cap}h remaining"


def budget_charge(seconds: float) -> None:
    """Add a finished GPU job's wall time to the consumed estimate (append-only credit)."""
    path = _budget_path()
    if not path.is_file():
        return
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        ext = doc["active_gpu_budget"]
        ext["consumed_gpu_hours_estimate"] = round(
            float(ext["consumed_gpu_hours_estimate"]) + seconds / 3600.0, 3)
        ext["remaining_gpu_hours"] = round(
            float(ext["cap_gpu_hours"]) - ext["consumed_gpu_hours_estimate"], 3)
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        pass  # accounting is best-effort; it never gates a run


def acquire_gpu_lock(job_id: str, pid: int) -> None:
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    GPU_LOCK.write_text(json.dumps(
        {"job": job_id, "pid": pid, "since": utcnow()}), encoding="utf-8")


def release_gpu_lock(job_id: str) -> None:
    if GPU_LOCK.exists():
        try:
            lock = json.loads(GPU_LOCK.read_text(encoding="utf-8"))
        except Exception:
            lock = {}
        if lock.get("job") == job_id:
            GPU_LOCK.unlink()


def verify_expect(job: dict) -> tuple[bool, str]:
    """§8 DONE-manifest check: report status/metrics, prediction sha, row count."""
    exp = job.get("expect")
    if not exp:
        return True, "no manifest declared"
    report = ROOT / exp["report"]
    if not report.is_file():
        return False, f"report missing: {exp['report']}"
    try:
        rep = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"report unreadable: {exc}"
    if rep.get("status") != "DONE":
        return False, f"report status={rep.get('status')!r} != DONE"
    if not rep.get("metrics"):
        return False, "report metrics are null"
    if "eval_rows" in exp and int(rep.get("eval_rows", -1)) != int(exp["eval_rows"]):
        return False, f"eval_rows {rep.get('eval_rows')} != {exp['eval_rows']}"
    if "prediction" in exp:
        pred = ROOT / exp["prediction"]
        if not pred.is_file():
            return False, f"prediction missing: {exp['prediction']}"
        want = rep.get("prediction_sha256")
        got = arr_sha256(__import__("numpy").load(pred))
        if not want or got != want:
            return False, f"prediction sha mismatch {got[:16]} != {str(want)[:16]}"
    return True, "manifest verified"


def cmd_init(_args) -> int:
    cat = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    doc = load() if SCHED.exists() else {
        "schema": "p20-scheduler-v1",
        "created_utc": utcnow(),
        "updated_utc": utcnow(),
        "ranking_rule": "expected private benefit / (GPU-h + wall-days risk); deadline 2026-10-31T23:59Z",
        "gpu_lock": None,
        "jobs": {},
    }
    jobs = doc["jobs"]
    added = 0
    for row in cat["experiments"]:
        num = row["experiment_number"]
        jid = f"J{int(num):03d}"
        if jid in jobs:  # never clobber existing state
            continue
        status = row["status"]
        state = STATUS_MAP[status]
        br = (row.get("blocking_reason") or "").strip()
        note = None
        if status == "NOT_STARTED" and br in DEP_REASONS:
            state = "DEPENDENCY_BLOCKED"
            note = br
        elif status == "NOT_STARTED" and br == "low-value":
            note = br  # stays PENDING, ranked low by `next`
        jobs[jid] = {
            "id": jid,
            "exp_row": num,
            "family": row["research_family"],
            "title": row["canonical_name"],
            "priority": row.get("priority", "?"),
            "state": state,
            "state_note": note,
            "kind": "unknown",       # cpu | gpu | network | unknown
            "cmd": None,             # registered via set-cmd once implemented
            "prereg": None,          # {path, sha256} via `prereg`
            "fingerprint": None,
            "depends_on": [],
            "attempts": 0,
            "expect": None,          # DONE-manifest contract once runnable
            "result": {"verdict": status if state in TERMINAL else None,
                       "report": None},
            "history": [{"ts": utcnow(), "event": "init", "from": status}],
        }
        added += 1
    save(doc)
    print(f"init: {added} jobs added, {len(jobs)} total")
    return 0


def _job_or_die(doc: dict, jid: str) -> dict:
    job = doc["jobs"].get(jid)
    if job is None:
        raise SystemExit(f"unknown job {jid}")
    return job


def cmd_list(args) -> int:
    doc = load()
    for jid in sorted(doc["jobs"]):
        job = doc["jobs"][jid]
        if args.state and job["state"] != args.state:
            continue
        if args.kind and job["kind"] != args.kind:
            continue
        fp = job["fingerprint"] or "-"
        print(f"{jid} r{int(job['exp_row']):03d} {job['state']:20s} {job['kind']:7s} "
              f"{fp:16s} {job['title'][:50]}")
    return 0


def cmd_show(args) -> int:
    doc = load()
    print(json.dumps(_job_or_die(doc, args.job), indent=1))
    return 0


def _refresh_fingerprint(job: dict) -> None:
    job["fingerprint"] = fingerprint(job)


def cmd_set_cmd(args) -> int:
    doc = load()
    job = _job_or_die(doc, args.job)
    job["cmd"] = list(args.cmd)
    job["kind"] = args.kind or job["kind"]
    if args.expect:
        job["expect"] = json.loads(args.expect)
    _refresh_fingerprint(job)
    job["history"].append({"ts": utcnow(), "event": "set-cmd", "cmd": job["cmd"]})
    if job["state"] in TERMINAL and job["state"] != "CLOSED_NEGATIVE":
        job["state"] = "PENDING"  # a real invocation exists now; un-stick it
    save(doc)
    print(f"{args.job}: cmd set, fingerprint={job['fingerprint']}")
    return 0


def cmd_prereg(args) -> int:
    doc = load()
    job = _job_or_die(doc, args.job)
    p = (ROOT / args.path) if not Path(args.path).is_absolute() else Path(args.path)
    if not p.is_file():
        raise SystemExit(f"prereg not found: {p}")
    job["prereg"] = {"path": p.relative_to(ROOT).as_posix(),
                     "sha256": sha256_file(p)}
    _refresh_fingerprint(job)
    job["history"].append({"ts": utcnow(), "event": "prereg",
                           "sha256": job["prereg"]["sha256"]})
    save(doc)
    print(f"{args.job}: prereg {job['prereg']['sha256'][:16]}, "
          f"fingerprint={job['fingerprint']}")
    return 0


def cmd_block(args) -> int:
    if args.state not in STATES:
        raise SystemExit(f"state must be one of {STATES}")
    doc = load()
    job = _job_or_die(doc, args.job)
    prev = job["state"]
    job["state"] = args.state
    job["state_note"] = args.why
    job["history"].append({"ts": utcnow(), "event": "block", "from": prev,
                           "to": args.state, "why": args.why})
    save(doc)
    print(f"{args.job}: {prev} -> {args.state} ({args.why})")
    return 0


def runnable(job: dict, require_prereg: bool = True) -> tuple[bool, str]:
    if job["state"] not in ("PENDING", "FAILED"):
        return False, f"state={job['state']}"
    if not job.get("cmd"):
        return False, "NOT_IMPLEMENTED (no cmd)"
    if require_prereg and not (job.get("prereg") or {}).get("sha256"):
        return False, "no prereg sha"
    fp = job.get("fingerprint")
    if fp:
        for other in load()["jobs"].values():  # fingerprint never rerun
            if other["id"] != job["id"] and other.get("fingerprint") == fp \
                    and other["state"] == "DONE":
                return False, f"identical fingerprint already DONE as {other['id']}"
    for dep in job.get("depends_on", []):
        if load()["jobs"][dep]["state"] not in ("DONE", "PROMOTED"):
            return False, f"blocked on {dep}"
    return True, "runnable"


def cmd_next(args) -> int:
    doc = load()
    picked = []
    for jid in sorted(doc["jobs"]):
        job = doc["jobs"][jid]
        if args.kind and job["kind"] != args.kind:
            continue
        if job.get("state_note") == "low-value":
            continue  # ranked last by design
        ok, why = runnable(job)
        if ok:
            picked.append(jid)
        elif args.verbose:
            print(f"{jid}: not runnable -- {why}")
    if not picked:
        print("no runnable job")
        return 1
    for jid in picked:
        job = doc["jobs"][jid]
        print(f"{jid} r{int(job['exp_row']):03d} {job['kind']:7s} "
              f"fp={job['fingerprint']} {job['title'][:50]}")
    return 0


def cmd_resume(_args) -> int:
    """Crash recovery: RUNNING whose pid is dead becomes FAILED (never DONE)."""
    if not SCHED.exists():
        print("nothing to resume")
        return 0
    doc = load()
    n = 0
    for job in doc["jobs"].values():
        if job["state"] != "RUNNING":
            continue
        pid = (job.get("history") or [{}])[-1].get("pid", -1)
        alive = False
        if pid > 0:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)
            if h:
                code = ctypes.c_ulong(0)
                k32.GetExitCodeProcess(h, ctypes.byref(code))
                k32.CloseHandle(h)
                alive = code.value == 259
        if not alive:
            job["state"] = "FAILED"
            job["state_note"] = "process died (resume)"
            job["history"].append({"ts": utcnow(), "event": "resume->FAILED", "pid": pid})
            n += 1
    if doc.get("gpu_lock"):
        lock_pid = doc["gpu_lock"].get("pid", -1)
        # stale lock without process is cleared here too
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, lock_pid) if lock_pid > 0 else None
        if h:
            k32.CloseHandle(h)
        else:
            doc["gpu_lock"] = None
    save(doc)
    print(f"resume: {n} RUNNING -> FAILED")
    return 0


def cmd_run(args) -> int:
    doc = load()
    job = _job_or_die(doc, args.job)
    ok, why = runnable(job, require_prereg=not args.no_prereg)
    if not ok:
        raise SystemExit(f"REFUSING {args.job}: {why}")
    is_gpu = job["kind"] == "gpu"
    if is_gpu:
        ok_budget, budget_note = budget_check(True)
        if not ok_budget:
            raise SystemExit(f"REFUSING GPU job {args.job}: {budget_note}")
        runtime_append({"ts": utcnow(), "job": args.job, "event": "budget-check",
                        "detail": budget_note})
        busy, reason = gpu_busy()
        if busy:
            raise SystemExit(f"REFUSING GPU job {args.job}: {reason}")
        # Keep a real, live scheduler PID in the lock throughout the subprocess.
        acquire_gpu_lock(args.job, os.getpid())

    # fingerprint-never-rerun also checked against a fresh load (raced writes)
    job["state"] = "RUNNING"
    job["attempts"] += 1
    job["history"].append({"ts": utcnow(), "event": "run-start",
                           "pid": os.getpid(), "cmd": job["cmd"], "why": args.why,
                           "fingerprint": job["fingerprint"]})
    save(doc)
    t0 = time.time()
    try:
        # Omitted --timeout means no deadline; timeout=0 aborts normal jobs.
        proc = subprocess.run(job["cmd"], cwd=ROOT,
                              timeout=args.timeout if args.timeout > 0 else None)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        exit_code = -999
    except Exception as exc:
        exit_code = -998
        job["history"].append({"ts": utcnow(), "event": "run-exception",
                               "error": repr(exc)})
    secs = round(time.time() - t0, 1)

    if exit_code == 0:
        good, detail = verify_expect(job)
        if good:
            job["state"] = "DONE"
            job["state_note"] = detail
            job["result"]["report"] = (job.get("expect") or {}).get("report")
        else:
            job["state"] = "FAILED"
            job["state_note"] = f"DONE-manifest rejected: {detail}"
    else:
        job["state"] = "FAILED"
        job["state_note"] = f"exit={exit_code}"
    job["history"].append({"ts": utcnow(), "event": "run-end",
                           "exit": exit_code, "seconds": secs,
                           "state": job["state"]})
    if is_gpu:
        release_gpu_lock(args.job)
        budget_charge(secs)
    save(doc)
    runtime_append({
        "ts": utcnow(), "job": args.job, "exp_row": job["exp_row"],
        "family": job["family"], "kind": job["kind"], "gpu": is_gpu,
        "cmd": job["cmd"], "fingerprint": job["fingerprint"],
        "prereg_sha256": (job.get("prereg") or {}).get("sha256"),
        "seconds": secs, "exit": exit_code, "state_after": job["state"],
        "why": args.why,
    })
    print(f"{args.job}: {job['state']} ({secs}s, exit={exit_code}) "
          f"{job['state_note']}")
    return 0 if job["state"] == "DONE" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="fn", required=True)
    sub.add_parser("init")
    for name in ("list", "next", "resume"):
        p = sub.add_parser(name)
        if name == "list":
            p.add_argument("--state")
            p.add_argument("--kind")
        if name == "next":
            p.add_argument("--kind")
            p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("show")
    p.add_argument("job")
    p = sub.add_parser("set-cmd")
    p.add_argument("job")
    p.add_argument("--kind", choices=["cpu", "gpu", "network"])
    p.add_argument("--expect", default=None, help="DONE-manifest JSON (inline)")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p = sub.add_parser("prereg")
    p.add_argument("job")
    p.add_argument("path")
    p = sub.add_parser("block")
    p.add_argument("job")
    p.add_argument("state")
    p.add_argument("--why", default="")
    p = sub.add_parser("run")
    p.add_argument("job")
    p.add_argument("--no-prereg", action="store_true",
                   help="allow a read-only diagnostic without prereg (audits)")
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--why", default="")
    args = ap.parse_args()
    if args.fn == "set-cmd" and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return args.fn and {"init": cmd_init, "list": cmd_list, "show": cmd_show,
                        "set-cmd": cmd_set_cmd, "prereg": cmd_prereg,
                        "block": cmd_block, "next": cmd_next,
                        "run": cmd_run, "resume": cmd_resume}[args.fn](args)


if __name__ == "__main__":
    raise SystemExit(main())
