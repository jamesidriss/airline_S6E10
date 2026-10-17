"""Idempotently certify and backfill the five measured S7 repair experiments.

Read-only with respect to trained artifacts and historical experiment ledger.
Updates only the scheduler metadata after verifying every prediction hash.
No model training, no implicit experiment re-execution.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.phase20 import scheduler as sch  # noqa: E402

S7_PREREG = "configs/phase20/s7_fidelity_repair_prereg.json"
S7_SHA256 = "c10cf3fec2efa1e35eb561009afcf72f717dbc0274490df46fb049145e79e5ed"

SPECS = {
    "J015": {
        "kind": "gpu", "script": "scripts/phase20/neural_screen_fold0.py",
        "args": ["--model", "fttransformer", "--view", "full", "--batch-size", "512",
                 "--infer-chunk", "2048", "--tag", "fttransformer_full_bounded_fold0"],
        "tag": "fttransformer_full_bounded_fold0", "folder": "neural_screen",
        "note": "FTTransformer full F2: 0.958639, portfolio -6.167e-4; inference chunk repair",
    },
    "J030": {
        "kind": "gpu", "script": "scripts/phase20/neural_screen_fold0.py",
        "args": ["--model", "interactionnet", "--view", "full", "--infer-chunk", "4096",
                 "--tag", "interactionnet_full_bounded_fold0"],
        "tag": "interactionnet_full_bounded_fold0", "folder": "neural_screen",
        "note": "InteractionNet full F2: 0.958710, portfolio -9.388e-4; inference chunk repair",
    },
    "J017": {
        "kind": "gpu", "script": "scripts/phase20/neural_screen_fold0.py",
        "args": ["--model", "dcnv2full", "--view", "full",
                 "--prereg", S7_PREREG, "--tag", "dcnv2full_full_fold0"],
        "tag": "dcnv2full_full_fold0", "folder": "neural_screen",
        "note": "Canonical full-matrix DCNv2 F2: 0.959041, portfolio -9.779e-4; route measured separately",
    },
    "J065": {
        "kind": "gpu", "script": "scripts/phase20/neural_screen_fold0.py",
        "args": ["--model", "factlog", "--view", "full",
                 "--prereg", S7_PREREG, "--tag", "factlog_full_fold0",
                 "--infer-chunk", "32768"],
        "tag": "factlog_full_fold0", "folder": "neural_screen",
        "note": "Rank-16 factorized logistic F2: 0.959788, portfolio -6.815e-4",
    },
    "J064": {
        "kind": "cpu", "script": "scripts/phase20/classical_screen_fold0.py",
        "args": ["--model", "rffsvm", "--view", "full", "--tag", "rffsvm_full_fold0"],
        "tag": "rffsvm_full_fold0", "folder": "classical_screen",
        "note": "RFF SVM F2: 0.929997, portfolio -2.175e-3",
    },
}


def main() -> int:
    p = ROOT / S7_PREREG
    assert hashlib.sha256(p.read_bytes()).hexdigest() == S7_SHA256
    doc = sch.load()
    verified = 0
    modified = 0
    for jid, spec in SPECS.items():
        job = doc["jobs"][jid]
        assert job["state"] == "CLOSED_NEGATIVE", (jid, job["state"])
        report = f"reports/phase20/{spec['folder']}/{spec['tag']}.json"
        prediction = f"artifacts/phase20/{spec['folder']}/{spec['tag']}.npy"
        report_data = json.loads((ROOT / report).read_text(encoding="utf-8"))
        assert report_data["status"] == "DONE", jid
        assert report_data["eval_rows"] == 139927, jid
        assert report_data["metrics"]["portfolio_delta"] < 0, jid
        meta = {
            "kind": spec["kind"],
            "cmd": [sys.executable, str(ROOT / spec["script"]), *spec["args"]],
            "prereg": {"path": S7_PREREG, "sha256": S7_SHA256},
            "expect": {"report": report, "prediction": prediction, "eval_rows": 139927},
        }
        tmp = dict(job, **meta)
        good, reason = sch.verify_expect(tmp)
        if not good:
            raise RuntimeError(f"{jid}: verified result rejected: {reason}")
        new_fingerprint = sch.fingerprint(tmp)
        assert new_fingerprint, jid
        changed = any(job.get(k) != value for k, value in meta.items()) or job.get("fingerprint") != new_fingerprint
        if changed:
            job.update(meta)
            job["fingerprint"] = new_fingerprint
            job["state_note"] = spec["note"]
            job["result"] = {"verdict": "COMPLETE_NEGATIVE", "report": report}
            job["history"].append({
                "ts": sch.utcnow(), "event": "s7-verified-provenance-repair",
                "source_script": spec["script"], "source_sha256": sch.sha256_file(ROOT / spec["script"]),
                "governing_s7_prereg_sha256": S7_SHA256,
                "runner_recipe_prereg": report_data.get("prereg"),
                "report": report, "prediction": prediction, "fingerprint": new_fingerprint,
                "verification": reason, "note": spec["note"],
            })
            modified += 1
        verified += 1
        print(f"{jid}: {reason}; delta={report_data['metrics']['portfolio_delta']:+.9f}; fp={new_fingerprint}")
    if modified:
        sch.save(doc)
    print(f"S7 scheduler: {verified} reports verified, {modified} records updated; no model retraining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
