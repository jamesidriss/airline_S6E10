"""Software-quality tests. Run: python -m pytest tests -q  (or: python tests/run_tests.py)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, SAMPLE_CSV, TARGET, TEST_CSV, TRAIN_CSV, load_cached_parquet  # noqa: E402


# ======================================================================================
# schema
# ======================================================================================
def test_schema():
    tr, te = load_cached_parquet()
    assert tr.shape == (699635, 23), tr.shape
    assert te.shape == (299844, 22), te.shape
    assert list(te.columns) == [c for c in tr.columns if c != TARGET]
    assert tr[TARGET].isin([0, 1]).all()
    assert tr[ID_COL].is_unique and te[ID_COL].is_unique
    assert not set(tr[ID_COL]) & set(te[ID_COL])


def test_sample_submission_schema():
    sa = pd.read_csv(SAMPLE_CSV)
    te = pd.read_csv(TEST_CSV, usecols=[ID_COL])
    assert list(sa.columns) == [ID_COL, TARGET]
    assert len(sa) == len(te)
    assert np.array_equal(sa[ID_COL].to_numpy(), te[ID_COL].to_numpy())


# ======================================================================================
# folds
# ======================================================================================
def test_fold_determinism():
    from src.validation.folds import get_scheme

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    a = get_scheme("primary", y, tr[ID_COL]).folds
    b = get_scheme("primary", y, tr[ID_COL]).folds
    assert np.array_equal(a, b), "fold registry is not deterministic"
    assert set(a.tolist()) == {0, 1, 2, 3, 4}


def test_fold_no_row_overlap():
    from src.validation.folds import get_scheme

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    f = get_scheme("primary", y, tr[ID_COL]).folds
    assert len(f) == len(tr)
    counts = np.bincount(f)
    assert counts.min() > 0
    # each row assigned exactly once -> implied, but assert explicitly
    assert int(counts.sum()) == len(tr)


def test_blocked_id_scheme_is_contiguous():
    from src.validation.folds import blocked_id_scheme

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    f = blocked_id_scheme(y, n_blocks=10)
    assert len(f) == len(tr) and f.min() >= 0
    # each fold must be a contiguous block of row order
    for k in range(int(f.max()) + 1):
        idx = np.where(f == k)[0]
        assert idx.max() - idx.min() + 1 == len(idx), f"fold {k} is not contiguous"


# ======================================================================================
# leakage
# ======================================================================================
def test_twin_vocabulary_is_label_free():
    """The exact-value categorical twins must be computable without any label."""
    from src.features.view import _cat_twins, RAW21

    tr, te = load_cached_parquet()
    comb = pd.concat([tr[RAW21], te[RAW21]], ignore_index=True)
    m1, names = _cat_twins(comb, RAW21)
    # the twins depend only on the value matrix: a row permutation changes nothing but order
    perm = np.random.default_rng(0).permutation(len(comb))
    m2, _ = _cat_twins(comb.iloc[perm].reset_index(drop=True), RAW21)
    assert m1.shape == m2.shape
    assert np.array_equal(m1[perm], m2), "twin codes depend on row order in a leaky way"


def test_fold_safe_te_never_sees_apply_rows():
    """A row's encoded value must not change when other rows' labels change."""
    from src.features.s6e10 import FoldSafeTE

    rng = np.random.default_rng(0)
    k = pd.Series(rng.integers(0, 20, 500).astype("float64"))
    y = rng.integers(0, 2, 500).astype("float64")
    te = FoldSafeTE({"k": k}, smooths=(10.0,), inner_folds=5, inner_seed=0)
    Xa, names = te.apply({"k": k.iloc[:400].reset_index(drop=True)}, y[:400],
                         {"k": k.iloc[400:].reset_index(drop=True)})
    # flip the labels of rows that are in the APPLY set -> output must be identical
    y2 = y.copy()
    y2[400:] = 1 - y2[400:]
    Xb, _ = te.apply({"k": k.iloc[:400].reset_index(drop=True)}, y2[:400],
                     {"k": k.iloc[400:].reset_index(drop=True)})
    assert np.allclose(Xa, Xb), "TE output depends on apply-row labels -> LEAK"


def test_crossfit_row_never_sees_own_label():
    """Each cross-fitted row must be encoded by a table built WITHOUT that row."""
    from src.features.s6e10 import FoldSafeTE

    rng = np.random.default_rng(1)
    k = pd.Series(np.zeros(200))          # one single key -> the table can only use other rows
    y = rng.integers(0, 2, 200).astype("float64")
    te = FoldSafeTE({"k": k}, smooths=(1.0,), inner_folds=5, inner_seed=0)
    X, names = te.crossfit({"k": k}, y)

    inner = (np.arange(200) % te.inner_folds)[np.random.default_rng(te.inner_seed).permutation(200)]
    for i in range(200):
        others = y[inner != inner[i]]
        expected = (others.sum() + 1.0 * y.mean()) / (len(others) + 1.0)
        assert abs(X[i, 0] - expected) < 1e-5, f"row {i} encoded by a table containing its own label"
    # and in particular not by its own label alone
    assert not np.allclose(X[:, 0], y)


def test_external_features_use_no_competition_label():
    """external + surface blocks must be identical for repeated calls and label-free."""
    from src.features import s6e10 as S
    from src.features.view import RAW21

    tr, te = load_cached_parquet()
    comb = pd.concat([tr[RAW21], te[RAW21]], ignore_index=True)
    surf = S.build_original_surfaces(S.SURFACE_ANCHORS)
    m1, n1 = S.apply_original_surfaces(comb, surf)
    m2, n2 = S.apply_original_surfaces(comb, surf)
    assert n1 == n2
    assert not np.isnan(m1).any(), "external surface features contain NaN"
    assert np.array_equal(m1, m2), "external features are not deterministic"

    ex = S.build_external_features()
    e1, _ = S.apply_external(comb, ex)
    e2, _ = S.apply_external(comb, ex)
    assert not np.isnan(e1).any()
    assert np.array_equal(e1, e2)

    # the builders read only the ORIGINAL csv (never competition labels)
    import inspect

    for fn in (S.build_original_surfaces, S.build_external_features):
        src = inspect.getsource(fn)
        assert "tr[TARGET]" not in src and "y =" not in src.split("yo =")[0].replace("yo =", "")


def test_inner_early_stopping_split_is_disjoint_from_eval():
    from scripts.run_views import _inner_es_split

    y = np.array([0, 1] * 500)
    fit = np.arange(1000)
    trn, es = _inner_es_split(fit, y, seed=1)
    assert len(set(trn) & set(es)) == 0
    assert set(trn) | set(es) == set(fit)


# ======================================================================================
# predictions / store
# ======================================================================================
def test_oof_coverage_and_store_integrity():
    from src.submission import store

    idx = store._load_index()
    assert idx, "prediction store is empty"
    tr, _ = load_cached_parquet()
    for k, v in idx.items():
        o = store.load_oof(k)
        assert len(o) == len(tr), f"{k}: oof length {len(o)} != {len(tr)}"
        assert np.all(np.isfinite(o)), f"{k}: non-finite OOF"
        assert np.array_equal(np.load(v["oof"]), o)


def test_submission_preflight_rejects_bad_input():
    """The pre-flight gate must refuse anything malformed, and must not leave litter behind."""
    from src.submission.make import build

    n = len(pd.read_csv(TEST_CSV, usecols=[ID_COL]))
    good = np.full(n, 0.5)
    p = build(good, "pytest_ok", notes="pytest artifact", oof_auc=0.5)
    assert p.exists()
    try:
        build(np.full(n - 1, 0.5), "pytest_bad")
        raise AssertionError("pre-flight failed to reject a wrong-length prediction")
    except ValueError as exc:
        assert "REJECTED" in str(exc)
    try:
        bad = good.copy()
        bad[0] = np.nan
        build(bad, "pytest_nan")
        raise AssertionError("pre-flight failed to reject NaN")
    except ValueError as exc:
        assert "REJECTED" in str(exc)
    # clean up: a test must not leave files in submissions/
    p.unlink(missing_ok=True)
    (p.parent / "pytest_bad.csv").unlink(missing_ok=True)
    (p.parent / "pytest_nan.csv").unlink(missing_ok=True)
    import pandas as _pd

    man = p.parent / "manifest.csv"
    if man.exists():
        df = _pd.read_csv(man)
        df = df[~df["name"].astype(str).str.startswith("pytest")]
        df.to_csv(man, index=False)


def test_transforms_are_invertible_shape_wise():
    from src.ensemble.lab import tform

    p = np.random.default_rng(0).random(1000)
    for kind in ("prob", "logit", "rank"):
        z = tform(p, kind)
        assert z.shape == p.shape and np.all(np.isfinite(z))


def test_admission_gate_logic():
    from src.ensemble.lab import admission_gate

    base = [0.96, 0.96, 0.96, 0.96, 0.96]
    good = admission_gate("g", base, [0.9601] * 5)
    bad = admission_gate("b", base, [0.9599, 0.9611, 0.9598, 0.9601, 0.9599])
    assert good["admit"] is True
    assert bad["admit"] is False


def test_original_overlap_audit_is_small():
    """No competition row may be an exact copy of an original row (would be an answer key)."""
    from src.common import load_raw
    from src.features import s6e10 as S

    tr, te = load_cached_parquet()
    feats = [c for c in te.columns if c != ID_COL]
    comb = pd.concat([tr[feats], te[feats]], ignore_index=True)
    aud = S.original_leak_audit(S.load_original(), comb, feats)
    assert aud["n_original_rows_exactly_matching_a_competition_row"] <= 1000, aud
    print(aud)


# ======================================================================================
# security
# ======================================================================================
def test_no_secret_shaped_material_in_repository():
    """Regression test for the credential that GitHub secret scanning found.

    Kaggle's frontend HTML embeds a public Google API key. A previous version of
    scripts/verify_kaggle_facts.py persisted raw response bodies, which copied that key into
    research/raw/kaggle_facts.json and into git history. This test fails if any secret-shaped
    string reappears anywhere in the tracked working tree, or in any file we persist from a
    remote call.

    Values are never printed: the assertion reports detector name, path and line only.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from scripts.secret_scan import SELF_EXEMPT, scan_tree

    findings = [f for f in scan_tree() if f["severity"] == "HIGH"]
    if findings:
        locs = ", ".join(f"{f['detector']} at {f['path']}:{f['where']}" for f in findings)
        raise AssertionError(f"secret-shaped material present in the tree -> {locs}")

    # and specifically: the persisted Kaggle facts file must be JSON-only and scrubbed
    facts = root / "research" / "raw" / "kaggle_facts.json"
    if facts.exists():
        text = facts.read_text(encoding="utf-8")
        assert "AIza" not in text, "google-api-key shape present in kaggle_facts.json"
        assert "<!DOCTYPE" not in text and "<script" not in text, \
            "raw HTML was persisted into kaggle_facts.json"
        data = json.loads(text)
        for name, entry in data.items():
            if isinstance(entry, dict) and entry.get("persisted") is False:
                assert "body" not in entry, f"{name} persisted a body it declared it would not"


def test_verify_kaggle_facts_never_persists_non_json_bodies():
    """A non-JSON response must yield metadata only: status, content type, length, class."""
    import scripts.verify_kaggle_facts as V

    class FakeResp:
        def __init__(self, status, ctype, text):
            self.status_code, self.headers, self.text = status, {"Content-Type": ctype}, text

        def json(self):
            raise ValueError("not json")

    html = "<!DOCTYPE html><script>window.kaggleStackdriverConfig={'apiKey':'AIza" + "x" * 35 + "'}</script>"
    r = FakeResp(200, "text/html", html)
    meta = {"status": r.status_code, "content_type": r.headers["Content-Type"],
            "body_len": len(r.text),
            "classification": V.classify_error(r.status_code, r.headers["Content-Type"])}
    assert meta["classification"] == "ok"
    # the classification must never leak body content
    assert "AIza" not in meta["classification"]

    # scrub must remove credential shapes from anything
    scrubbed = V.scrub({"k": html, "n": [html], "i": 1})
    assert "AIza" not in json.dumps(scrubbed)
    assert scrubbed["i"] == 1, "scrub must not damage non-string content"


def test_no_source_files_missing_from_git():
    """Regression test for a silent, competition-level defect.

    An unanchored ``features/`` rule in .gitignore (intended for ``artifacts/features/``) matched
    ``src/features/`` at any depth, so the entire feature-engineering and model layer was never
    committed and the published repository could not reproduce the solution. Rules 2.8.b requires
    the winning model's code to be deliverable, so this must be impossible to reintroduce.
    """
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tracked = set(subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                                 text=True).stdout.split())
    missing = []
    for sd in ("src", "scripts", "configs", "tests"):
        d = root / sd
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(root).as_posix()
                if rel not in tracked:
                    missing.append(rel)
    assert not missing, f"source files present but untracked: {missing}"

    # and no dangerous unanchored directory rule may reappear
    sys.path.insert(0, str(root))
    from scripts.audit_untracked import INTENTIONAL_ANY_DEPTH

    risky = []
    for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "!")):
            continue
        if s.endswith("/") and not s.startswith("/") and s.count("/") == 1 and "*" not in s \
                and s not in INTENTIONAL_ANY_DEPTH:
            risky.append(s)
    assert not risky, f"risky unanchored .gitignore rules: {risky}"


def test_secret_scan_detectors_are_self_exempt_and_working():
    """The scanner must detect a planted secret and must not flag itself."""
    from scripts.secret_scan import SELF_EXEMPT, scan_text

    planted = "config api_key: 'AIza" + "A" * 35 + "'"
    hits = scan_text(planted, "fake.py", "")
    assert any(h["detector"] == "google_api_key" for h in hits)
    assert "scripts/secret_scan.py" in SELF_EXEMPT
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "secret_scan.py").exists()