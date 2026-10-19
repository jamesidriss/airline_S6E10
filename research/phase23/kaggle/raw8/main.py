#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S6E10 Phase-23 RAW8 -- native TabPFN-3.5 raw-context 8-estimator member.

Tests the strongest unresolved lead of Phase 22: an independently produced,
fold-safe, native TabPFN-3.5 RAW model with eight internal estimators
(the public `tabpfn35` vector: raw 21 columns, n_estimators=8, full labelled
context, mixed precision; advertised raw CV 0.961145 reproduced locally as
OOF AUC 0.9611450967635963; +2.995e-5 full-OOF gain vs v7 at w=0.10 with all
five primary folds positive).  This kernel produces OUR OWN version of that
member under OUR certified protocol (immutable primary folds seed=20261010,
our checkpoint, our full-context geometry) so the hypothesis can be judged on
our own evidence instead of an unverifiable external artefact.

It runs entirely on free Kaggle GPU quota (T4), never on the local RTX 5070 Ti
(which is under a hardware stop), and never spends the local 22 h budget.

Modes
-----
PROBE = True   -- validate environment, hashes, patch equivalence and timing.
PROBE = False  -- production: 40 units (5 folds x 8 members), then assemble.

Worker (internal CLI, spawned by the parent):
    python main.py --unit FOLD MEMBER --gpu 0 --deadline UNIX_TS --ckpt PATH
    python main.py --assemble
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ------------------------------------------------------------------ constants
PROBE = True                      # flipped to False for the production push
LIMIT_H = 10.6                    # internal wall-clock stop (session limit 12 h)
MIN_TAIL_S = 900.0                # never start a unit with less than this left

RAW21 = ['Inflight wifi service', 'Departure/Arrival time convenient',
         'Ease of Online booking', 'Gate location', 'Food and drink',
         'Online boarding', 'Seat comfort', 'Inflight entertainment',
         'On-board service', 'Leg room service', 'Baggage handling',
         'Checkin service', 'Cleanliness', 'Gender', 'Customer Type',
         'Type of Travel', 'Class', 'Age', 'Flight Distance',
         'Departure Delay in Minutes', 'Arrival Delay in Minutes']
META4 = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
TARGET = 'satisfaction'

CKPT_REPO = 'Prior-Labs/tabpfn_3_5'
CKPT_FILE = 'tabpfn-v3.5-20260909.safetensors'
CKPT_REV = '06bf2ba35c80a92a3b9abb436b99cf49e7a0365e'
CKPT_SHA = 'ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3'

FOLD_SEED = 20261010              # immutable primary folds
MODEL_SEED = 1201                 # certified model random_state
N_EST = 8                         # the hypothesis: eight internal estimators
CHUNK_CELLS = 262144              # certified stock attribute (stages 0-2)
COL_CHUNK = 1                     # certified stock attribute
BATCH = 32768                     # prediction batch (fallback halves to 1024)
LEAN_CHUNK = 16384                # KV-build row chunk of the lean patch

WORK = Path('/kaggle/working')
UNITS = WORK / 'units'
DATA_NPZ = WORK / 'raw8_data.npz'

EXPECT = {
    'n_train': 699635,
    'n_test': 299844,
    'ids_sha': 'b8941ff79489583c5aae1e1343c3629ac6d1629bd5d4ddc1c545f7818ea38876',
    'idt_sha': '858650e1798a294a5ee43a52429349a4560264bf7769e20c97ba5c2320593f71',
    'y_sha': '5ab860613b7a26fc1aec55978ecde760cb2826f638152234e2b1da8b34a31d40',
    'folds_sha': 'b41f1d7ccf2fc504a1631bd8b84ba59b62b825895871949159a89f3f00a9f23b',
    'X_round6_sha': 'b2a49d79991068a745195550ca39c5f1889e4e57287f3a867cf492a6b3587790',
    'XT_round6_sha': 'a7004d08ddb9634f4403516868bc690a83d94474251a21ff47e45e1df6e4450d',
}


def log(*a):
    print('[raw8]', *a, flush=True)


def arr_sha256(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    if a.size:
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def round6(a):
    return np.round(np.asarray(a, dtype=np.float64), 6).astype(np.float64)


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ environment
def pip_install():
    log('pip install tabpfn==9.1.0 huggingface_hub ...')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'tabpfn==9.1.0', 'huggingface_hub', 'safetensors'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log('PIP STDOUT', r.stdout[-3000:])
        log('PIP STDERR', r.stderr[-3000:])
        raise SystemExit('pip install failed')
    log('pip ok')


def download_checkpoint() -> Path:
    from huggingface_hub import hf_hub_download
    cache = str(WORK / 'hf')
    p = Path(hf_hub_download(CKPT_REPO, CKPT_FILE, revision=CKPT_REV,
                             cache_dir=cache))
    sha = file_sha256(p)
    log('checkpoint:', p, 'sha256:', sha[:16], '...')
    if sha != CKPT_SHA:
        raise SystemExit(f'checkpoint sha mismatch: {sha}')
    return p


def gpu_report():
    r = subprocess.run('nvidia-smi', shell=True, capture_output=True, text=True)
    log(r.stdout)
    import torch
    log('torch', torch.__version__, 'cuda', torch.version.cuda,
        'devices', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        log(' gpu', i, torch.cuda.get_device_name(i),
            round(torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1), 'GiB')
    log('bf16_supported:', torch.cuda.is_bf16_supported())
    import tabpfn
    log('tabpfn', tabpfn.__version__)
    return torch


# ------------------------------------------------------------------ data
def find_competition_csvs():
    cands = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    tr = sorted(cands)[0] if cands else None
    for p in sorted(cands):
        if 's6e10' in p.lower():
            tr = p
            break
    if tr is None:
        raise SystemExit('no train.csv under /kaggle/input')
    te = str(Path(tr).parent / 'test.csv')
    if not Path(te).exists():
        raise SystemExit('no test.csv next to train.csv')
    return tr, te


def build_dataset():
    """Rebuild the certified `route=False` raw context and check every hash."""
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    tr_path, te_path = find_competition_csvs()
    log('train csv:', tr_path)
    log('test  csv:', te_path)
    tr = pd.read_csv(tr_path)
    te = pd.read_csv(te_path)
    if len(tr) != EXPECT['n_train'] or len(te) != EXPECT['n_test']:
        raise SystemExit(f'row count mismatch: {len(tr)} {len(te)}')

    y = tr[TARGET].astype(str).str.lower().map({'true': 1, 'false': 0})
    if y.isna().any():
        raise SystemExit('unmapped target values')
    y = y.to_numpy(dtype=np.int8)
    ids = tr['id'].to_numpy()
    idt = te['id'].to_numpy()

    if arr_sha256(ids) != EXPECT['ids_sha']:
        raise SystemExit('train ids hash mismatch')
    if arr_sha256(idt) != EXPECT['idt_sha']:
        raise SystemExit('test ids hash mismatch')
    if arr_sha256(y) != EXPECT['y_sha']:
        raise SystemExit('label hash mismatch')

    folds = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y), dtype=np.int8), y)):
        folds[va] = k
    if (folds < 0).any() or arr_sha256(folds) != EXPECT['folds_sha']:
        raise SystemExit('fold reconstruction mismatch')
    log('folds:', [int((folds == k).sum()) for k in range(5)])

    cov = pd.concat([tr[RAW21], te[RAW21]], ignore_index=True)
    for c in META4:
        levels = sorted(cov[c].unique().tolist())
        cov[c] = pd.Categorical(cov[c], categories=levels).codes
    cov = cov.fillna(-1)
    X = cov.iloc[:len(tr)].to_numpy(dtype='float32')
    XT = cov.iloc[len(tr):].to_numpy(dtype='float32')
    if arr_sha256(round6(X)) != EXPECT['X_round6_sha']:
        raise SystemExit('X content hash mismatch')
    if arr_sha256(round6(XT)) != EXPECT['XT_round6_sha']:
        raise SystemExit('XT content hash mismatch')
    log('dataset ok:', X.shape, XT.shape, 'pos_rate', float(y.mean()))
    return X, y, folds, XT, ids, idt


def ensure_data_npz():
    if not DATA_NPZ.exists():
        X, y, folds, XT, ids, idt = build_dataset()
        np.savez(DATA_NPZ, X=X, y=y, folds=folds, XT=XT, ids=ids, idt=idt)
        log('wrote', DATA_NPZ)
    z = np.load(DATA_NPZ)
    return z['X'], z['y'], z['folds'], z['XT'], z['ids'], z['idt']


# ------------------------------------------------------------------ lean patch
_LEAN_ORIG = None


def apply_lean_patch():
    """Memory-lean KV-cache build for TabPFN-3.5 (ported from the public
    hemingweb anchor notebook; measured equivalent at the certified 2e-6 gate).
    The stock fit_with_cache path runs every ICL layer over all ~560k context
    rows at once -> OOM on a 16 GB card.  Same math as the original."""
    global _LEAN_ORIG
    from tabpfn.architectures import tabpfn_v3_5 as A
    if _LEAN_ORIG is None:
        _LEAN_ORIG = A.ICLTransformerBlock.forward
    CH = LEAN_CHUNK

    def _lean_forward_impl(self, x_BRE, single_eval_pos, save_peak_memory_factor=None, *,
                           cached_kv=None, return_kv=False):
        if not return_kv or cached_kv is not None:
            return _LEAN_ORIG(self, x_BRE, single_eval_pos, save_peak_memory_factor,
                              cached_kv=cached_kv, return_kv=return_kv)
        import torch
        att = self.icl_attention
        x = x_BRE.contiguous()
        B, R, _ = x.shape
        N = R if single_eval_pos is None else single_eval_pos
        nh = att.num_kv_heads_test
        with torch.no_grad():
            K = V = None
            for s in range(0, N, CH):
                e = min(s + CH, N)
                h = self.layernorm(x[:, s:e])
                k = att.k_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)
                v = att.v_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)
                k = att.k_norm(k)
                if K is None:
                    K = torch.empty((B, N, att.num_kv_heads, att.head_dim),
                                    dtype=v.dtype, device=x.device)
                    V = torch.empty_like(K)
                K[:, s:e] = k.to(v.dtype)
                V[:, s:e] = v
                del h, k, v
            starts = list(range(0, N, CH)) + list(range(N, R, CH))
            for s in starts:
                e = min(s + CH, N if s < N else R)
                xc = x[:, s:e]
                q = att.q_norm(att.q_projection(self.layernorm(xc)).view(
                    B, e - s, att.num_heads, att.head_dim))
                if s < N or nh is None or N == R:
                    out = A._batched_scaled_dot_product_attention(
                        q, K, V, softmax_scaling_layer=att.softmax_scaling_layer)
                else:
                    out = A._batched_scaled_dot_product_attention(
                        q, K[:, :, :nh], V[:, :, :nh],
                        softmax_scaling_layer=att.softmax_scaling_layer)
                xc.add_(att.out_projection(out.reshape(B, e - s, att.head_dim * att.num_heads)))
                del q, out
            if nh is not None:
                k_cache, v_cache = K[:, :, :nh].contiguous(), V[:, :, :nh].contiguous()
            else:
                k_cache, v_cache = K, V
            del K, V
            kv_entry = A.KVCacheEntry(key=k_cache.detach(), value=v_cache.detach())
            for s in range(0, R, CH):
                xc = x[:, s:min(s + CH, R)]
                xc.add_(self.mlp(self.layernorm_mlp(xc)))
        return x, kv_entry

    A.ICLTransformerBlock.forward = _lean_forward_impl


def restore_stock_forward():
    from tabpfn.architectures import tabpfn_v3_5 as A
    if _LEAN_ORIG is not None:
        A.ICLTransformerBlock.forward = _LEAN_ORIG


# ------------------------------------------------------------------ certified port
def make_params(ckpt: Path, clf_device: str = 'cuda') -> dict:
    return dict(model_path=str(ckpt), n_estimators=N_EST,
                auto_scale_n_estimators=False, random_state=MODEL_SEED,
                device=clf_device, ignore_pretraining_limits=True,
                fit_mode='fit_with_cache', kv_cache_precision='int8',
                keep_cache_on_device=True, inference_precision='autocast',
                categorical_features_indices=[RAW21.index(c) for c in META4],
                memory_saving_mode=True, n_preprocessing_jobs=1)


def create_model(params: dict, icl_bf16: bool = False, chunk_cells=None, col_chunk=None):
    import torch
    from tabpfn import TabPFNClassifier
    if not icl_bf16 and chunk_cells is None and col_chunk is None:
        return TabPFNClassifier(**params)
    from tabpfn.base import ModelSpecs, initialize_tabpfn_model
    models, configs, _, inference = initialize_tabpfn_model(
        params['model_path'], 'classifier', n_estimators_override=1,
        devices=[torch.device('cpu')])
    if icl_bf16:
        models[0].enable_icl_bf16()
    if chunk_cells is not None:
        assert chunk_cells > 0
        models[0].inference_chunk_cells = chunk_cells
    if col_chunk is not None:
        assert col_chunk > 0
        models[0].inference_col_chunk_size = col_chunk
    spec = ModelSpecs(models[0], configs[0], inference)
    return TabPFNClassifier(**dict(params, model_path=spec))


def serializable(value):
    import dataclasses
    from enum import Enum
    if dataclasses.is_dataclass(value):
        return serializable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, 'features'):
        return serializable(value.features)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unserializable official configuration: {type(value)}')


def config_hash(config):
    return hashlib.sha256(json.dumps(serializable(config), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def prepare(clf, x, y):
    """Certified port: official fit up to engine construction, members captured."""
    from unittest.mock import patch as _patch
    from tabpfn.classifier import create_inference_engine
    captured = {}

    def intercept(**kwargs):
        prep = kwargs['ensemble_preprocessor']
        assert prep.subsample_row_indices is None, 'Require every frozen FIT row'
        members = prep.fit_transform_ensemble_members(X_train=kwargs['X_train'],
                                                      y_train=kwargs['y_train'])
        captured.update(kwargs=kwargs, members=members)
        return None

    with _patch('tabpfn.classifier.create_inference_engine', intercept):
        clf.fit(x, y)
    members, kwargs = captured['members'], captured['kwargs']
    assert len(members) == clf.n_estimators_ == clf.n_estimators
    assert clf.tuning_config is None and clf.downsample_correction_weights_ is None
    prep = kwargs['ensemble_preprocessor']
    manifests = []
    for i, member in enumerate(members):
        a, b = np.asarray(member.X_train), np.asarray(member.y_train)
        assert len(a) == len(b) == len(x)
        manifests.append({
            'index': i, 'config_sha256': config_hash(member.config),
            'pipeline_seed': int(prep.pipeline_seeds[i]),
            'prepared_x_sha256': arr_sha256(a),
            'prepared_shape': list(a.shape)})
    metadata = {
        'members': manifests,
        'softmax_temperature': float(clf.softmax_temperature_),
        'average_before_softmax': clf.average_before_softmax,
        'balance_probabilities': clf.balance_probabilities,
        'row_subsampling': None,
        'n_estimators_resolved': clf.n_estimators_,
        'checkpoint_n_estimators': serializable(clf.inference_config_.N_ESTIMATORS)}
    return create_inference_engine, kwargs, members, metadata


class OneMemberPreprocessor:
    def __init__(self, member, parent):
        self.member, self.parent = member, parent

    def fit_transform_ensemble_members(self, X_train, y_train):
        assert self.member.X_train is not None, 'Member cache has already been consumed'
        return [self.member]

    def any_estimator_uses_gpu_svd(self):
        return self.parent.any_estimator_uses_gpu_svd()


def release(clf):
    import torch
    clf.executor_ = None
    gc.collect()
    torch.cuda.empty_cache()


def probabilities(clf, logits):
    from tabpfn.constants import PROBABILITY_EPSILON_ROUND_ZERO, SKLEARN_16_DECIMAL_PRECISION
    p = clf.logits_to_probabilities(logits).float().detach().cpu().numpy()
    p = clf._maybe_reweight_probas(probas=p)
    if clf.inference_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
        p = np.around(p, decimals=SKLEARN_16_DECIMAL_PRECISION)
        p = np.where(p < PROBABILITY_EPSILON_ROUND_ZERO, 0., p)
    return p / p.sum(axis=1, keepdims=True)


def is_oom(e: BaseException) -> bool:
    import torch
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    s = str(e).lower()
    return 'out of memory' in s


def build_member_engine(clf, prepared, member_index):
    factory, kwargs, members, metadata = prepared
    m = members[member_index]
    clf.executor_ = factory(**{**kwargs, 'ensemble_preprocessor':
                               OneMemberPreprocessor(m, kwargs['ensemble_preprocessor'])})
    return clf.executor_


def predict_raw(clf, Xq, batch, deadline=None, label=''):
    """Predict raw (class-unpermuted) logits in row batches."""
    import torch
    chunks = []
    i = 0
    while i < len(Xq):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f'{label}: deadline reached at {i}/{len(Xq)}')
        j = min(i + batch, len(Xq))
        try:
            out = clf.predict_raw_logits(Xq[i:j])
        except Exception as e:  # noqa: BLE001
            if is_oom(e) and batch > 1024:
                batch //= 2
                gc.collect()
                torch.cuda.empty_cache()
                log(f'  {label}: OOM -> batch {batch}')
                continue
            raise
        if isinstance(out, torch.Tensor):
            out = out.detach().float().cpu().numpy()
        assert out.shape == (1, j - i, 2), out.shape
        chunks.append(out.astype(np.float32))
        i = j
    return np.concatenate(chunks, axis=1)


def unit_paths(fold, member):
    tag = f'f{fold}_m{member}'
    return (UNITS / f'{tag}_val_logits.npy',
            UNITS / f'{tag}_test_logits.npy',
            UNITS / f'{tag}_meta.json')


def unit_done(fold, member):
    _, _, m = unit_paths(fold, member)
    return m.exists()


# ------------------------------------------------------------------ unit runner
def run_unit(fold: int, member: int, gpu: int, deadline_ts: float, ckpt: Path):
    import torch
    torch.cuda.set_device(gpu)
    X, y, folds, XT, ids, idt = ensure_data_npz()
    fit_rows = np.flatnonzero(folds != fold)
    val_rows = np.flatnonzero(folds == fold)
    Xf, yf = X[fit_rows], y[fit_rows]
    Xv = X[val_rows]
    apply_lean_patch()
    bf16 = bool(torch.cuda.is_bf16_supported())
    variants = [
        dict(memory_saving_mode=True, keep_cache_on_device=True),
        dict(memory_saving_mode='auto', keep_cache_on_device=True),
        dict(memory_saving_mode=True, keep_cache_on_device=False),
    ]
    t_start = time.time()
    last_err = None
    for vi, variant in enumerate(variants):
        clf = None
        try:
            params = make_params(ckpt)
            params.update(variant)
            clf = create_model(params, icl_bf16=bf16, chunk_cells=CHUNK_CELLS,
                               col_chunk=COL_CHUNK)
            t0 = time.time()
            prepared = prepare(clf, Xf, yf)
            t_prep = time.time() - t0
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            build_member_engine(clf, prepared, member)
            t_fit = time.time() - t0
            peak_fit = torch.cuda.max_memory_allocated()

            t0 = time.time()
            val_logits = predict_raw(clf, Xv, BATCH, label=f'f{fold}m{member} val')
            t_val = time.time() - t0
            per_row = t_val / max(1, len(Xv))

            # durable: persist the val part before attempting the test part
            UNITS.mkdir(parents=True, exist_ok=True)
            vp, tp, mp_ = unit_paths(fold, member)
            np.save(vp, val_logits[0])
            meta = dict(
                fold=fold, member=member, gpu=gpu, variant=vi, variant_kwargs=variant,
                device_name=torch.cuda.get_device_name(gpu), bf16=bf16,
                chunk_cells=CHUNK_CELLS, col_chunk=COL_CHUNK,
                lean_chunk=LEAN_CHUNK, batch_last=BATCH,
                t_prep=t_prep, t_fit=t_fit, t_val=t_val, t_test=None,
                val_rows=int(len(Xv)), test_rows=int(len(XT)),
                peak_fit_bytes=int(peak_fit),
                softmax_temperature=float(clf.softmax_temperature_),
                average_before_softmax=bool(clf.average_before_softmax),
                balance_probabilities=bool(clf.balance_probabilities),
                member_manifest=prepared[3]['members'][member],
                val_logits_sha256=arr_sha256(val_logits[0]),
                test_logits_sha256=None,
                total_seconds=time.time() - t_start,
                library=__import__('tabpfn').__version__)
            mp_.write_text(json.dumps(meta, indent=1), encoding='utf-8')

            test_logits = None
            t_test = None
            est_test = per_row * len(XT) * 1.25 + 240
            if time.time() + est_test < deadline_ts:
                t0 = time.time()
                try:
                    test_logits = predict_raw(clf, XT, BATCH, deadline=deadline_ts,
                                              label=f'f{fold}m{member} test')
                    t_test = time.time() - t0
                except TimeoutError:
                    log(f'f{fold}m{member}: test deadline reached -> val kept, test skipped')
                    test_logits = None
                    t_test = None
            else:
                log(f'f{fold}m{member}: test prediction skipped (deadline policy)')

            if test_logits is not None:
                np.save(tp, test_logits[0])
                meta['t_test'] = t_test
                meta['test_logits_sha256'] = arr_sha256(test_logits[0])
                meta['total_seconds'] = time.time() - t_start
                mp_.write_text(json.dumps(meta, indent=1), encoding='utf-8')

            log(f'UNIT DONE f{fold} m{member}: prep={t_prep:.0f}s fit={t_fit:.0f}s '
                f'val={t_val:.0f}s test={t_test if t_test is None else round(t_test)}s '
                f'peak={peak_fit / 2 ** 30:.2f}GiB variant={vi}')
            release(clf)
            return 0
        except Exception as e:  # noqa: BLE001
            last_err = e
            if is_oom(e):
                log(f'UNIT f{fold} m{member}: OOM in variant {vi} ({variant})')
                try:
                    if clf is not None:
                        release(clf)
                except Exception:  # noqa: BLE001
                    pass
                continue
            log(f'UNIT f{fold} m{member}: fatal {type(e).__name__}: {e}')
            raise
    log(f'UNIT f{fold} m{member}: all variants failed: {last_err}')
    return 2


# ------------------------------------------------------------------ probe
def probe(ckpt: Path):
    import torch
    X, y, folds, XT, ids, idt = ensure_data_npz()
    summary = {'mode': 'probe', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'gpu_count': torch.cuda.device_count(),
               'gpu_name': torch.cuda.get_device_name(0),
               'bf16_supported': bool(torch.cuda.is_bf16_supported())}
    bf16 = bool(torch.cuda.is_bf16_supported())

    # --- A. small-context equivalence: stock vs ported (no patch) vs ported (patch)
    Xc, yc = X[:30000], y[:30000]
    Xq = X[30000:31000]
    params1 = make_params(ckpt)
    params1['n_estimators'] = 1

    clf_s = create_model(params1, icl_bf16=bf16, chunk_cells=CHUNK_CELLS, col_chunk=COL_CHUNK)
    clf_s.fit(Xc, yc)
    p_stock = clf_s.predict_proba(Xq)[:, 1]
    release(clf_s)
    log('A1 stock ok')

    restore_stock_forward()
    clf_a = create_model(params1, icl_bf16=bf16, chunk_cells=CHUNK_CELLS, col_chunk=COL_CHUNK)
    prep_a = prepare(clf_a, Xc, yc)
    build_member_engine(clf_a, prep_a, 0)
    lg_a = predict_raw(clf_a, Xq, 4096, label='A2')
    p_a = probabilities(clf_a, lg_a)[0][:, 1]
    release(clf_a)
    log('A2 ported-no-patch ok')

    apply_lean_patch()
    clf_b = create_model(params1, icl_bf16=bf16, chunk_cells=CHUNK_CELLS, col_chunk=COL_CHUNK)
    prep_b = prepare(clf_b, Xc, yc)
    build_member_engine(clf_b, prep_b, 0)
    lg_b = predict_raw(clf_b, Xq, 4096, label='A3')
    p_b = probabilities(clf_b, lg_b)[0][:, 1]
    release(clf_b)
    log('A3 ported-patched ok')

    d_stock_a = float(np.max(np.abs(p_stock - p_a)))
    d_a_b = float(np.max(np.abs(p_a - p_b)))
    summary['equiv'] = {'stock_vs_port': d_stock_a, 'port_vs_lean': d_a_b,
                        'gate': 2e-6, 'pass': bool(d_stock_a <= 2e-6 and d_a_b <= 2e-6)}
    log('A equivalence: stock-vs-port', d_stock_a, 'port-vs-lean', d_a_b)
    if not summary['equiv']['pass']:
        (WORK / 'probe_summary.json').write_text(json.dumps(summary, indent=1),
                                                 encoding='utf-8')
        raise SystemExit('equivalence gate failed')

    # --- B. full-context member-0 feasibility (fold 0 against 4/5 context)
    fit_rows = np.flatnonzero(folds != 0)
    val_rows = np.flatnonzero(folds == 0)
    Xf, yf = X[fit_rows], y[fit_rows]
    n_pred = 8192
    Xv = X[val_rows][:n_pred]
    torch.cuda.reset_peak_memory_stats()
    params = make_params(ckpt)
    clf = create_model(params, icl_bf16=bf16, chunk_cells=CHUNK_CELLS, col_chunk=COL_CHUNK)
    t0 = time.time()
    prepared = prepare(clf, Xf, yf)
    t_prep = time.time() - t0
    n_members = len(prepared[2])
    log(f'B prepare on {len(Xf)} rows: {t_prep:.0f}s, members={n_members}')
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    build_member_engine(clf, prepared, 0)
    t_fit = time.time() - t0
    peak_fit = torch.cuda.max_memory_allocated()
    t0 = time.time()
    lg = predict_raw(clf, Xv, BATCH, label='B val')
    t_val = time.time() - t0
    peak_all = torch.cuda.max_memory_allocated()
    release(clf)
    per_row = t_val / n_pred
    est_unit = t_fit + per_row * (139927 + 299844)
    summary['full_context'] = {
        'fit_rows': int(len(Xf)), 'prepare_seconds': t_prep, 'n_members': n_members,
        'member0_fit_seconds': t_fit, 'predict_seconds_8192': t_val,
        'per_row_seconds': per_row, 'peak_fit_bytes': int(peak_fit),
        'peak_all_bytes': int(peak_all),
        'est_unit_seconds_fit_val_test': est_unit,
        'est_40_units_gpu_hours': est_unit * 40 / 3600,
        'est_wall_hours_2gpu': est_unit * 40 / 3600 / 2,
    }
    log('B full-context:',
        json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in summary['full_context'].items()
                    if k != 'peak_fit_bytes'}, indent=1))
    (WORK / 'probe_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    log('PROBE COMPLETE')
    return 0


# ------------------------------------------------------------------ assembly
def assemble():
    import torch  # noqa: F401
    from sklearn.metrics import roc_auc_score
    X, y, folds, XT, ids, idt = ensure_data_npz()
    present = {}
    temps = {}
    for f in range(5):
        for m in range(8):
            _, _, mp_ = unit_paths(f, m)
            if mp_.exists():
                meta = json.loads(mp_.read_text(encoding='utf-8'))
                present.setdefault(f, {})[m] = meta
                temps[m] = (round(float(meta['softmax_temperature']), 12),
                            bool(meta['average_before_softmax']),
                            bool(meta['balance_probabilities']))
    temps_set = set(temps.values())
    assert len(temps_set) <= 1, f'aggregation scalars differ across units: {temps_set}'

    ckpt = download_checkpoint()
    agg_params = make_params(ckpt)
    agg_params['n_estimators'] = 1
    agg = create_model(agg_params, icl_bf16=False, chunk_cells=CHUNK_CELLS, col_chunk=COL_CHUNK)
    prepare(agg, X[:2000], y[:2000])  # tiny fit: official post-processing attributes
    release(agg)

    oof = np.full(len(y), np.nan, dtype=np.float32)
    test_stack = []
    fold_info = {}
    for f in range(5):
        ms = sorted(present.get(f, {}).keys())
        if not ms:
            continue
        vals = [np.load(unit_paths(f, m)[0]) for m in ms]
        stacked = np.stack(vals, 0)
        p = probabilities(agg, stacked)[0][:, 1]
        rows = np.flatnonzero(folds == f)
        oof[rows] = p
        tst = [np.load(unit_paths(f, m)[1]) for m in ms if unit_paths(f, m)[1].exists()]
        if tst:
            test_stack.append(probabilities(agg, np.stack(tst, 0))[0][:, 1])
        fold_info[f] = {'members': ms, 'n_members': len(ms),
                        'oof_auc_fold': float(roc_auc_score(y[rows], p)),
                        'n_test_members': len(tst)}
        log(f'fold {f}: members={len(ms)} auc={fold_info[f]["oof_auc_fold"]:.9f} '
            f'test_members={len(tst)}')
    result = {'fold_info': fold_info}
    if not np.isnan(oof).any():
        result['oof_auc_full'] = float(roc_auc_score(y, oof))
        log('FULL OOF AUC:', result['oof_auc_full'])
    np.save(WORK / 'raw8_oof.npy', oof)
    if test_stack:
        pt = np.mean(np.stack(test_stack, 0), axis=0).astype(np.float32)
        np.save(WORK / 'raw8_test.npy', pt)
        result['test_folds'] = len(test_stack)
    (WORK / 'raw8_result.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    log('ASSEMBLY COMPLETE', json.dumps(result)[:500])
    return 0


# ------------------------------------------------------------------ production
def estimate_unit_seconds():
    got = []
    for f in range(5):
        for m in range(8):
            _, _, mp_ = unit_paths(f, m)
            if mp_.exists():
                meta = json.loads(mp_.read_text(encoding='utf-8'))
                got.append(float(meta['total_seconds']))
    if not got:
        return 2400.0
    return float(np.median(got)) * 1.15


def production(ckpt: Path, limit_h: float):
    X, y, folds, XT, ids, idt = ensure_data_npz()
    UNITS.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline = start + limit_h * 3600
    queue = [(f, m) for f in range(5) for m in range(8)]
    pending = [u for u in queue if not unit_done(*u)]
    retried = set()
    log(f'production: {len(pending)}/{len(queue)} units pending, deadline in {limit_h:.1f} h')
    procs = {}
    while True:
        now = time.time()
        est = estimate_unit_seconds()
        for g in (0, 1):
            if g in procs or not pending:
                continue
            if now + est + MIN_TAIL_S > deadline:
                continue
            f, m = pending.pop(0)
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(g)
            cmd = [sys.executable, os.path.abspath(__file__), '--unit', str(f), str(m),
                   '--gpu', '0', '--deadline', f'{deadline:.0f}', '--ckpt', str(ckpt)]
            log(f'start unit f{f} m{m} on gpu {g} (est {est:.0f}s)')
            procs[g] = (f, m, subprocess.Popen(cmd, env=env))
        for g in list(procs):
            f, m, p = procs[g]
            rc = p.poll()
            if rc is not None:
                log(f'unit f{f} m{m} on gpu {g} finished rc={rc}')
                del procs[g]
                if not unit_done(f, m) and (f, m) not in retried:
                    retried.add((f, m))
                    pending.append((f, m))
                    log(f'requeue f{f} m{m}')
        if not procs and (not pending or time.time() + estimate_unit_seconds() + MIN_TAIL_S > deadline):
            break
        time.sleep(10)
    log('workers done; assembling')
    return assemble()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', nargs=2, type=int, metavar=('FOLD', 'MEMBER'))
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--deadline', type=float, default=0.0)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--limit-h', type=float, default=LIMIT_H)
    ap.add_argument('--assemble', action='store_true')
    args = ap.parse_args()

    t_all = time.time()
    if args.unit is not None:
        ckpt = Path(args.ckpt)
        return run_unit(args.unit[0], args.unit[1], args.gpu,
                        args.deadline if args.deadline > 0 else time.time() + 9e9, ckpt)

    pip_install()
    gpu_report()
    ckpt = download_checkpoint()
    if args.assemble:
        return assemble()
    if PROBE:
        return probe(ckpt)
    rc = production(ckpt, args.limit_h)
    log(f'total wall {time.time() - t_all:.0f}s')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
