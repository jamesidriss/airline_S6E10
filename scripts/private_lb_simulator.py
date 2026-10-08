"""OOF robustness diagnostics, NOT probabilities of future competition ranks.

Fixed candidate-independent splits. The default pseudo test population matches
the real test size, avoiding an artificially precise 699635-row private board.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, TARGET, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.validation.compare import logit, spearman
from src.validation.folds import get_scheme
from src.validation.private_sim import RankedAUC, summarize
from src.submission import store
from scripts.audit_sol_state import reconstruct_v5

SEED = 20261010


def choose_stratified(rng, y, pool, size, weights=None):
    chosen = []
    npos = int(round(size * y[pool].mean()))
    for label, count in ((1, npos), (0, size - npos)):
        idx = pool[y[pool] == label]
        p = None if weights is None else weights[idx] / weights[idx].sum()
        chosen.append(rng.choice(idx, count, replace=False, p=p))
    return np.concatenate(chosen)


def make_partition(y, folds, seed, mode, population, stress=None, fold=None):
    rng = np.random.default_rng(seed)
    pool = choose_stratified(rng, y, np.arange(len(y)), population)
    public_n = int(round(.2 * population))
    if mode == "fold_aware":
        # One actual validation fold supplies public, the other four private.
        # Sample the two populations separately for exact 20/80 sizes.
        pu = choose_stratified(rng, y, np.flatnonzero(folds == fold), public_n)
        pr = choose_stratified(rng, y, np.flatnonzero(folds != fold), population - public_n)
    else:
        weights = None if stress is None else np.where(stress, 4.0, 1.0)
        pu = choose_stratified(rng, y, pool, public_n, weights)
        pr = np.setdiff1d(pool, pu, assume_unique=True)
    mask = np.zeros(len(y), dtype="uint8")
    mask[pu], mask[pr] = 1, 2
    assert len(pu) == public_n and len(pr) == population - public_n
    assert not np.intersect1d(pu, pr).size
    return mask


def segments(tr, champion):
    out = {}
    for col in ("Class", "Type of Travel", "Customer Type"):
        for level in sorted(tr[col].unique()):
            out[f"{col}={level}"] = (tr[col] == level).to_numpy()
    delay = tr["Departure Delay in Minutes"].fillna(0).to_numpy()
    out["departure_delay_zero"] = delay == 0
    out["departure_delay_60plus"] = delay >= 60
    distance = tr["Flight Distance"].to_numpy()
    out["distance_lower_quartile"] = distance <= np.quantile(distance, .25)
    out["distance_upper_quartile"] = distance >= np.quantile(distance, .75)
    from scripts.run_phase13 import RATINGS
    zeros = (tr[RATINGS] == 0).sum(axis=1).to_numpy()
    out["survey_any_zero"] = zeros > 0
    out["survey_many_zeros"] = zeros >= 3
    # Fixed reference confidence, not candidate-specific partitions.
    out["champion_low_confidence"] = np.abs(champion) <= np.quantile(np.abs(champion), .25)
    out["champion_high_confidence"] = np.abs(champion) >= np.quantile(np.abs(champion), .75)
    return {k: v for k, v in out.items() if .01 <= v.mean() <= .99}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--stress-repeats", type=int, default=30)
    ap.add_argument("--population", type=int, default=299844)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default="reports/sol_private_sim.json")
    ap.add_argument("--candidate", action="append", default=[], help="name=full OOF .npy path; probabilities")
    ap.add_argument("--reference",default="v5_aux_cross_OOF",help="Fixed comparison candidate; does not change partition definitions")
    ap.add_argument("--portfolio-contract",help="Verified full-primary route/auxiliary10 report for added candidates")
    args = ap.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Preserve the existing simulation report; use a new output name')
    tr, te = load_cached_parquet()
    y = tr[TARGET].to_numpy(dtype="int8")
    folds = get_scheme("primary", y, tr["id"]).folds
    base, champion, _ = reconstruct_v5(y, folds)
    cand = {"v3_final": base, "v4_fulldata_CV_PROXY": base, "v5_aux_cross_OOF": champion}
    # Strongest available distinct family is included as an ineligible diagnostic.
    tm = ["z5_tabm_e25", "z5_tabm_e25_s2"]
    cand["tabm_family_INELIGIBLE"] = np.mean([logit(store.load_oof(e)) for e in tm], axis=0)
    portfolio=None
    if args.portfolio_contract:
        portfolio=json.loads(Path(args.portfolio_contract).read_text(encoding='utf-8'))
        assert portfolio['scope']=='full primary OOF' and portfolio['scheme']=='primary'
        assert portfolio['fold_sha256']==arr_sha256(folds)
        assert portfolio['ordered_train_ids_sha256']==arr_sha256(tr.id.to_numpy())
        assert portfolio['data_sha256']=={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
        assert {r['fold'] for r in portfolio['folds']}==set(range(5)) and len(portfolio['folds'])==5
    for item in args.candidate:
        name, path = item.split("=", 1)
        assert name not in cand, 'Candidate name would replace a banked diagnostic'
        pred=np.load(path)
        assert pred.shape==(len(y),) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
        if portfolio:
            keys={'candidate_oof':'oof_sha256','aux10_oof':'strict_aux10_oof_sha256','route_oof':'route_oof_sha256'}
            assert Path(path).stem in keys and arr_sha256(pred)==portfolio[keys[Path(path).stem]]
            assert np.array_equal(np.load(Path(path).parent/'train_ids.npy'),tr.id.to_numpy())
        cand[name] = logit(pred)
    assert args.reference in cand
    if args.reference!='v5_aux_cross_OOF':
        assert portfolio is not None, 'A clean-reference simulation requires a verified full-OOF contract'
    aucs = {name: RankedAUC(y, p) for name, p in cand.items()}
    full = {name: a.auc(np.ones(len(y))) for name, a in aucs.items()}
    stress = segments(tr, champion)
    definitions, draws = [], {name: [] for name in cand}
    plans = [("random_stratified", None, j) for j in range(args.repeats)]
    plans += [("fold_aware", None, j) for j in range(args.repeats)]
    plans += [("segment_stressed", key, j) for key in sorted(stress) for j in range(args.stress_repeats)]
    for run, (mode, key, j) in enumerate(plans):
        seed = args.seed + run * 104729
        mask = make_partition(y, folds, seed, mode, args.population,
                              stress.get(key), fold=j % 5)
        defs = {"run": run, "mode": mode, "segment": key, "seed": seed,
                "public_fold": j % 5 if mode == "fold_aware" else None,
                "mask_sha256": arr_sha256(mask), "population": args.population,
                "public_rows": int((mask == 1).sum()), "private_rows": int((mask == 2).sum())}
        if key:
            defs.update(public_segment_share=float(stress[key][mask == 1].mean()),
                        private_segment_share=float(stress[key][mask == 2].mean()))
        definitions.append(defs)
        ref = aucs[args.reference]
        pb, pr = ref.auc(mask == 1), ref.auc(mask == 2)
        for name, scorer in aucs.items():
            draws[name].append({"mode": mode, "segment": key,
                                "public_delta": scorer.auc(mask == 1) - pb,
                                "private_delta": scorer.auc(mask == 2) - pr})
        if (run + 1) % 50 == 0:
            print(f"simulations {run + 1}/{len(plans)}", flush=True)
    report = {"git": git_commit(), "seed": args.seed, "population": args.population,
              "reference":args.reference,"partition_confidence_reference":"fixed legacy v5; candidate independent",
              "portfolio_contract_sha256":file_sha256(args.portfolio_contract) if portfolio else None,
              "public_fraction": .2, "stress_odds_multiplier": 4,
              "fold_sha256": arr_sha256(folds), "ids_sha256": arr_sha256(tr["id"].to_numpy()),
              "source_sha256": {p: file_sha256(p) for p in ("scripts/private_lb_simulator.py", "src/validation/private_sim.py")},
              "oof_prediction_sha256": {name: arr_sha256(p) for name, p in cand.items()},
              "split_definitions_sha256": hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest(),
              "disclaimer": "OOF resampling diagnostics, not probabilities of future competition outcomes. Validation-trained predictions are fixed; training uncertainty and unknown shift are not simulated. v4 uses v3 OOF as a proxy only. Legacy TE prior self-exclusion defect and v5 test-config mismatch are documented in sol_s0_audit.json.",
              "candidates": {}}
    ref_test = pd.read_csv("submissions/v5_aux_cross.csv")[TARGET].to_numpy()
    for name, records in draws.items():
        rec = {"oof_auc": full[name], "delta_vs_v5": full[name] - full["v5_aux_cross_OOF"],
               "delta_vs_reference":full[name]-full[args.reference],
               "oof_logit_corr_vs_reference":float(np.corrcoef(cand[name],cand[args.reference])[0,1]),
               "oof_spearman_vs_reference":spearman(cand[name],cand[args.reference]),
               "oof_logit_corr_vs_v5": float(np.corrcoef(cand[name], champion)[0, 1]),
               "oof_spearman_vs_v5": spearman(cand[name], champion), "modes": {}, "stress_segments": {}}
        for mode in ("random_stratified", "fold_aware", "segment_stressed"):
            chosen = [d for d in records if d["mode"] == mode]
            rec["modes"][mode] = summarize([d["public_delta"] for d in chosen],
                                               [d["private_delta"] for d in chosen], rec["delta_vs_reference"])
        for key in stress:
            chosen = [d for d in records if d["segment"] == key]
            rec["stress_segments"][key] = summarize([d["public_delta"] for d in chosen],
                                                    [d["private_delta"] for d in chosen], rec["delta_vs_reference"])
        if name in ("v3_final", "v4_fulldata_CV_PROXY", "v5_aux_cross_OOF"):
            actual = name.replace("_CV_PROXY", "").replace("_OOF", "")
            testp = pd.read_csv(f"submissions/{actual}.csv")[TARGET].to_numpy()
            rec["test_logit_corr_vs_v5"] = float(np.corrcoef(logit(testp), logit(ref_test))[0, 1])
            rec["test_spearman_vs_v5"] = spearman(testp, ref_test)
        report["candidates"][name] = rec
    dest = ARTIFACTS / "private_sim"
    dest.mkdir(exist_ok=True)
    draw_key=hashlib.sha256(json.dumps({'splits':report['split_definitions_sha256'],
        'predictions':report['oof_prediction_sha256'],'reference':args.reference},sort_keys=True).encode()).hexdigest()
    draw_path=dest/f'{draw_key}.json'
    payload={"definitions":definitions,"draws":draws}
    if draw_path.exists():
        assert json.loads(draw_path.read_text(encoding='utf-8'))==payload, 'Existing simulation artifact differs'
    else:
        save_json(payload,draw_path)
    report["split_and_draw_artifact"] = str(draw_path)
    save_json(report, args.output)
    print(json.dumps({n: r["modes"]["random_stratified"] for n, r in report["candidates"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
