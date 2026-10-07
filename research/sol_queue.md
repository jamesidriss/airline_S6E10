# Independent research queue — 7 October 2026

Ordering uses private performance and methodological novelty. Upside ranges below
are subjective planning estimates, not measured gains or forecasts. No public
score is an optimization target. Execute the first eligible mechanism; do not
run a model catalogue.

| Rank | Remaining mechanism | Novelty and expected standalone upside | Private hedge potential | Risk; GPU hours | Overlap and falsification |
|---|---|---|---|---|---|
| 1 | Own full-context TabPFN 3.5, raw vs Flight Distance categorical + fixed coarse route | High: pretrained in-context inference; route ablation could gain 0.0001–0.0006 over raw | High if competitive and different rankings survive full-context test inference | Memory/timing medium; probe first, provisional 1–3 h for five folds | Not a closed tree/seed/retrieval branch. Reject promotion on weak AUC plus no positive fixed equal-member ensemble contribution. No imported predictions. |
| 2 | Satisfaction + masked-rating multitask encoder, matched reconstruction weight 0 vs small/moderate | High objective novelty; 0–0.0002 vs the matched network | Medium/high, but only if standalone gap to A becomes <=0.0003 | Neural optimization medium/high; 0.5–2 h | Conditional on replicated SSL/distribution evidence. SSL B1 mean three-fold standalone gain +0.0001217, but actual v5 replacement negative in all three; SSL+aux fold0 negative. Plain soft-target and seed branches remain closed. |
| 3 | Nested disagreement specialist on high auxiliary surprise | Medium/high: tests concentration of known A pair errors; 0–0.0001 ensemble | Medium, conditional on error concentration and stable segment performance | Nested selection high; 0.5–2 h | First a label-free segmented diagnostic. Train only if errors concentrate materially. Fit every label-dependent correction inside outer FIT; existing local reliability result is a warning, not support. |
| 4 | Field-aware factorization of raw answers and auxiliary residuals | Medium/high pair-interaction inductive bias; 0–0.0002 standalone | Medium; pointwise sparse interaction model differs from trees | Capacity and optimization medium; 0.5–1.5 h | Not a generic MLP rerun. Compare a fixed second-order interaction arm to a matched linear control. Lower EV until auxiliary block replicates. |
| 5 | ModernNCA learned metric with sampled candidate objective | Medium objective novelty; 0–0.0002 standalone | Medium/high if it closes the standalone gap | Retrieval scale and implementation high; 2–4 h | Substantial overlap with rejected TabR/kNN; only new evidence justifies execution. No default rerun. |

The measured branches already in progress are not additional queue entries:
distribution A1 failed fold1; A2 failed fold2 with −0.000201215 standalone and
three-fold mean −0.000001304. A3/A4 lose standalone. The frozen SSL encoder was
evaluated on three folds, with adverse actual v5 slot deltas, and its XT check
lost −0.000044376. Multitask and field-factorization conditions are unmet; the
surprise diagnostic does not justify a specialist. Full-context foundation
inference is still the only eligible new mechanism. Neural replay is mandatory
repair of withdrawn outer-selected CV, not a new model/seed search.

Own full-context discovery is now positive, not a resource-only result: raw f0
AUC0.960972296, route f0 AUC0.961132446 (+0.000160150). Both fixed 1/60
legacy-stack additions gain about0.000020. The predeclared 50/50 route/strict
auxiliary10 combination reaches0.961727180, versus0.961414204 for its strict
classical component. This single-fold result promotes exact replication; it
does not admit a finalist. Route f1 and the current-code auxiliary counterpart
replay take priority. Full699635-row test-context feasibility is a separate
resource gate before committing to a larger finalist run.

Primary sources:

- [TabPFN repository](https://github.com/PriorLabs/TabPFN) and
  [TabPFN 3.5 checkpoint/license](https://huggingface.co/Prior-Labs/tabpfn_3_5).
  Version 9.1.0 supports up to one million context rows; actual local feasibility
  is measured, not inferred from the headline limit. Provenance is recorded in
  `research/raw/sol_tabpfn_provenance.json`.
- [MET](https://arxiv.org/abs/2206.08564) and
  [VIME](https://papers.nips.cc/paper_files/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)
  motivate reconstruction, not a guarantee of satisfaction improvement.
- [TabM](https://github.com/yandex-research/tabm) already supplies the campaign's
  stronger neural baseline. Generic FT-Transformer, TabTransformer, ResNet,
  SAINT, Trompt, NODE and deep ensemble sweeps have weaker campaign-specific
  evidence than the queue above.
- [TALENT authors' toolkit](https://github.com/LAMDA-Tabular/TALENT) contains
  ModernNCA and those architecture families. The learned neighbor objective is
  distinct, but its inference mechanism overlaps TabR.
- [Field-matrixed Factorization Machines](https://arxiv.org/abs/2102.12994)
  supplies a categorical interaction mechanism; any proposed auxiliary extension
  is our hypothesis.
- [InterpretML](https://github.com/interpretml/interpret) supplies EBM/GAM with
  interactions. Low-order additive capacity seems unlikely to close the current
  ensemble gap without a new conditional mechanism; no broad sweep is planned.

## Public source audit

Source downloads live under ignored `artifacts/sol_public/`; code was inspected,
not executed, and no competition prediction CSV was adopted.

| Source | Classification | Decision |
|---|---|---|
| `cdeotte/stacking-tfms-with-public-oof` | C imported prediction library; D base nesting unclear | Reject as reproducible CV evidence. Useful list of methods only. |
| `matterhorn3838/s6e10-aiming-for-space-0-96170` | C imported predictions; D provenance/nesting insufficient | Do not adopt predictions or optimize their published score. |
| `goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160` | F own full-context foundation-model mechanism; tree/stack validation invalid for our comparison | Reimplement the raw/route mechanism on immutable primary folds. Its tree early stopping uses outer eval targets; meta folds differ from base folds without full nesting. Reject those gains as evidence. |

The source's TabPFN representation declares Flight Distance a route category and
adds a fixed `Flight Distance // 10` category. This is a label-free representation,
not an ID feature. Our raw/route paired ablation freezes seed, estimator count,
checkpoint, population and prediction geometry before seeing our fold results.

Rules snapshot `research/raw/sol_rules_20261007.json`, section 2.6, permits public
external models. The checkpoint's license section 1.c explicitly includes data
science competitions. Own predictions use no external competition targets.
