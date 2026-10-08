# Phase 16 noise evidence and ranking headroom

The available evidence does not establish an irreducible 3.8% flip rate or a competition AUC ceiling. OOF above 0.9621 remains statistically plausible, but achieving it requires a measured improvement rather than label cleaning based on this proxy.

The [author's notebook](https://www.kaggle.com/code/dariushafshar/s6e10-the-gap-is-3-8-label-noise-not-features), retrieved through the Kaggle CLI, estimates an **equivalent corruption rate**. It fits a fixed LightGBM on an equal-sized competition sample and on original labels corrupted by one nested random-flip draw. Interpolating the AUC-loss curve yields about 3.77%, rounded to 3.8%. This is not a direct observation of which competition labels were flipped. Original-model disagreement, transfer error and residual model approximation can all contribute.

Sections 1–6 use fixed tree counts and ordinary OOF fitting rather than outer-label early stopping. The original-only model never trains on competition labels, although exact original/competition overlaps need exclusion. Later sections use imported public OOF libraries. Their logistic-regression validation holds out labels only at the meta layer; it is not fully nested at the base-model layer. Member leakage contracts were not audited by the author. Those stack scores and their public-score comparisons are ineligible as our local promotion evidence. No external prediction vectors were downloaded.

Let q be observed positive prevalence, e a symmetric independent flip probability, and A the clean-label AUC. Under those assumptions, latent prevalence is p=(q-e)/(1-2e), a=p(1-e)/q and b=pe/(1-q). Observed AUC is 0.5+(a-b)(A-0.5). Setting A=1 gives a conditional population ceiling. The assumptions require equal flip probabilities in both classes and flips independent of features and score given the latent class. The notebook demonstrates a simulation with those properties, not that they govern competition labels.

Our [reproduction](phase16_noise_reproduction_20261009.json) uses only the certified primary predictions and the existing original-only five-model teacher. Its original rows matching any competition covariate row were excluded. Its 600-tree, 63-leaf, seeds 0–4 recipe differs from the author's single model, so this reproduces the diagnostic mechanism rather than every published number. Original OOF and flip-grid artifacts were not previously saved; their fitting experiment was not repeated.

| Own measurement | Result |
|---|---:|
| Competition rows assigned below 0.001 by original-only teacher | 195,401 |
| Satisfied among those rows | 3.456%, Wilson 95% interval 3.376–3.538% |
| Competition rows assigned at least 0.999 | 210,332 |
| Not satisfied among those rows | 4.595%, Wilson 95% interval 4.506–4.685% |
| All competition rows contradicting those near-certain predictions | 2.347% |
| V6 observed-label AUC within the low-confidence-probability region | 0.591758, conditional interval 0.584931–0.598585 |
| V6 observed-label AUC within the high-confidence-probability region | 0.643126, conditional interval 0.637575–0.648677 |
| V6 AUC loss after flattening scores in both near-certain regions | 0.003980 |
| V6 erroneous pairs involving a contradicting row | 63.34% |

The disagreement rates differ across the two extreme regions, and competition models rank observed labels above chance within each region. These findings leave room for structured generator effects and model approximation error. They do not identify latent clean labels: source-domain confidence can be wrong on synthetic rows. Original versus synthetic passengers also differ, so density reweighting alone cannot prove that all performance loss is random label corruption.

At our measured positive prevalence 0.44357272, a hypothetical perfect clean-label ranker has ceilings of 0.969201 for 3.0% flips, 0.964063 for 3.5%, 0.961288 for 3.77%, 0.960979 for exactly 3.8%, and 0.958923 for 4.0%. Rounding the flip rate is material at this scale. The flip rate whose **assumed** perfect-ranker ceiling equals 0.9621 is 3.690974%; this inversion is not an estimate of actual noise. The notebook's sampling-only uncertainty also omits refitting, source transfer, noise identification and model misspecification.

Wilson and DeLong intervals here condition on saved models and regions, with an independent-row approximation. They omit OOF training dependence and refitting uncertainty. The author uses one flip draw; its interpolation and two AUC standard errors do not settle irreducibility or a Bayes frontier. A finite observed OOF score can also differ from a population bound.

V6 needs another 0.000334386 AUC to reach 0.9621. That is about a 0.875% net reduction in its remaining pair-ranking error. Useful improvements would distinguish difficult observed-label pairs within source-confident or source-uncertain regions, rather than indiscriminately deleting disagreement rows. Internal estimator configurations and fixed aggregation geometries are therefore still justified for bounded testing. The stretch target is plausible and unproved; private performance remains the decision criterion.
