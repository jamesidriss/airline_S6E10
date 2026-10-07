# Conditional SOL-C protocol

Frozen before any multitask satisfaction CV result. Execute only after A2
replicates on primary fold2; folds0/1 already gain +0.000106176/+0.000091126
standalone. This tests shared representation regularization, not soft labels.

- Covariate encoder: the existing label-free `sol_ssl12` checkpoint, width256,
  latent32, categorical embeddings8. Fine-tune only on inner training rows.
- Satisfaction head: concatenate the unmasked latent with the full285 view,
  then Linear256/SiLU/Dropout0.05/Linear128/SiLU/Linear1.
- Matched weights: lambda **0, 0.05, 0.2**, seed1201, AdamW lr0.002,
  weight_decay0.0001, batch2048, cosine schedule, at most24 epochs.
- Auxiliary loss: the existing masked CE/SmoothL1 objective, masking0.25;
  no satisfaction label enters the auxiliary targets. Both twins are masked.
- Strict TE assembled once from outer FIT, inner seed=fold; normalization
  fits inner training rows. Inner10 percent ES uses seed1. Select best epoch
  by inner ES AUC, patience6. Evaluate outer labels once after selection.
- Run a60-step timing probe first; no outer AUC for a timing probe.
- Treatment admission for fold1: at least+0.00005 over lambda0 AND no adverse
  fixed equal-member delta to the legacy-v5 diagnostic reference. Freeze one
  winning weight; no further architecture/lambda sweep.
- A finalist requires full OOF, clean comparison portfolio, shadow/private
  simulation and corresponding inference. Legacy v5 is not a clean finalist.
