# Frozen raw/route representation candidate

Declared before fold1 measurement. Reuse original raw fold0 only after its
source, data, ID, feature, checkpoint, parameter, library and AUC checks pass.
Raw fold1 uses the identical original21-column raw recipe, seed1201, one
estimator, checkpoint3.5, int8 on-device KV cache, autocast with bf16 ICL,
the verified Windows MQA/query reuse and in-place GELU implementation,
1024 query rows,262144 chunk cells,one column per chunk. Resource limits stay
unchanged. The successful original full-fold cost1030s is the timing estimate.

One portfolio only: route0.375/raw0.125/clean auxiliary10 collective0.5,
in logit space. Auxiliary roles retain equal weights. No weight grid,
seed choice, new checkpoint, gradient training or699635-row GPU context.

Stage1 discovery and stage2 replication retain the declared gates. Fold2
requires replicated positive portfolio gains. Complete folds3/4 only when all
three portfolio gains are positive and their mean is at least0.000015.
Full admission requires mean paired gain>=0.000015,>=2.5 paired fold SE,
positive in>=4/5 folds, and a competitive pooled gain. Prefer>=0.00003.

Before submission, confirm the exact frozen candidate on **shadow fold2**.
This fold has no existing route/auxiliary portfolio measurement; train all
base contexts with the same outer-FIT exclusion. The previously measured
shadow0/1 do not become fresh weight-selection holdouts. This is one shadow
fold, not full shadow OOF. Require positive candidate gain versus frozen v6
and at least0.00001 to justify a performance submission. No gate retuning.

Test policy: infer all299844 test rows from each of the five exact primary
FIT contexts for raw, seed1201. Equally average probabilities within raw.
Reuse the already certified five-context route probability average and the
ten corrected full-label auxiliary refits. Combine the three method vectors
with the frozen logit weights. Verify each raw context's FIT matrix and IDs
against its CV report, the raw train/test schema, source/checkpoint/library
hashes and all output IDs. Bind the candidate OOF, test, bootstrap,
private-simulation and independent confirmation reports before uploading.

Ordinary meta-only CV on fixed base OOF is not fully nested validation and
will not be used to justify a selected weight. Fixed-prediction bootstrap and
private simulations measure uncertainty conditional on these models; neither
reveals the hidden private labels or simulates refitting.
