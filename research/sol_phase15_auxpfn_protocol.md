# Conditional third hypothesis: route plus predicted covariate ratings

Declared before the raw/route five-fold decision. Execute only if that candidate
fails final promotion and the historical classical replay trigger stays unmet.
This is the existing third ranked hypothesis, not another search branch.

Mechanism: append exactly13 numeric expected-rating features to the frozen
22-column TabPFN route representation. The total is35 columns. Keep all route
categorical indices, category maps, checkpoint, seed1201, one estimator and
inference settings unchanged. These aux_* features predict covariates; they
contain no satisfaction or row-ID input. No additional feature combinations.

Reuse the hash-addressed probability caches proven by the clean auxiliary10
primary reports. Verify ordered FIT/apply IDs, raw21 matrix hashes, builder
source, immutable fold hash, seed1,250 rounds and three inner rating folds.
Verify complete cache probability hashes and simplex values. Recompute13
expectations and compare their hashes to the corresponding clean C1 report.
The FIT expectations are inner OOS; applied expectations use outer-FIT-only
models. Never build an auxiliary model on all rows before splitting.

First run one timing/memory probe:100000 label-agnostic sampled outer FIT rows,
seed1201, and the first1024 outer validation covariates. No AUC, pair comparison
or portfolio result is calculated. Retain original reserves:4 GiB before fit,
2 GiB before prediction,20 GiB disk before fit,45-minute total maximum,85percent
GPU allocation cap and unchanged periodic resource aborts. The probe does not
establish full-context feasibility. Resource failures are INVALID, not negative
scientific results; do not alter the checkpoint or memory policy to rescue them.

After a successful probe, discovery uses all primary fold0 FIT rows and its full
validation population. Matched standalone control is frozen route fold0; the
portfolio is0.125 new TabPFN +0.875 immutable v6 in logit space, as already
declared in the scope JSON. No weight optimization. Discovery requires at least
0.00005 standalone gain or0.00001 actual portfolio gain. Replication requires
positive portfolio gain on fold1 with the same recipe. After three folds,
complete five only if all portfolio gains are positive and their mean is at
least0.000015. Final admission remains mean0.000015,2.5 paired fold SE and4/5
positive; a performance upload also requires pooled gain0.00003, untouched
shadow fold2 gain0.00001, a five-context test certificate and private simulation.

Each full-fold exploratory run must fit within45 minutes. Do not start a job
whose expected completion would exceed the eight-hour cycle checkpoint.
Preserve source/data/library/config/checkpoint/feature/ID hashes and every
completed or invalid attempt under a unique name. No fourth mechanism this cycle.
