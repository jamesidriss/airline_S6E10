# Frozen continuation and resource budget

The fixed portfolio replicated in two independent primary evaluation folds:
paired gains +0.000312977866/+0.000329353076, mean +0.000321165471 and paired
SE0.000008187605. All ten fold0 controls reproduce exactly. This exceeds the
handoff's projected +0.0001 upside condition for extending a promoted branch's
compute budget. No public score or new weight chooses the continuation.

The third clean auxiliary fold completed with candidate AUC0.961787550190,
clean auxiliary AUC0.961442312237 and paired gain+0.000345237952. The fixed
three-fold portfolio's mean gain is+0.000329189631, SE0.000009313043 and3/3
positive. Report `sol_route_aux10_threefold.json` proves all ten fold0 controls
reproduce exactly. The declared promotion condition is now met: complete the
remaining primary folds without changing recipe or weights.
Then perform one frozen shadow confirmation and block10 finalist confirmation,
plus one fixed seed1202/fold0 sensitivity diagnostic. These are independent
checks, not parameter/weight selection. Per-process maximum remains45 minutes;
larger fits stay serialized. Total confirmation may exceed four hours because
the already replicated gain warrants it; progress and actual durations remain
in the append-only ledger and reports. Stop expansion if confirmation contradicts
the candidate materially.

All699635-row test inference has met GPU allocation limits, without predictions
or AUC. The remaining allocator-only check requests
`PYTORCH_ALLOC_CONF=expandable_segments:True` in a fresh process, preserving the
85 percent GPU cap. It must first pass the same complete-model100k-context
probability gate (max gap<=0.000002) and preserve all context rows. Record the
requested policy, actual peak allocation and any platform warning. If this
also fails, use the already predeclared five-primary-context probability-average
fallback rather than continuing unbounded memory experiments.

Native39 remains the one eligible lower-priority mechanism after the promoted
portfolio. Its matched protocol is fixed separately; conditional multitask and
field-factorization gates remain unmet. Resource failures count as neither
valid negative research branches nor model-performance comparisons.
