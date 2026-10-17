# H19-B — aux10 deployment policy: single full-FIT vs bag-of-five (fold 4)

**Verdict:** `NEGATIVE` · portfolio Δ = -2.783e-04 (paired SE 1.75e-04, -1.6×SE)

| metric | control (full-FIT) | treatment (bag-of-five) | Δ |
|---|---|---|---|
| aux10 standalone AUC | 0.9620929850 | 0.9620533686 | -3.962e-05 |
| portfolio AUC | 0.9624956399 | 0.9622173311 | -2.783e-04 |

- pair net credit -1344941.5 (identity err 4.5e-07); ctrl right/tie/wrong 4651310560/132/181241720, bag5 4649965620/129/182586663
- Spearman 0.992822 · logit-Pearson 0.998990
- |Δp| mean 1.04e-01 · median 1.13e-01 · p95 1.59e-01 · max 2.99e-01 · sign-flip 0.0119

## Segment effects (portfolio Δ, sorted)
| segment | rows | pos rate | Δ |
|---|---|---|---|
| Type of Travel=Personal Travel | 40420 | 0.096 | -1.130e-03 |
| FlightDistance_q4 | 34979 | 0.742 | -8.709e-04 |
| Class=Business | 68374 | 0.727 | -7.382e-04 |
| Customer Type=disloyal Customer | 24583 | 0.199 | -5.736e-04 |
| Class=Eco | 65614 | 0.166 | -4.760e-04 |
| DepartureDelay>0 | 12901 | 0.406 | -4.521e-04 |
| Age_q3 | 33962 | 0.607 | -4.071e-04 |
| FlightDistance_q3 | 34741 | 0.435 | -3.756e-04 |
| Gender=Female | 69565 | 0.438 | -3.474e-04 |
| Type of Travel=Business travel | 99507 | 0.585 | -3.470e-04 |
| Age_q2 | 38379 | 0.407 | -3.158e-04 |
| Age_q1 | 35219 | 0.261 | -2.954e-04 |
| Customer Type=Loyal Customer | 115344 | 0.496 | -2.827e-04 |
| DepartureDelay==0 | 127026 | 0.447 | -2.734e-04 |
| ArrivalDelay==0 | 127983 | 0.452 | -2.440e-04 |
| ArrivalDelay>0 | 11886 | 0.353 | -2.335e-04 |
| Gender=Male | 70362 | 0.449 | -2.140e-04 |
| Age_q4 | 32367 | 0.515 | -1.955e-04 |
| FlightDistance_q1 | 35396 | 0.308 | -7.581e-05 |
| FlightDistance_q2 | 34811 | 0.291 | -6.763e-05 |
| Class=Eco Plus | 5939 | 0.242 | -2.315e-05 |

## Limitations
- Pseudo-test standard: OOF is unchanged; this cannot pass the admission gate.
- Fold 4 is part of the primary OOF and was used in earlier research; not a hidden holdout.
- A pseudo-test gain is not proof of private performance; S6E9 shows policy deltas do not transfer cleanly.
