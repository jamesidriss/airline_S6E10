# Phase 20 -- 96-Experiment Master Catalogue

Generated 2026-10-17 · 96 experiments · champion v7 0.961852923 · target 0.962740

| # | Experiment | Family | Historical | Status | Pri | Fidelity | Eligibility | Blocking | Next action |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Mitra-v2 | FLM | NEGATIVE (this session) | COMPLETE_NEGATIVE | P1 | F2 | OK | done | CLOSED: prereg arms (20k/full) resource-infeasible on sm_120 (VRAM thrashing, no flash-attn); feasible-scale 2048ctx probe port_delta -1.63e-3 (bar +5e-5); FLM class 6/6 negative; do not rerun identical recipe |
| 2 | TabICLv2 | FLM | 20k-fold0 screen -1.40e-3 vs v7 bl | PARTIAL_VALID | P1 | F2 | OK | - | preserve 20k neg; probe native context; one bo |
| 3 | EXAONE Tabular | FLM | none | LICENSE_BLOCKED | P2 | F0 | NC | weights non-commercial; competition use not established | verify intended-use; do NOT submit with NC wei |
| 4 | Kumo Tabular | FLM | NEGATIVE (this session) | COMPLETE_NEGATIVE | P1 | F2 | CUSTOM | done | CLOSED: fold-0 screen @4096ctx AUC 0.9576, port_delta -1.46e-3 (bar +5e-5); sits on the FLM plateau (7/7 negative); in-context predict is feasible (550 rows/s) but cannot lift past ~0.957; do not rerun identical recipe |
| 5 | TabDPT v1.3/Turbo | FLM | 20k-fold0 screen -1.41e-3 | PARTIAL_VALID | P2 | F2 | OK | - | preserve 20k neg; only distinct larger-context |
| 6 | xRFM raw | FLM | raw-view 0.9306, -9.3e-3 | PARTIAL_VALID | P3 | F2 | OK | - | preserve raw neg; rich-view variant is item 47 |
| 7 | LimiX-2 | FLM | none | NOT_STARTED | P1 | F0 | CUSTOM | - | verify custom license vs prize; estimate VRAM; |
| 8 | LimiX-2M & 16M | FLM | none | NOT_STARTED | P1 | F0 | CUSTOM | - | verify licenses; inspect size/arch; feasibilit |
| 9 | Xiaomi-TabLDM | FLM | none | NOT_STARTED | P1 | F0 | OK | - | verify checkpoint terms; inspect context-memor |
| 10 | Causilo | FLM | 20k-fold0 screen -1.01e-3 | PARTIAL_VALID | P3 | F2 | OK | - | preserve neg; only clearly different mechanism |
| 11 | TabFM | FLM | 2k-CPU-fold0 -1.59e-3 | PARTIAL_VALID | P2 | F2 | NC | non-commercial weights | preserve 2k neg; audit native context limit; G |
| 12 | RealTabPFN-2.5 | FLM | none | NOT_STARTED | P2 | F0 | OK | - | locate official repo+ckpt+license; feasibility |
| 13 | TabPFN-3 vs 2.6 | FLM | none | NOT_STARTED | P1 | F0 | OK | - | verify exact checkpoints exist; pin revisions; |
| 14 | EXAONE/TabICL adaptation | FLM | none | NOT_STARTED | P2 | F0 | NC/OK | - | audit official adaptation; FIT-only model sele |
| 15 | FT-Transformer | NN | none (RealMLP/TabM ~1e-3 below) | NOT_STARTED | P2 | F1 | OK | - | canonical recipe; smoke; fold-0 OOF vs v7 delt |
| 16 | SAINT | NN | none | NOT_STARTED | P3 | F1 | OK | - | canonical recipe; smoke; fold-0 |
| 17 | DCNv2 | NN | none | NOT_STARTED | P2 | F1 | OK | - | explicit-cross inductive bias (distinct from R |
| 18 | AutoInt | NN | none | NOT_STARTED | P3 | F1 | OK | - | smoke; fold-0 |
| 19 | DeepFM & xDeepFM | NN | none | NOT_STARTED | P3 | F1 | OK | - | smoke; fold-0 |
| 20 | NODE | NN | none | NOT_STARTED | P3 | F1 | OK | - | smoke; fold-0 |
| 21 | GRANDE | NN | none | NOT_STARTED | P3 | F1 | OK | - | smoke; fold-0 |
| 22 | TabNet | NN | none | NOT_STARTED | P4 | F1 | OK | - | low prior (weak historically); smoke only if t |
| 23 | ExcelFormer | NN | none | NOT_STARTED | P3 | F1 | OK | - | smoke; fold-0 |
| 24 | AMFormer | NN | none | NOT_STARTED | P3 | F1 | OK | - | smoke; fold-0 |
| 25 | T2G-Former | NN | none | NOT_STARTED | P3 | F1 | OK | - | smoke; fold-0 |
| 26 | Trompt | NN | none | NOT_STARTED | P3 | F1 | OK | - | smoke; fold-0 |
| 27 | Periodic/PLR MLP | NN | none | NOT_STARTED | P2 | F1 | OK | - | PLR numeric embeddings; compare numeric-encode |
| 28 | Tabular ResNet | NN | none | NOT_STARTED | P2 | F1 | OK | - | smoke; fold-0 |
| 29 | Supervised MoE | NN | none | NOT_STARTED | P2 | F1 | OK | - | gating uses FIT labels only; smoke; fold-0 |
| 30 | Service-interaction net | NN | none | NOT_STARTED | P2 | F1 | OK | - | domain-motivated cross-rating interactions; sm |
| 31 | Fine-tune TabICLv2 | OBJ | none | NOT_STARTED | P2 | F0 | OK | - | official FT procedure; FIT-only selection |
| 32 | Fine-tune Mitra-v2 | OBJ | NEGATIVE (this session) | COMPLETE_NEGATIVE | P1 | F1 | OK | done | CLOSED: the 50-step FT + 8-fold bag recipe was exercised by the row-1 screen (prereg mitra_screen_prereg) and is negative; do not rerun identical recipe |
| 33 | LoRA/adapters | OBJ | none | NOT_STARTED | P3 | F0 | OK | - | audit which FLMs support adapters; bounded tes |
| 34 | Fine-tune TabPFN 3.5 | OBJ | none | NOT_STARTED | P2 | F0 | OK | - | verify license+16GB VRAM sufficiency for FT |
| 35 | Multi-task | OBJ | prior conditional multi-task | NOT_STARTED | P2 | F1 | OK | - | audit prior conditional multi-task; true share |
| 36 | Ordinal aux loss | OBJ | none | NOT_STARTED | P3 | F1 | OK | - | ordinal aux objective; smoke; fold-0 |
| 37 | Supervised contrastive | OBJ | generic SSL32 closed | NOT_STARTED | P3 | F1 | OK | - | distinct supervised-contrastive (not generic S |
| 38 | Diff AUC surrogate | OBJ | none | NOT_STARTED | P3 | F1 | OK | - | AUC surrogate vs standard BCE; fold-0 |
| 39 | Hard-pair mining | OBJ | H19-A fixed reweight NEGATIVE/INC | NOT_STARTED | P3 | F1 | OK | - | distinct mining (not fixed reweight); never ou |
| 40 | Noise-robust loss | OBJ | none | NOT_STARTED | P4 | F1 | OK | - | compare noise-robust losses; fold-0 |
| 41 | EMA / SWA | OBJ | none | NOT_STARTED | P3 | F1 | OK | - | EMA/SWA on a neural candidate to cut seed vari |
| 42 | Multi-seed ensemble | OBJ | none | NOT_STARTED | P3 | F2 | OK | - | only after one new architecture shows competit |
| 43 | Nested distillation of v7 | OBJ | generic distillation closed | HISTORICAL_EQUIVALENT | P4 | F0 | OK | closed | do not reopen generic distillation; nested-onl |
| 44 | Cross-family co-regularization | OBJ | none | NOT_STARTED | P4 | F1 | OK | - | needs 2 strong base candidates first |
| 45 | H19-C nearest-key smoothing | FEAT | DEAD (this session) | COMPLETE_NEGATIVE | P3 | F5 | OK | done | CLOSED: fold-0 PATH was a fluke (full-5 mean -2.25e-6, 2/5 folds, OOF unique-key negative); base-6/Hamming-1/lam30/mc2 nearest-key smoothing closed; do not rerun identically |
| 46 | Learned similarity metric | FEAT | none | NOT_STARTED | P3 | F1 | OK | - | FIT-only learned metric; member-family compat  |
| 47 | xRFM rich view | FEAT | distinct from item 6 | NOT_STARTED | P3 | F2 | OK | - | rich-view xRFM (distinct mechanism from raw-vi |
| 48 | Hierarchical 13-rating | FEAT | none | NOT_STARTED | P3 | F1 | OK | - | hierarchical rating model; fold-0 |
| 49 | Conditional ordinal embed | FEAT | none | NOT_STARTED | P4 | F1 | OK | - | smoke; fold-0 |
| 50 | Learned interactions | FEAT | te_cond crosses exist | NOT_STARTED | P3 | F1 | OK | - | learned interaction rep vs manual crosses |
| 51 | Observed-minus-predicted ratings | FEAT | residual-group-bias closed | NOT_STARTED | P3 | F1 | OK | - | distinct residual-feature (not group-bias); fo |
| 52 | Supervised rating factorization | FEAT | none | NOT_STARTED | P3 | F1 | OK | - | supervised factorization; fold-0 |
| 53 | Conditional clustering | FEAT | none | NOT_STARTED | P4 | F1 | OK | - | clusters fit without eval labels |
| 54 | Smoothed categorical model | FEAT | conditional-density closed | NOT_STARTED | P4 | F1 | OK | - | distinct smoothing (not conditional-density) |
| 55 | LGBM leaf embeddings | FEAT | none | NOT_STARTED | P3 | F2 | OK | - | leaf-embedding rep into a second family |
| 56 | Local continuous transforms | FEAT | generic local transforms partly ab | NOT_STARTED | P4 | F1 | OK | - | distinct local transform; redundancy vs existi |
| 57 | Low-rank rating interactions | FEAT | none | NOT_STARTED | P3 | F1 | OK | - | low-rank cross-rating; fold-0 |
| 58 | Rare-combination regularizer | FEAT | none | NOT_STARTED | P4 | F1 | OK | - | fold-0 |
| 59 | Native sklearn ExtraTrees | CLASS | LGBM extra_trees rich-view +2.3e-4 | NOT_STARTED | P3 | F2 | OK | - | distinct from LGBM flag; rich view only (raw v |
| 60 | Rotation/oblique forest | CLASS | none | NOT_STARTED | P4 | F1 | OK | - | fold-0 |
| 61 | Genuine EBM | CLASS | H18-B boosted-stump NEGATIVE | NOT_STARTED | P3 | F2 | OK | - | real EBM distinct from failed stump; fold-0 |
| 62 | GAM splines+interactions | CLASS | none | NOT_STARTED | P4 | F1 | OK | - | fold-0 |
| 63 | Kernel Ridge | CLASS | none | NOT_STARTED | P4 | F1 | OK | - | Nyström/RFF approx; no dense 700k^2 kernel |
| 64 | Approx RBF SVM | CLASS | none | NOT_STARTED | P4 | F1 | OK | - | RFF approx; fold-0 |
| 65 | Factorized logistic interactions | CLASS | none | NOT_STARTED | P4 | F1 | OK | - | fold-0 |
| 66 | Hierarchical kernels | CLASS | none | NOT_STARTED | P4 | F0 | OK | - | resource-limited likely |
| 67 | Local density classifier | CLASS | prior density failures | HISTORICAL_EQUIVALENT | P5 | F0 | OK | weak prior | audit prior; only distinct mechanism |
| 68 | Supervised mixed-factor | CLASS | none | NOT_STARTED | P4 | F1 | OK | - | fold-0 |
| 69 | Conditional rule ensembles | CLASS | none | NOT_STARTED | P4 | F1 | OK | - | fold-0 |
| 70 | Nested v7+FLM stack | ENS | none | NOT_STARTED | P4 | F0 | OK | blocked-on-base | needs a strong FLM base first |
| 71 | Cross-family MoE | ENS | none | NOT_STARTED | P4 | F0 | OK | blocked-on-base | needs 2+ strong families |
| 72 | Covariate-only gating | ENS | none | NOT_STARTED | P4 | F1 | OK | - | gate uses covariates only |
| 73 | Shared-error specialist | ENS | 85% residual is shared | NOT_STARTED | P4 | F0 | OK | hard | needs a model right WHERE both halves fail (H1 |
| 74 | Near-tie specialist | ENS | none | NOT_STARTED | P4 | F1 | OK | - | FIT-only margins |
| 75 | Hard-pair specialist | ENS | H19-A INC (mech real, net ~0) | NOT_STARTED | P4 | F1 | OK | - | distinct from H19-A fixed reweight; must not d |
| 76 | Feature-subspace ensemble | ENS | none | NOT_STARTED | P4 | F2 | OK | - | fold-0 |
| 77 | Champion B utility selection | ENS | Champion B UNRESOLVED | NOT_STARTED | P4 | F2 | OK | no-candidate | optimize only with a real B candidate |
| 78 | Cross-fold robust ensemble | ENS | none | NOT_STARTED | P4 | F2 | OK | - | fold-0 + shadow |
| 79 | Fixed vs nested blending | ENS | fixed blend certified | NOT_STARTED | P5 | F0 | OK | low-value | do not do large in-sample weight search |
| 80 | AUC ceiling estimate | GEN | prior ~3.8%-noise discussion | NOT_STARTED | P5 | F1 | OK | - | distinguish theory vs measurement; sensitivity |
| 81 | Generator conditional model | GEN | none | NOT_STARTED | P5 | F0 | OK | - | conceptual; must yield actionable signal |
| 82 | Random vs feature noise | GEN | none | NOT_STARTED | P5 | F0 | OK | - | identifiability assumptions |
| 83 | Feature-dependent transition | GEN | none | NOT_STARTED | P5 | F0 | OK | blocked-identifiability | only under defensible identifiability |
| 84 | Original conditional transfer | GEN | original conditional stats useful; | NOT_STARTED | P4 | F2 | OK | - | audit prior; conditional (not append) transfer |
| 85 | Source-trained specialists | GEN | ogte_/ogs_ features exist | NOT_STARTED | P4 | F2 | OK | - | original-label features already used; incremen |
| 86 | Joint probabilistic model | GEN | none | NOT_STARTED | P5 | F0 | OK | - | conceptual |
| 87 | Causal mechanism | GEN | none | NOT_STARTED | P5 | F0 | OK | - | treat as hypothesis; observational limits |
| 88 | H19-B full-fit vs bag5 | DEP | NEGATIVE (this session) | COMPLETE_CERTIFIED | P0 | F5 | OK | done | do not repeat; certified full-FIT confirmed |
| 89 | H19-B replication | DEP | conditional | NOT_STARTED | P5 | F0 | OK | not-warranted | frozen rule: only if first policy test positiv |
| 90 | FIT size / bag count | DEP | none | NOT_STARTED | P5 | F0 | OK | blocked-on-arch | only after a competitive new architecture |
| 91 | Seed vs subset variance | DEP | none | NOT_STARTED | P4 | F1 | OK | - | measure separately |
| 92 | Independent FLM bagging | DEP | none | NOT_STARTED | P5 | F0 | OK | blocked-on-acc | only if both FLMs accurate+eligible |
| 93 | Matched-coverage policies | DEP | H19-B NEGATIVE | NOT_STARTED | P5 | F0 | OK | not-warranted | H19-B already showed bagging hurts at matched  |
| 94 | OOF/test stability audit | DEP | none | NOT_STARTED | P4 | F1 | OK | - | audit before finalist lock |
| 95 | New nested Champion B | DEP | Champion B UNRESOLVED | NOT_STARTED | P3 | F0 | OK | no-strong-candidate | construct after a strong distinct base exists |
| 96 | Independent finalist validation | DEP | shadow reused | NOT_STARTED | P3 | F0 | OK | - | note reused shadow is not fresh; new scheme if |
