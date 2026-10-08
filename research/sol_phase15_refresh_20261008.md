# Phase 15 targeted primary-source refresh

Accessed 2026-10-08 18:30 UTC. Sanitized discussion bodies and dated comments are
in `raw/sol_phase15_public_refresh_20261008.json`. No competitor predictions
were imported. Notebook search returned Unauthenticated; author-linked notebook
methods were reviewed through their discussions. GitHub library documentation
was retrieved directly.

Authenticated API retrieval subsequently succeeded for the feature-ladder and
seed-transfer notebooks (code only, no execution or prediction download).
`reports/sol_phase15_notebook_protocol_audit_20261008.json` records exact file
hashes. Both use outer evaluation labels for checkpoint selection. The
seed-transfer notebook separately excludes its fake-test20percent population,
so its transfer comparison is useful mechanism evidence, while its inner OOF
remains ineligible for our admission contract.

| Source | Evidence classification | Decision |
|---|---|---|
| [Feature ladder, 747358](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/747358) | Author explicitly says round counts were selected on the same evaluation folds; reported gains are not independent. Aux expected-rating features help that LightGBM, whereas categorical crosses were weak there. | No admission of reported scores. Supports bounded representation test only; mechanism already implemented for our classical models. |
| [39 crosses, updated comment](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745892) | Original pseudocode uses outer early stopping. New comment claims improvement on a rich CatBoost frame; its selection protocol is unclear. | Run our own frozen inner-ES native CatBoost experiment; no reported OOF gain used as promotion evidence. |
| [Seed blend transfer, 747182](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/747182) | Author's held-out pseudo-test comparison suggests averaging can reduce apparent OOF diversity gains; full selection details were not inspected. | Transfer caution, not a new tuning branch. Keep heterogeneous representations, certified test policy and independent checks. |
| [TFM survey and starter notebooks](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363) | Own pretrained inference on author folds; public stacking uses imported predictions. | Already-tested TabPFN family. No imported predictions, other heavy TFM catalogue or checkpoint modification. |
| [PriorLabs classifier implementation](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py) and [release notes](https://github.com/PriorLabs/TabPFN/blob/main/CHANGELOG.md) | Primary implementation explains categorical handling and estimator feature coverage. It provides no S6E10 auxiliary-feature gain estimate. | Preserve installed9.1.0/checkpoint and categorical protocol. A representation benefit must be measured locally. |

The ranked queue remains the three predeclared entries in
`sol_phase15_scope_20261008.json`: raw/route complement, native39, and a
conditional third mechanism. No fourth branch is authorized by this cycle.
