# Phase 19 shared-error atlas — BOTH v7 halves miss what?

- champion portfolio AUC **0.961852923** (hash verified, composition bit-exact)
- residual fail mass (pairs with port_n >= port_p): **0.038147** of all 1.2081e+11 pairs
- **shared fail (neither route nor aux ranks the pair right): 3,932,165,796 = 85.3% of portfolio fail mass**
- shared strict: 3,932,148,426; within-fold shared pairs that the portfolio ties: 0
- aggregation loss (portfolio fails while >=1 base ranks it right): 676,526,498 (0.005600 AUC mass)
- route-only errors: 717,231,183 | aux-only errors: 729,001,259 (one family right, the other wrong)

## Per-fold

 fold  shared_ge  shared_share_of_port_fail  route_only  aux_only  aggregation_loss_ge  port_auc
    0  156922098                   0.850671    29335122  29545883             27546451  0.961828
    1  157127065                   0.855391    28364157  28414288             26563339  0.961989
    2  157105937                   0.853084    28802458  29226819             27056423  0.961891
    3  160976466                   0.856309    28041874  29555671             27012395  0.961099
    4  154240035                   0.851018    28742231  28948309             27001817  0.962496

## Are shared errors predictable WITHOUT validation labels?

- set A (rich view only): pooled AUC **0.9318** (folds ['0.9321', '0.9319', '0.9318', '0.9310', '0.9323'])
- set B (+ inference-time confidences): pooled AUC **0.9235**
- trivial margin baseline: 0.0741

## Row-level structure

- rows participating in >=1 shared pair: 696,527 (99.6%)
- confident-both-wrong rows: 41,197
- near-tie-in-both rows: 230
- teacher-confident-wrong enrichment in top-burden quintile: 4.999x

## Highest-burden segments

                         segment   rows  participant_rate  mean_burden_rate  port_auc  route_auc  aux_auc  high_burden_share
                 Class==Eco Plus  30019          0.998301          0.034879  0.933572   0.933355 0.932628           0.287085
               unique_rating_key 254260          0.997971          0.034291  0.950323   0.949896 0.949564           0.250161
                      Class==Eco 327404          0.998314          0.033979  0.906539   0.905926 0.905709           0.241787
Customer Type==disloyal Customer 122645          0.994994          0.033469  0.926651   0.926095 0.925932           0.261519
                          age<30 206092          0.998234          0.033149  0.945516   0.945228 0.944796           0.217665
                      delay_none 606255          0.996038          0.032935  0.961272   0.960971 0.960841           0.203113
                    Gender==Male 351683          0.995814          0.032930  0.961245   0.960960 0.960782           0.208247
 Type of Travel==Personal Travel 202194          0.999723          0.032838  0.834314   0.832771 0.833309           0.185559

## Reading guide

- `shared_ge / port_fail` is the ceiling for what a *third model added to the average* could ever touch: pairs already failed by BOTH halves are invisible to reweighting.
- `aggregation_loss` is the headroom of aggregation changes (closed axis).
- `route_only` vs `aux_only` asymmetry shows which family carries the correcting signal on shared-adjacent pairs.
- Predictability AUCs answer whether a difficulty-aware mechanism can operate at test time (covariates/confidences only, no labels).