# Phase18 error atlas — certified v7 vs v6 (primary OOF)

- v7 AUC **0.961852923** vs v6 **0.961765614** (delta **+8.73e-5**)
- Exact pair credit: rescued 148033942 / damaged 137485816 of 1.208e+11 pairs; net reproduces AUC delta exactly.
- Rows corrected: 374,387 | damaged: 322,805 | unchanged: 2,443
- Near-ties (|margin|<0.01): v7 2.0486% vs v6 2.0660%
- High-confidence wrong (margin<-0.20): v7 12,022,117,279 vs v6 12,056,245,956
- route/aux10 Pearson 0.997693 (Spearman 0.972317)

## Segment deltas (v7 - v6), worst five
                      segment   rows     delta  neg_cells
            Delay arr missing    292 -0.000101          0
              Class==Eco Plus  30019  0.000018          2
                     Age45-60 219777  0.000081          0
Customer Type==Loyal Customer 576990  0.000085          0
                 Gender==Male 351683  0.000086          0

## Segment deltas, best five
                        segment   rows    delta  neg_cells
Type of Travel==Personal Travel 202194 0.000392          0
                   Delay dep>60    353 0.001266          2
              All ratings equal    431 0.001598          1
                   Delay arr>60    310 0.001898          0
                       Zeros>=4   2940 0.003089          1

## Eco Plus
 rows    delta  delta_f0  delta_f1  delta_f2  delta_f3  delta_f4
30019 0.000018 -0.000073  0.000387  0.000035  -0.00046  0.000164