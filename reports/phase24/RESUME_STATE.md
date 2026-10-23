# Phase 24 — état de reprise

**Dernière mise à jour** : 2026-10-23T00:00Z
**Échéance** : 2026-10-31T23:59Z — **8 jours**
**Phase** : 24 — **clôturée par décision utilisateur**, aucun envoi

## Addendum 2026-10-23 — sonde externe, candidat X non déployé

La sonde du 22–23/10 a mesuré 27 vecteurs OOF publics sur nos folds immuables :

| vecteur | OOF | Δ vs v7 | t | folds |
|---|---|---|---|---|
| 11 TFM Deotte, individuellement | 0,9571–0,9611 | `−4,8e-3` à `−7,2e-4` | — | 0/5 |
| stack nested des 11 TFM | 0,961236 | `−6,2e-4` | −28,1 | 0/5 |
| v7 + stack TFM | 0,961933 | `+8,2e-5` | 5,9 | 5/5 |
| nj05 (blend public najiama) | 0,962234 | `+3,75e-4` | 11,4 | 5/5 |
| **candidat X = 0,15 v7 + 0,85 nj05** | **0,962247** | **`+3,886e-4`** | **13,4** | **5/5** |
| candidat X, shadow (777001) | 0,962247 | `+3,944e-4` | 9,5 | 5/5 |

Le côté test du candidat X est certifié : 299 844 lignes, ids exacts, distribution identique à
l'OOF, SHA `7032b4d928911cdb…`. La règle d'acceptation publique (`≥ 0,96170`) avait été gelée
**avant** tout score ; le score public n'a jamais été dépensé.

**Décision utilisateur : arrêt.** Aucun envoi. Les finalistes restent **A = v7 (57090508)** et
**B = v6 (56951906)**. Le candidat X reste local dans `artifacts/phase24_candidateX/` et
`reports/phase24/candidate_x_certification.json` ; il ne peut devenir finaliste que via un
nouvel envoi, non effectué.

Correction de lecture : notre plafond de modélisation propre est 0,96185 ; la frontière honnête
publique est ~0,9622 ; le 0,96370 public est de la sélection de surface, pas de la qualité de
modèle.

## Où en est le programme

| élément | valeur |
|---|---|
| champion | v7 `sol_tabpfn_internal2_aux10`, OOF `0,9618529228157235`, réf. `57090508`, public `0,96146` |
| finaliste A | v7, réf. `57090508` |
| finaliste B | v6, réf. `56951906`, OOF `0,9617656138245796`, public `0,96144` |
| réserve | v4, réf. `56845975`, OOF `0,96150857858`, public `0,96100` |
| soumissions utilisées aujourd'hui | `0/10` |
| soumissions utilisées en Phase 24 | `0` |
| GPU utilisé en Phase 24 | `0 h` |
| dernier GPU Kaggle connu | `28,57 h` restantes, rafraîchissement `2026-10-24` (à relire dans l'UI) |

## Prochaine action, dans l'ordre

1. **Utilisateur, critique** : sélectionner sur Kaggle les soumissions **57090508** et
   **56951906** comme finales (2 slots), avant le 2026-10-31T23:59Z.
2. **Utilisateur, critique** : relire la page *Rules* vers le 2026-10-30 et confirmer la règle
   « meilleure des deux sélectionnées ».
3. **Agent, si relancé** : `.\.venv\Scripts\python.exe tests\run_tests.py` puis vérifier
   `git log --oneline -1` et `git status --porcelain`.
4. **Agent, si un accélérateur ≥ 16 Gio apparaît** : rouvrir H23-01 (raw8) selon
   `configs/phase23/raw8_kaggle_prereg.json`. Ne pas rouvrir la géométrie de composition, les
   gates de segment, la re-pondération aux10, ni l'augmentation par lignes d'origine : ces
   quatre axes sont mesurés négatifs.

## Ce qui est fermé, avec le chiffre

| axe | chiffre de fermeture |
|---|---|
| géométrie de composition (7 règles) | meilleure `+1,694e-6`, t 0,95, 3/5, Holm 1,0 |
| gate de segment (16 tests gelés) | meilleur `+2,799e-6`, t 0,85 ; poids nul dans 11/16 |
| re-pondération des 10 membres aux10 | meilleure suppression `+2,379e-6` |
| moyenne de graines route | parent `+8,93e-5` → portefeuille `≈6,3e-6` (conversion 7,1 %) |
| store de 133 vecteurs | 0 delta positif ; plafond rétrospectif `+1,5368e-5` |
| doublons exacts train↔test | 0 |
| augmentation par lignes d'origine | fonction source transférée à `0,9499` seulement ; retards ne pèsent que `3,28e-4` |
| structure id/générateur | `AUC(id) = 0,500073` (phase 23) |
| transfert OOF→public (17 soumissions) | décalage commun `−3,69e-4` ; résidu `1,21e-4` ; pente `0,724`, `r 0,819` |

## Ce qui n'est pas fermé, et pourquoi

| mécanisme | blocage | condition de réouverture |
|---|---|---|
| raw8 TabPFN natif (H23-01) | T4 : 14,56 Gio < 12,4 Gio KV + 2,14 Gio activation | accélérateur ≥ 16 Gio utilisables |
| TabICLv2 vue riche (item 2) | 3 échecs ressource | idem |
| RealTabPFN-2.5 (item 12) | RESOURCE_BLOCKED | idem |
| fine-tune TabPFN 3.5 (item 34) | 6 sondes OOM | idem |
| confirmation `shadow` | aucun candidat n'a passé la découverte | un candidat à `+1,5e-5`, 5/5, t ≥ 2,5 |

## Fichiers à lire en premier

1. `reports/phase24/FINAL_REPORT.md` — synthèse et mécanisme.
2. `reports/phase24/private_finalist_decision.md` — A/B et arithmétique de transfert.
3. `reports/phase24/scientific_novelty_audit.md` — ce qui était vraiment non mesuré.
4. `reports/phase24/candidate_vs_v7_comparisons.json` — tous les chiffres.

## État git attendu

- Branche `main` en phase avec `origin/main`.
- Commit de clôture : `a81c739` — `phase24: close on a measured plateau; generator determinism
  mechanism; composition/segment/member/store axes falsified; no upload; finalists A=57090508
  B=56951906`, suivi du commit de mesure du transfert OOF→public.
- Pas de force push ; arbre de travail propre ; `reports/phase24/oof_public_transfer.py` et
  `_append_ledger.py` conservés pour reproductibilité.
