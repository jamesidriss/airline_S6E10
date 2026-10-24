# Phase 25 — état de reprise

**Dernière mise à jour** : 2026-10-23T22:30Z
**Échéance** : 2026-10-31T23:59Z — **8 jours**
**Objectif** : gagner le privé.

## En une ligne

Le candidat X est **envoyé et accepté** (public `0,961840`, réf. `57570936`) ; le pool OOF
externe (288 vecteurs) est **mesuré et épuisé** ; l'écran de features a trouvé **deux blocs
réellement positifs** (`full_tec` `+1,24e-4`, `t 3,76`, 5/5 ; `full_enrich` `+6,6e-5`,
`t 2,68`, 5/5) mais **aucun n'ajoute quoi que ce soit au finaliste** (poids appris négatif).

## Finalistes

| slot | réf. | vecteur | public | OOF |
|---|---|---|---|---|
| **A** | **57570936** | X = `0,85·logit(nj05) + 0,15·logit(v7)` | **0,96184** | 0,962247 |
| B | 57090508 | v7, pipeline 100 % interne | 0,96146 | 0,961853 |

À faire par l'utilisateur : **sélectionner ces deux soumissions dans l'UI Kaggle** avant
2026-10-31T23:59Z, et relire la page *Rules* vers le 2026-10-30.

## Budget

| poste | valeur |
|---|---|
| soumissions du 2026-10-23 | **1/10** |
| soumissions Phase 25 | 1 |
| GPU Phase 25 | 0 h |
| GPU local | RTX 5070 Ti, 15,9 Gio, torch 2.11+cu128 (sm_120) — **débloque les essais TabPFN 16 Gio** |

## Ce qui a été fermé en Phase 25, avec le chiffre

| axe | chiffre |
|---|---|
| transfert OOF→public de X | prédit `0,961878`, observé `0,961840` (`0,3 σ`) ; `+3,8e-4` vs v7 |
| stack logistique imbriqué, 120 membres | `0,962201`, `−3,3e-5` vs nj05, **0/5** folds positifs |
| stack + 25 vecteurs maison (145) | `0,962206`, `−2,8e-5`, 1/5 |
| écran pairwise X + chacun des 288 | meilleur `+4,5e-6`, 3/5, poids négatif → **0 admissible** |
| équipondération top-288 | `0,961765` (monotone décroissant avec k) |
| `full_enrich` vs `full` (`extra_trees`) | `+6,61e-5`, SE `2,47e-5`, `t 2,68`, **5/5** → porte franchie |
| `full_enrich` déterministe | `−1,10e-4`, 1/5 → mécanisme `extra_trees` confirmé |
| `full_tec` vs `full` | `+1,244e-4`, SE `0,331e-4`, `t 3,76`, **5/5** → porte franchie ; 357 colonnes |
| X + membre `full_tec` (ρ 0,980) | `−0,03e-4`, 1/5, poids **−0,052** → aucun apport |
| validation adversariale train vs test | AUC **0,49979** → aucun décalage de distribution |
| fuite d'id (flux contigu) | id seul `0,50007` ; GBDT 9 features d'id `0,5016` (contigu `0,5014`) → rien à récolter |
| courbe d'apprentissage (fold 0) | `+4,5e-4` puis `+5,0e-4` par doublement : base encore limitée par les données |
| glouton Caruana, 368 candidats | `0,962226`, **−2,31e-5 vs X, 0/5** ; `+3,65e-4 vs v7, 5/5` |
| second slot (bootstrap privé 239 875) | `E[max] ≤ 2e-6` pour tout candidat, `0` pour v7 ; `P(v7 > X) = 0,00` |

## Ce qui n'est pas fermé

| mécanisme | blocage | condition de réouverture |
|---|---|---|
| TabPFN natif 16 Gio (raw8, H23-01) | T4 Kaggle 14,56 Gio ; **local 15,9 Gio disponible** | essai local, non lancé ; espérance bornée à `+3e-5` pour ≥25 h |
| TabICLv2 vue riche / RealTabPFN-2.5 / fine-tune TabPFN 3.5 | mêmes limites mémoire | idem |
| écran `full_tec`, `full_all21te` | terminé | verdict : `reports/p25a_screen.json` |

## Lecture honnête de la probabilité de victoire

La frontière mesurable est `0,9622` et nous sommes à `0,962247`. Aucun des 288 vecteurs
publics, aucune géométrie de blend, aucun bloc de features ne produit de gain admissible sur
X. Le privé (239 875 lignes, `SE ≈ 4,2e-4`) départagera des modèles qui estiment la même
fonction à ~`2e-5` près : la victoire est un tirage parmi la dizaine d'équipes situées à ce
niveau, avec un léger avantage mesuré pour nous (`+3,8e-4` public sur v7).

Estimation : **~10–20 % de victoire**, **~40–55 % de top-10**. Ce n'est pas un plafond de
méthode mais la conséquence du lissage de la fonction d'étiquetage par le générateur
(source `0,9888` → synthétique `0,9253` sur profil 16 clés).

**Addendum 2026-10-24 (Phase 25b)** : la courbe d'apprentissage explique *pourquoi* cette
frontière est solide. Un LGBM unique gagne encore `5e-4` par doublement de données à 560 000
lignes, donc l'estimateur de base n'est pas au plafond de Bayes — mais X est `1,08e-3`
au-dessus de ce même modèle sur le même fold, soit `2,2` doublements, donc un ensemble
équivalent à ~2,6 millions de lignes. Le reste de l'erreur est **partagé** par tous les
estimateurs entraînés sur les mêmes 700 000 lignes, ce qui est exactement pourquoi un membre
à `0,9612` et `ρ 0,98` n'ajoute rien. Détail : `PHASE25B.md`.
