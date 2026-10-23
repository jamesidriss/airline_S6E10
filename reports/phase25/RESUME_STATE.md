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

## Ce qui n'est pas fermé

| mécanisme | blocage | condition de réouverture |
|---|---|---|
| TabPFN natif 16 Gio (raw8, H23-01) | T4 Kaggle 14,56 Gio ; **local 15,9 Gio disponible** | essai local, non lancé |
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
