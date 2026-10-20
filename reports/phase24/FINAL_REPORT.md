# Phase 24 — rapport final

**Date** : 2026-10-20 (UTC) · **Échéance compétition** : 2026-10-31T23:59Z (11 jours)
**Champion** : v7 `sol_tabpfn_internal2_aux10`, OOF `0,9618529228157235`, réf. Kaggle `57090508`
**Décision** : aucun envoi · finalistes A = v7 (57090508), B = v6 (56951906)

---

## 1. Ce que la Phase 24 devait faire, et ce qu'elle a fait

Le mandat était triple : revue forensique des phases 17–23, file d'hypothèses classée et
préenregistrée, expériences réelles contre v7 avec confirmation indépendante. Les onze livrables
demandés existent ; six mesures nouvelles ont été produites, toutes sur CPU, sans GPU et sans
soumission.

| # | mesure | résultat | statut |
|---|---|---|---|
| E1 | inventaire des 133 vecteurs OOF stockés | 0 vecteur avec delta fold moyen positif ; plafond rétrospectif `+1,5368e-5` | exhaustion |
| E2 | laboratoire de géométrie de composition (7 règles gelées) | meilleure règle `+1,694e-6` (t 0,95, 3/5, Holm 1,0) | falsifié |
| E3 | diagnostic des parents (10 membres aux10, 2 graines route) | moyenne aux10 optimale à `2,4e-6` ; graine route `+8,93e-5` sur fold 0 | dilué à 7,1 % |
| E4 | gate de segment (8 familles × 2 segments, 16 tests gelés) | meilleur `+2,799e-6` (t 0,85) ; poids nul dans 11/16 | falsifié |
| E5 | audit de doublons et de structure de clefs | 0 doublon exact train↔test ; clef de profil médiane 1, p90 3 | fermé |
| E6 | ancre source + écart de déterminisme | source `0,988806` à 130k ; LOO de profil `0,9710` (source) vs `0,9253` (synthétique) | **mécanisme identifié** |
| E7 | transfert OOF → public sur les 17 soumissions | décalage commun `−3,69e-4` ; dispersion résiduelle `1,21e-4` ; pente `0,724`, `r 0,819` | `P(v6 > v7)` entre 15 % et 41 % |

## 2. Le résultat scientifique central

La question restée ouverte depuis la Phase 19 était : *pourquoi 85,3 % des paires manquées par
le champion sont-elles partagées par tous les membres, alors que chaque famille prise seule est
meilleure qu'un hasard ?* La Phase 24 y répond par une mesure, pas par une hypothèse.

### 2.1 L'ancre source

Sur les 129 880 lignes du jeu d'origine (les 21 mêmes variables, aucun doublon — vérifié), un
GBDT fort à 5 folds atteint **`0,988806`** ; à 64 940 lignes, `0,987873` (gain de doublement
`+9,33e-4`). La distribution d'origine est donc très au-dessus du plafond synthétique.

### 2.2 L'écart de déterminisme, à clé identique

Même clé de 16 colonnes (13 notes + `Class` + `Type of Travel` + `Customer Type`), même
statistique (moyenne de groupe en leave-one-out) :

| statistique | source | synthétique |
|---|---|---|
| lignes en groupes non singletons | 36 074 | 395 171 |
| AUC de la moyenne de groupe LOO | **0,970995** | **0,925285** |
| part de lignes en groupes *purs* (taille ≥ 2) | 26,8 % | 30,8 % |
| taille moyenne de groupe | 1,245 | 1,812 |

Lecture : la population synthétique **répète davantage de profils** que la source et les
**étiquette plus bruyamment**. Deux passagers distincts partageant exactement le même profil de
16 notes reçoivent beaucoup plus souvent des étiquettes différentes dans le synthétique que
dans la source. Aucune fonction de ces 16 colonnes ne peut les séparer.

### 2.3 Deux hypothèses alternatives réfutées par la mesure

- **« Le générateur a détruit le signal des retards. »** Faux : le modèle source sans les deux
  colonnes de retard perd seulement `3,28e-4` (`0,988806 → 0,988478`), et la relation
  retard→étiquette est conservée (voire légèrement renforcée) dans le synthétique.
- **« La source est gonflée par des doublons. »** Faux : 129 880 vecteurs de features uniques,
  zéro doublon.

La conclusion est donc que le générateur a **lissé la fonction d'étiquetage** de la source, et
que le plafond pratique de la distribution synthétique se situe autour de `0,9618–0,9620`.
Notre champion y est.

## 3. Pourquoi l'objectif `0,962740` n'est pas atteignable par la modélisation

L'objectif aspirational demandait `+8,9e-4` au-dessus de v7. Les mesures de la Phase 24
chiffrent les trois seules voies possibles :

| voie | gain maximal mesuré | verdict |
|---|---|---|
| meilleure combinaison des parents existants | `+1,7e-6` (C1), toutes les règles `≤ +1,7e-6` | fermée |
| n'importe quel vecteur déjà stocké, avec sélection rétrospective | `+1,54e-5` sur 133 vecteurs × 7 poids | fermée |
| amélioration locale d'un parent, propagée | `+3,8e-6` mesuré pour `+5,34e-5` de route (conversion 7,1 %) | fermée |

Il faudrait donc une **nouvelle information** sur les 21 colonnes, ou un **nouveau parent**
suffisamment décorrélé. Les deux espaces ont été inventoriés : l'information nouvelle est
absente (0 doublon, clef médiane de taille 1) et les quatre familles restantes (raw8, TabICLv2
vue riche, RealTabPFN-2.5, fine-tune TabPFN) sont toutes bloquées par le matériel, pas par un
résultat.

Une précision d'honnêteté : ceci ne démontre pas que `0,962740` est impossible. Cela démontre
que **tous les leviers connus et exécutables sur ce matériel sont épuisés**, et que le seul
levier non testé (raw8) est chiffré à `≈+3e-5` au portefeuille pour `≥25 h` de GPU — soit
`0,03×` l'objectif.

## 4. Ce que la Phase 24 ajoute, au-delà des négatifs

1. **La géométrie de composition est optimale, et on sait pourquoi.** La masse d'agrégation de
   `0,0055997` mesurée par l'atlas de la Phase 19 n'est pas récupérable par changement de
   combinateur : le rang, la CDF, la moyenne de probabilités et la moyenne géométrique sont tous
   *strictement pires* de `6e-6` à `1,1e-5`, et les règles ajustées plafonnent à `+1,7e-6`.
2. **La conversion parent→portefeuille est confirmée sur un axe indépendant.** L'audit
   `internal2 → internal4` donnait 7,1 % ; la moyenne de graines route donne `+8,93e-5` au
   parent pour `≈6,3e-6` attendus au portefeuille, et la re-pondération des membres aux10 ne
   vaut que `2,4e-6`. Trois mesures indépendantes, même ordre de grandeur.
3. **La borne d'exhaustion du store est chiffrée.** 133 vecteurs, zéro delta positif, plafond
   rétrospectif `+1,54e-5` — et l'enveloppe `(AUC, rho)` tombe à `+4,3e-6` dès qu'on exige
   `rho ≥ 0,998`, c'est-à-dire dès qu'on refuse la sélection sur le bruit.
4. **L'absence totale de fuite par doublon est établie** : 0 correspondance exacte
   train↔test sur 21 colonnes, 0 doublon interne à chaque split.
5. **La structure id/générateur reste fermée** (`AUC(id) = 0,500073`), confirmée et non rouverte.
6. **Le pouvoir de résolution de la place publique est chiffré.** Sur les 17 soumissions, après
   retrait du décalage commun de `−3,69e-4`, la pente OOF→public est de `0,724` avec `r = 0,819` :
   l'OOF n'explique qu'environ deux tiers de la variance des scores publics. La dispersion
   résiduelle par soumission (`1,21e-4`, soit `1,72e-4` appariée) est du même ordre que la bande
   entière qui sépare le top public (`0,96159`) de v7 (`0,96146`), soit `1,3e-4`. Dans cette
   bande, le classement public est dominé par le bruit d'échantillonnage.

## 5. Décision de soumission

```
SUBMISSION_DECISION = NO_UPLOAD_IN_PHASE_24
```

Aucun candidat n'a franchi PATH A (alternative `+1,5e-5`, 5/5, `t ≥ 2,5`) ni PATH B
(`rho < 0,99`). Le budget est intact (`0/10` aujourd'hui, 0 sur la phase). Le champion est
re-certifié pièce par pièce :

- OOF `0,9618529228157235`, SHA tableau `1981745c…`, SHA fichier `bd4d288b…` ;
- prédiction test SHA tableau `2b744d0d…`, fichier `e5b3b819…` ;
- CSV `fa4947fe…`, 299 844 lignes, ids identiques à `sample_submission.csv`, écart max
  `1,11e-16` avec le vecteur certifié ;
- recomposition depuis les deux parents test : `5,163e-8`.

Finalistes : **A = 57090508 (v7)**, **B = 56951906 (v6)**. Les 17 soumissions ont été
revérifiées `complete` via l'API le 2026-10-20 ; le slot B est gratuit sous la règle du maximum
et `P(v6 > v7)` vaut **15,4 %** (OOF seul), **26,6 %** (rétraction à mi-vie de l'avantage public
de `6,73e-5` de v6) ou **40,8 %** (avantage public pris au réel). v4 (`56845975`) reste la
couverture de queue si l'utilisateur craint une défaillance structurelle de la famille TabPFN
plutôt qu'un écart de score : `P(v4 > v7) ≤ 3e-5`.

## 6. Conditions exactes de réouverture

La Phase 24 se ferme sur un plateau **mesuré**, pas sur un abandon. Quatre conditions
rouvriraient le programme, dans l'ordre de rentabilité :

1. **Accélérateur ≥ 16 Gio utilisables** → raw8 (H23-01) et TabICLv2 vue riche deviennent
   exécutables ; protocole déjà gelé (`configs/phase23/raw8_kaggle_prereg.json`).
2. **Source d'information nouvelle** sur les 21 colonnes (autre millésime du jeu d'origine,
   autre générateur, labels d'un sous-ensemble) → rouvre la seule voie qui n'est pas diluée.
3. **Changement de règle** sur la page Rules (slots finaux, métrique, split privé) → rouvre la
   décision de sélection, pas la modélisation.
4. **Preuve qu'un concurrent public est au-dessus de `0,9620` de façon reproductible** → alors
   le plafond mesuré ici serait faux et l'audit devrait être refait.

## 7. Index des livrables

| fichier | contenu |
|---|---|
| `RESUME_STATE.md` | état durable, prochaine commande, budgets |
| `scientific_novelty_audit.md` | revue forensique 17–23, F0/F1, items 80 et 91 exécutés |
| `hypothesis_registry.json` | 9 hypothèses classées, champs obligatoires complets |
| `execution_queue.json` | file exécutée, statuts, budgets |
| `experiment_results.jsonl` | 6 résultats de phase, append-only |
| `candidate_vs_v7_comparisons.json` | toutes les comparaisons chiffrées contre v7 |
| `resource_and_budget_status.json` | GPU, quotas, soumissions, blocages |
| `independent_confirmation.md` | pourquoi `shadow` n'a pas été ouvert, puissance |
| `submission_certification.md` | re-certification du champion, décision |
| `private_finalist_decision.md` | A/B, arithmétique de transfert, action UI |
| `champion_reverification.json` | sortie brute de la vérification du jour |
| `oof_public_transfer.json` | table de transfert OOF→public sur les 17 soumissions |
| `kaggle_submissions_20261020.csv` | liste distante brute (17 lignes, toutes `COMPLETE`) |

## 8. Intégrité

- Aucun artefact certifié n'a été modifié ; toutes les vérifications de la Phase 24 sont en
  lecture seule.
- Aucune soumission ; aucun GPU ; aucun secret dans les fichiers ajoutés.
- Tous les résultats négatifs sont conservés, y compris ceux qui contredisent des rapports
  antérieurs (l'hypothèse « retards détruits par le générateur » est explicitement réfutée
  ci-dessus).
- Tests du dépôt : exécutés avant commit, `15/15`.
