# Phase 24 — décision sur les finalistes privés

## Décision

| slot | candidat | réf. Kaggle | OOF | public | rôle |
|---|---|---|---|---|---|
| **A** | `v7_sol_tabpfn_internal2_aux10` | **57090508** | `0,9618529228157235` | `0,96146` | meilleur estimateur ponctuel, artefact certifié |
| **B** | `v6_sol_tabpfn_route_aux10` | **56951906** | `0,9617656138245796` | `0,96144` | assurance gratuite sous la règle du max |

Réserve documentée : `v4_fulldata`, réf. **56845975** (OOF `0,96150857858`, public `0,96100`),
uniquement comme couverture contre une défaillance *structurelle* de la famille TabPFN, pas
comme candidat de score.

Vérification faite le 2026-10-20 via l'API Kaggle : les 17 soumissions sont `COMPLETE`, y
compris `57090508` (public `0,96146`) et `56951906` (public `0,96144`). Liste brute :
`reports/phase24/kaggle_submissions_20261020.csv`.

## Pourquoi le slot B est gratuit

La Phase 23 a confirmé sur la page Rules que le score privé est le **maximum** des soumissions
sélectionnées (rapport `reports/phase23/finalist_B_analysis.md`, ligne 65). Sélectionner B ne
peut donc pas dégrader le score ; B n'a de valeur que si `P(B > A) > 0`. Il faut malgré tout
**relire la page Rules vers le 2026-10-30** : c'est la seule règle dont le changement
invaliderait cette décision.

## Table de transfert OOF → public (17 soumissions, mesure complète)

Calculée par `reports/phase24/oof_public_transfer.py`, sortie `oof_public_transfer.json`.
Écart = public − OOF. « résidu » = écart − écart commun moyen.

| réf. | candidat | OOF | public | écart | résidu |
|---|---|---|---|---|---|
| 57386012 | endgame_10_…_d255 | 0,961185 | 0,96093 | `−2,55e-4` | `+1,14e-4` |
| 57386008 | endgame_09_xt_xt_d127_s2 | 0,961189 | 0,96075 | `−4,39e-4` | `−6,96e-5` |
| 57386003 | endgame_08_…_ss06 | 0,961194 | 0,96109 | `−1,04e-4` | `+2,65e-4` |
| 57385997 | endgame_07_z4_xt_f10_s3 | 0,961196 | 0,96090 | `−2,96e-4` | `+7,27e-5` |
| 57385994 | endgame_06_…_d63 | 0,961201 | 0,96095 | `−2,51e-4` | `+1,18e-4` |
| 57385988 | endgame_05_…_cs05 | 0,961223 | 0,96100 | `−2,23e-4` | `+1,46e-4` |
| 57385982 | endgame_04_xt_xt_f10 | 0,961242 | 0,96086 | `−3,82e-4` | `−1,31e-5` |
| 57385975 | endgame_03_…_s1 | 0,961250 | 0,96094 | `−3,10e-4` | `+5,93e-5` |
| 57385972 | endgame_02_z4_xt_f10_s4 | 0,961260 | 0,96085 | `−4,10e-4` | `−4,13e-5` |
| 56788249 | v1_equal4 | 0,961273 | 0,96093 | `−3,43e-4` | `+2,59e-5` |
| 56793823 | v2_equal34 | 0,961438 | 0,96092 | `−5,18e-4` | `−1,49e-4` |
| 57385968 | endgame_01_cand_family_balanced | 0,961502 | 0,96101 | `−4,92e-4` | `−1,23e-4` |
| 56805526 | v3_final | 0,961509 | 0,96098 | `−5,29e-4` | `−1,60e-4` |
| 56845975 | v4_fulldata | 0,961509 | 0,96100 | `−5,09e-4` | `−1,40e-4` |
| 56918343 | v5_aux_cross | 0,961523 | 0,96103 | `−4,93e-4` | `−1,24e-4` |
| 56951906 | **v6_sol_tabpfn_route_aux10** | 0,961766 | 0,96144 | `−3,26e-4` | `+4,33e-5` |
| 57090508 | **v7_sol_tabpfn_internal2_aux10** | 0,961853 | 0,96146 | `−3,93e-4` | `−2,40e-5` |

Statistiques : moyenne commune `−3,69e-4` ; écart-type `1,21e-4` ; régression OOF→public
pente `0,724`, ordonnée `0,2648`, `r = 0,819`.

Trois lectures, toutes mesurées :

1. **Le décalage commun `−3,7e-4` s'annule dans les comparaisons appariées.** Il reflète la
   difficulté de la tranche publique, pas une faiblesse de modèle.
2. **La dispersion résiduelle par soumission est `1,21e-4`**, soit un écart-type apparié public
   de `1,72e-4`. Sur le privé (239 875 lignes, 4× le public), si cette dispersion est dominée
   par l'échantillonnage, elle tombe à `0,86e-4`.
3. **La pente `0,72 < 1` et `r = 0,82`** signifient que l'OOF n'explique qu'environ 67 % de la
   variance des scores publics de cette cohorte. La bande qui sépare le top public
   (`0,96159`) de v7 (`0,96146`) fait `1,3e-4` : elle est du même ordre que le bruit. Deux
   exemples extrêmes le montrent : `endgame_08` (OOF `0,961194`) a fini à `0,96109`
   (`+2,65e-4` au-dessus de la tendance), tandis que `v2_equal34` (OOF `0,961438`) a fini à
   `0,96092` (`−1,49e-4`). Des modèles médiocres en OOF peuvent donc se retrouver à `1,7e-4`
   l'un de l'autre sur le public.

## Arithmétique de `P(v6 > v7)`

Écart OOF : `Δ = v7 − v6 = +8,73e-5`. Écart public : `+2,0e-5`. Le transfert public de v6 est
donc **meilleur de `6,73e-5`** que ne le prédirait l'OOF, et v6 a le deuxième meilleur résidu
de la cohorte (`+4,33e-5`).

Trois modèles de transfert, du plus conservateur au plus favorable, avec l'écart-type apparié
privé `0,858e-4` :

| modèle | écart privé attendu | `P(v6 > v7)` |
|---|---|---|
| N — l'écart public est du bruit pur | `+8,73e-5` | **15,4 %** |
| S — rétraction à mi-vie de l'écart public | `+5,40e-5` | **26,6 %** |
| B — l'écart public est réel | `+2,00e-5` | **40,8 %** |

Si la dispersion résiduelle était entièrement systématique et ne se réduisait pas avec la
taille de l'échantillon privé, l'écart-type apparié resterait `1,72e-4` et les trois
probabilités monteraient à `30,5 %`, `37,8 %` et `45,4 %`. Le chiffre conservateur à retenir
est donc **≈ 15 %**, le chiffre central **≈ 27 %**.

Pour v4, le même calcul donne `Δ_OOF = 3,44e-4` et un résidu `−1,40e-4` (transfert *moins* bon que
la tendance), donc `e_v7 − e_v4 = +1,16e-4` : l'écart privé attendu vaut `3,44e-4 + 1,16e-4 =
4,60e-4` si le résidu public est réel, et `3,44e-4` s'il est du bruit pur. Avec un écart-type
apparié de `0,858e-4`, `P(v4 > v7)` est compris entre `3e-5` (bruit pur) et `4e-8` (résidu
réel) — dans tous les cas négligeable. v4 n'est donc **pas** un candidat de score ; c'est
uniquement une police d'assurance contre une défaillance structurelle du pipeline TabPFN (v6
partage le parent aux10 et une route de la même famille, v4 est un blend classique de 59
membres). Si l'utilisateur préfère cette couverture de queue au gain marginal de v6, v4 est
déjà uploadé et peut être sélectionné à la place.

## Pourquoi pas un troisième changement

- La géométrie de composition est optimale mesurée (`+1,69e-6` au mieux, t 0,95) : il n'y a rien
  à ajuster côté combinateur.
- Les 10 membres aux10 sont à `2,4e-6` de l'optimum : rien à re-pondérer.
- Le store de 133 vecteurs offre au mieux `+1,54e-5` avec sélection rétrospective, c'est-à-dire
  du bruit.
- Les gains de parent se diluent à 7,1 % : il faudrait `+2e-4` au niveau d'un parent pour
  espérer `+1,5e-5` au portefeuille.
- Aller chercher un nouveau modèle pour gagner `1e-4` d'OOF serait de toute façon illisible :
  la pente `0,72` dit qu'un gain OOF de `1e-4` ne produit qu'environ `7e-5` de gain public
  attendu, pour un bruit apparié de `1,7e-4`.

## Action utilisateur (chemin critique)

1. Se connecter à Kaggle, ouvrir la compétition `playground-series-s6e10`, onglet
   *My Submissions*.
2. Vérifier que **57090508** (v7) et **56951906** (v6) sont tous deux `complete`.
3. Les marquer tous les deux comme *final submissions* (2 slots autorisés).
4. Relire la page *Rules* et confirmer que la règle « meilleure des deux » n'a pas changé.
5. Rapporter les deux références finales dans ce document.

**Échéance : 2026-10-31T23:59Z.** Ne pas attendre le dernier jour : la sélection est une action
d'interface, pas une soumission, mais elle doit être faite avant la fermeture.
