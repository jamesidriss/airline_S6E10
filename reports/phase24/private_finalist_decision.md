# Phase 24 — décision sur les finalistes privés

## Décision

| slot | candidat | réf. Kaggle | OOF | public | rôle |
|---|---|---|---|---|---|
| **A** | `v7_sol_tabpfn_internal2_aux10` | **57090508** | `0,9618529228157235` | `0,96146` | meilleur estimateur ponctuel, artefact certifié |
| **B** | `v6_sol_tabpfn_route_aux10` | **56951906** | `0,9617656138245796` | `0,96144` | assurance gratuite sous la règle du max |

Réserve documentée : `v4_fulldata`, réf. **56845975** (OOF `0,96150857858`, public `0,96100`),
uniquement comme couverture contre une défaillance *structurelle* de la famille TabPFN, pas
comme candidat de score.

## Pourquoi le slot B est gratuit

La Phase 23 a confirmé sur la page Rules que le score privé est le **maximum** des soumissions
sélectionnées (rapport `reports/phase23/finalist_B_analysis.md`, ligne 65). Sélectionner B ne
peut donc pas dégrader le score ; B n'a de valeur que si `P(B > A) > 0`. Il faut malgré tout
**relire la page Rules vers le 2026-10-30** : c'est la seule règle dont le changement
invaliderait cette décision.

## Table de transfert OOF → public (mesurée, 5 soumissions)

| soumission | OOF | public | écart public − OOF |
|---|---|---|---|
| v3_final | 0,961508579 | 0,96098 | `−5,29e-4` |
| v4_fulldata | 0,961508579 | 0,96100 | `−5,09e-4` |
| v5_aux_cross | 0,961523 | 0,96103 | `−4,93e-4` |
| v6_private_hedge | 0,961765614 | 0,96144 | `−3,26e-4` |
| v7_internal2_aux10 | 0,961852923 | 0,96146 | `−3,93e-4` |

Deux lectures, toutes deux utiles :

1. **Le décalage commun est de `−4,5e-4` environ** (moyenne des cinq, écart-type `0,83e-4`).
   Il est compatible avec le bruit d'échantillonnage de la tranche publique (59 969 lignes,
   erreur type autonome `1,5e-3`). Il s'annule dans les comparaisons appariées : ce qui compte
   est l'écart *entre* modèles.
2. **v3 et v4 ont le même vecteur OOF** (`0,961508579`) et des CSV qui ne diffèrent que par le
   re-fit sur toutes les données : leur écart public de `+2e-5` est une mesure directe du bruit
   public résiduel entre deux candidats quasi identiques. C'est l'échelle de résolution réelle
   de la place publique, cohérente avec le `±2e-4` apparié de l'AGENTS.md (ici deux vecteurs
   corrélés à `0,9999+`).

## Arithmétique de `P(v6 > v7)`

Écart OOF : `Δ = v7 − v6 = +8,73e-5`. Écart public : `+2,0e-5`. Le transfert public de v6 est
donc **meilleur de `6,7e-5`** que ne le prédirait l'OOF.

Trois modèles de transfert, du plus conservateur au plus favorable :

| modèle | écart privé attendu | écart-type privé | `P(v6 > v7)` |
|---|---|---|---|
| N — l'écart public est du bruit pur | `+8,73e-5` | `≈0,6e-4` | **≈ 7 %** |
| S — rétraction à mi-vie de l'écart public | `+5,4e-5` | `≈0,6e-4` | **≈ 18 %** |
| B — l'écart public est réel | `+2,0e-5` | `≈0,6e-4` | **≈ 37 %** |

Le privé compte 239 875 lignes, quatre fois le public : la composante d'échantillonnage de
l'écart décroît d'un facteur deux par rapport au public, ce qui est déjà incorporé ci-dessus.
Même sous le modèle le plus conservateur, le slot B vaut d'être occupé, puisqu'il est gratuit
sous la règle du max.

Pour v4, le même calcul donne `Δ_OOF = 3,44e-4`, un transfert public meilleur de `1,16e-4`,
soit un écart privé attendu de `2,3e-4` et `P(v4 > v7) ≈ 0,1 %`. v4 n'est donc **pas** un
candidat de score ; c'est uniquement une police d'assurance contre une défaillance
structurelle du pipeline TabPFN (même parent aux10, même famille de route pour v6, alors que v4
est un blend classique de 59 membres). Si l'utilisateur préfère cette couverture de queue au
gain marginal de v6, v4 est déjà uploadé et peut être sélectionné à la place — la décision est
consignée ici pour être réversible sans nouvelle expérience.

## Pourquoi pas un troisième changement

- La géométrie de composition est optimale mesurée (`+1,69e-6` au mieux, t 0,95) : il n'y a rien
  à ajuster côté combinateur.
- Les 10 membres aux10 sont à `2,4e-6` de l'optimum : rien à re-pondérer.
- Le store de 133 vecteurs offre au mieux `+1,54e-5` avec sélection rétrospective, c'est-à-dire
  du bruit.
- Les gains de parent se diluent à 7,1 % : il faudrait `+2e-4` au niveau d'un parent pour
  espérer `+1,5e-5` au portefeuille.

## Action utilisateur (chemin critique)

1. Se connecter à Kaggle, ouvrir la compétition `playground-series-s6e10`, onglet
   *My Submissions*.
2. Vérifier que **57090508** (v7) et **56951906** (v6) sont tous deux `complete`.
3. Les marquer tous les deux comme *final submissions* (2 slots autorisés).
4. Relire la page *Rules* et confirmer que la règle « meilleure des deux » n'a pas changé.
5. Rapporter les deux références finales dans `reports/phase24/private_finalist_decision.md`.

**Échéance : 2026-10-31T23:59Z.** Ne pas attendre le dernier jour : la sélection est une action
d'interface, pas une soumission, mais elle doit être faite avant la fermeture.
