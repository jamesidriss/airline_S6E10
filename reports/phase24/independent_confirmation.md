# Phase 24 — confirmation indépendante

## Statut : **NON OUVERTE**, par application du protocole gelé

Le protocole de Phase 24 (et celui de la Phase 23 avant lui) impose un ordre strict :

1. découverte sur les folds `primary` (seed 20261010) ;
2. si et seulement si la porte de découverte est franchie, **gel** de la configuration
   gagnante (fichier de config, SHA git, hashes de données et de folds) ;
3. exécution sur les folds `shadow` (seed 777001), folds 2 et 3 seulement, sans aucun
   réglage.

Aucune hypothèse de Phase 24 n'a franchi l'étape 1. La condition d'ouverture n'a donc jamais
été satisfaite, et ouvrir `shadow` aurait constitué une violation du protocole (usage de
l'échantillon de confirmation pour *chercher* plutôt que pour *confirmer*).

## Ce qui aurait été confirmé, et avec quelle puissance

| hypothèse | delta découverte | t | folds positifs | ouverture shadow ? |
|---|---|---|---|---|
| H24-C1 composition (C1) | `+1,694e-6` | 0,95 | 3/5 | non |
| H24-C2 segment (tabm/S2) | `+2,799e-6` | 0,85 | 3/5 | non |
| H24-C3 re-pondération aux10 | `+2,379e-6` | — | — | non |

Pour référence, la plus petite différence détectable avec les folds `shadow` disponibles
(2 folds, environ 280 000 lignes) à une puissance de 80 % et α = 0,05 bilatéral est d'environ
`4e-5` en AUC appariée. Les effets mesurés ci-dessus sont 15 à 20 fois plus petits : même
ouverte, la confirmation n'aurait pas eu la puissance de distinguer ces candidats du bruit.

## Ce que la Phase 24 confirme indirectement, sans ouvrir `shadow`

Trois résultats de la Phase 24 sont **des réplications internes** et non des découvertes :

1. l'identité de composition : `sigmoid(0,5·logit(route)+0,5·logit(aux10))` reproduit le
   portefeuille certifié à `5,59e-8` et la prédiction test certifiée à `5,16e-8` — la chaîne
   d'artefacts est intacte ;
2. l'absence de tout vecteur stocké avec un delta fold moyen positif contre v7 (133 vecteurs,
   0 positif) reproduit et étend le constat de la Phase 21 ;
3. la moyenne aux10 est optimale à `2,4e-6` près, ce qui reproduit la conclusion de la Phase 18
   selon laquelle le membre TabPFN porte le portefeuille et les GBDT le régularisent.

## Condition exacte de réouverture

`shadow` ne sera ouvert que si un candidat mesure, sur `primary`, un delta fold moyen
`>= +1,5e-5` avec `5/5` folds positifs et `t >= 2,5`. Un tel candidat n'existe pas dans
l'inventaire actuel. Si un run GPU devenait possible (accélérateur `>= 16 Gio`), le protocole
`configs/phase23/raw8_kaggle_prereg.json` prévoit déjà l'ordre découverte → gel → `shadow`
sans réglage.
