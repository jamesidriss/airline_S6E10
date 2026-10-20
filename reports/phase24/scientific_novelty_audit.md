# Phase 24 — audit de nouveauté scientifique

Date UTC : 2026-10-20. Portée : revue forensique des phases 17 à 23, statut réel du catalogue
de 96 items, et inventaire de ce qui était **réellement non mesuré** avant d'ouvrir la Phase 24.

## 1. Ce que la revue forensique établit

| phase | conclusion affichée | preuve réelle | statut scientifique |
|---|---|---|---|
| 17 | v7 certifié, OOF `0.9618529228157235` | 5 folds complets sur disque, hashes bit-exacts, recomposition `5.59e-8` | **mesuré, solide** |
| 18 | `internal2 → internal4` : route `+5.34e-5` sur fold 0, portefeuille `+3.818e-6` | audit fold 0 isolé, bras uniques tous perdants | **mesuré** ; conversion parent→portefeuille = 7,1 % |
| 19 | 85,3 % des paires manquées sont partagées ; masse d'agrégation `0.0055997` | atlas de 4,83e9 paires, identité de comptage vérifiée | **mesuré** |
| 20 | 96 items : 67 négatifs, 8 dépendances fermées, 5 partiels | catalogue + ledger | **hétérogène** (voir §2) |
| 21 | aucun challenger ; conserver v7 | réévaluation documentaire | **administratif** (aucune nouvelle mesure) |
| 22 | lean84 / own_raw / 24 combinaisons : 0 admission | 5 folds, ES interne, TE reconstruite | **mesuré** |
| 23 | raw8 = `RESOURCE_BLOCKED_HARDWARE` | OOM T4 mesuré (12,4 Gio KV + 2,14 Gio activation sur 14,56 Gio) | **mesuré, blocage matériel** ; hypothèse H23-01 non réfutée |
| 23 | structure id/générateur fermée négative | `AUC(id)=0.500073`, χ² p 0,975–1,000 | **mesuré** |

Aucune de ces clôtures n'est contredite par la Phase 24. En revanche, deux items du catalogue
étaient explicitement marqués `PARTIAL_VALID` et n'avaient jamais été exécutés.

## 2. Clôtures administratives versus scientifiques

Distribution du catalogue de 96 items (recomptée depuis le JSON) :

- statut : `COMPLETE_NEGATIVE` 67, `DEPENDENCY_CLOSED` 8, `PARTIAL_VALID` 5,
  `CONCEPTUAL_CLOSED` 5, `LICENSE_BLOCKED` 3, `RESOURCE_BLOCKED` 2,
  `HISTORICAL_EQUIVALENT` 2, `NOT_WARRANTED` 2, `LOW_VALUE_CLOSED` 1, `COMPLETE_CERTIFIED` 1 ;
- fidélité : **F0 23, F1 9**, F2 60, F3 2, F5 2.

Les 32 items F0/F1 sont les seuls où un échec peut être un artefact de fidélité. Répartition :

| catégorie | items | lecture correcte |
|---|---|---|
| `LICENSE_BLOCKED` | 3, 7, 8 | non testable, jamais un négatif |
| `RESOURCE_BLOCKED` | 12, 34 | non testable sur le matériel disponible |
| `DEPENDENCY_CLOSED` | 44, 70, 71, 73, 90, 92, 95 | fermé faute d'opérande, pas faute de résultat |
| `CONCEPTUAL_CLOSED` | 81, 82, 83, 86, 87 | fermé par argument, pas par mesure |
| `HISTORICAL_EQUIVALENT` | 43, 67 | renvoi à une mesure antérieure |
| `NOT_WARRANTED` | 89, 93 | non exécuté par choix |
| `LOW_VALUE_CLOSED` | 79 | non exécuté par choix |
| `COMPLETE_NEGATIVE` F0/F1 | 14, 27, 28, 29, 32, 33 | négatifs à faible fidélité |
| `PARTIAL_VALID` F1 | **80, 91** | **exécutés en Phase 24** |

Conséquence de méthode : les items 80 et 91 sont les deux seuls du catalogue à la fois F1 et
`PARTIAL_VALID`. Les traiter était donc le seul travail scientifique obligatoire restant, et il
ne nécessitait aucun GPU.

## 3. Items 80 et 91 exécutés

**#80 — estimation du plafond AUC (théorie vs mesure).** Aucune borne théorique n'est
calculable sans le générateur. Trois mesures ont été produites à la place :

1. ancre source (`reports/phase24/source_anchor.json`) : sur les 129 880 lignes d'origine, les
   mêmes 21 variables, un GBDT fort à 5 folds atteint `0.988806` ; à 64 940 lignes `0.987873`
   (gain de doublement `+9.33e-4`, donc la source est encore limitée par les données mais son
   plafond est très au-dessus de 0,988) ;
2. écart de déterminisme (`reports/phase24/determinism_compare.json`) : même clé de 16 colonnes,
   moyenne de groupe en leave-one-out — source `0.970995` sur 36 074 lignes non singletons,
   synthétique `0.925285` sur 395 171 lignes ; part de lignes en groupes purs (taille ≥ 2)
   `26,8 %` contre `30,8 %`, alors que la taille moyenne de groupe est `1,245` contre `1,812` ;
3. exhaustion du store (`reports/phase24/oof_inventory.json`, `frontier.json`) : 133 vecteurs,
   aucun delta fold moyen positif, meilleur gain de mélange avec sélection rétrospective
   `+1,5368e-5`.

La lecture est cohérente : la population synthétique **répète davantage de profils** que la
source mais **les étiquette beaucoup plus bruyamment**. Le générateur a lissé la fonction
d'étiquetage quasi déterministe de la source.

**#91 — variance de graine vs variance de sous-ensemble.** Séparées pour la première fois :

- composante graine, mesure directe sur fold 0 (`scripts/phase24/parent_member_diagnostics.py`) :
  route seed 1201 `0.9614577`, seed 1202 `0.9611372`, corrélation des logits `0,9973` ;
  l'écart entre graines est donc de `3,2e-4` en AUC, et la moyenne de deux graines vaut
  `0.9615469` (`+8,93e-5` sur le parent) ;
- composante sous-ensemble : les AUC par fold du champion s'étalent de `0.961099` à `0.962496`,
  soit `1,4e-3`, très au-dessus de la composante graine ;
- conséquence : l'erreur type d'un delta **apparié au niveau fold** est d'environ `1e-5`
  (mesurée sur les comparaisons du store), et c'est elle qui rend le seuil `+3e-4` de
  l'AGENTS.md à la fois prudent et coûteux.

## 4. Ce qui reste réellement non mesuré

Quatre mécanismes seulement, tous bloqués par autre chose qu'un résultat :

| mécanisme | blocage | condition de réouverture |
|---|---|---|
| raw8 TabPFN-3.5 contexte natif (H23-01) | T4 : 14,56 Gio < 12,4 Gio KV + 2,14 Gio activation | accélérateur ≥ 16 Gio utilisables |
| TabICLv2 vue riche (item 2) | 3 échecs ressource à 16 Gio | idem |
| RealTabPFN-2.5 (item 12) | RESOURCE_BLOCKED | ≥ 5 h GPU dédiées |
| fine-tune TabPFN 3.5 (item 34) | 6 sondes OOM, aucune prédiction fold 0 | idem |

Aucun de ces quatre ne peut être exécuté sous le plafond matériel actuel, et aucun n'a de
traduction portfolio favorable démontrée : la conversion parent→portefeuille mesurée est de
7,1 % (item #18 ci-dessus), donc un parent à `+1e-3` ne vaudrait que `+7e-5` au portefeuille.

## 5. Mécanismes ouverts et testés en Phase 24

Tous ont été préenregistrés avant résultat, exécutés sur CPU, sans GPU et sans soumission.

| id | mécanisme | résultat | verdict |
|---|---|---|---|
| H24-C1 | géométrie de composition à faible variance (7 règles) | meilleure règle `+1,69e-6`, t 0,95, 3/5 | falsifié |
| H24-C2 | gate de segment sur clé de profil peu fréquente | meilleur test `+2,80e-6`, t 0,85 ; poids optimal nul dans 11/16 | falsifié |
| H24-C3 | re-pondération des 10 membres aux10 + moyenne de graines route | moyenne aux10 optimale à `2,4e-6` ; gain graine dilué à 7 % | falsifié |
| H24-C4 | information nouvelle par doublons/clefs | 0 doublon exact train↔test, 0 dans chaque split | fermé |
| H24-C5 | census de l'espace résiduel du store | 133 vecteurs, plafond rétrospectif `+1,54e-5` | exhaustion |
| H24-C6 | déterminisme source vs synthétique | LOO `0,9710` vs `0,9253` | **mécanisme identifié** |

## 6. Conclusion de l'audit

La Phase 24 n'a pas trouvé de levier manqué par négligence : elle a trouvé que l'espace des
corrections **au niveau du score** est vide, et que l'espace des **nouvelles informations** est
contraint par les 21 colonnes. Le résultat nouveau est un mécanisme mesuré — la perte de
déterminisme introduite par le générateur — qui explique pourquoi 133 familles de modèles
plafonnent dans une bande de `3e-4` et pourquoi l'objectif `0,962740` n'est pas atteignable par
une meilleure modélisation de ces 21 colonnes.
