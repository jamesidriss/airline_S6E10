# Phase 25b — reprise du 2026-10-24 : ce qui reste vraiment ouvert

Suite de `SCREEN_AND_POOL.md`. Question posée : existe-t-il encore un levier **mesurable** de
plus de ~1e-4 AUC, ou la frontière 0,9622 est-elle une propriété de la fonction d'étiquetage ?

---

## 1. Validation adversariale — pas de décalage de distribution

Un modèle qui distingue une ligne de train d'une ligne de test sur les 21 covariables seules :

| test | valeur | lecture |
|---|---|---|
| AUC adversariale (LGBM 300 arbres, 5 folds) | **0,49979** | indistinguable de 0,5 |
| dérive des moyennes de features sur 14 tranches d'id | `Age 38,85–39,07` · `FD 1345,9–1361,3` | dans le bruit |
| taux de positifs par tranche d'id | `0,4406–0,4476` | bruit binomial (`±0,0022`) |

Conclusion : train et test sont des tirages **i.i.d.** du même générateur. Il n'y a donc **aucun
levier de repondération** : toutes les lignes d'entraînement sont également pertinentes pour le
test, et l'astuce classique « pondérer les lignes proches du test » ne peut rien apporter ici.

## 2. Fuite par l'ordre des ids — fermée

Les ids forment un flux contigu unique (train `0…699 634`, test `699 635…999 478`), donc le
split est un découpage en blocs d'une séquence générée : c'est exactement la configuration où
un effet de lot se traduirait en « fuite » dans un OOF à folds aléatoires.

| test | AUC | z (SD nulle 6,95e-4) |
|---|---|---|
| `id` seul comme score | 0,50007 | 0,1 |
| GBDT sur 9 features d'id, folds `primary` | 0,50164 | 2,4 |
| GBDT sur 9 features d'id, folds **contigus** (`blocked`) | 0,50137 | 2,0 |

Un AUC de 0,5016 sur 700 000 lignes vaut `1,6e-3` : même si c'était réel, ce serait 60 fois
trop petit pour être exploitable, et le schéma contigu donne le même chiffre. **Aucune
information d'id à récolter.**

## 3. Courbe d'apprentissage — le modèle de base est encore loin de saturer

Un seul fold (fold 0), même jeu d'évaluation pour les trois points donc comparaison appariée,
sous-ensembles imbriqués des lignes d'entraînement, 2 graines par point, config identique au
baseline stocké (`full`, 285 colonnes, `extra_trees`, 127 feuilles, lr 0,02) :

| n lignes d'entraînement | AUC fold 0 (moyenne 2 graines) | Δ par doublement |
|---|---|---|
| 139 927 | 0,960244 | — |
| 279 854 | 0,960695 | **+4,51e-4** |
| 559 708 | **0,961198** | **+5,03e-4** |

Les deux graines donnent `0,960401/0,960087`, `0,960677/0,960713`, `0,961174/0,961221` : le
bruit de graine est de `±1,5e-4`, le doublement vaut `5e-4`. **La pente ne s'aplatit pas** au
dernier doublement — un modèle unique sur 700 000 lignes a encore ~5e-4 d'erreur d'estimation
à gagner par doublement de données.

Lecture décisive : notre ensemble X vaut `0,962282` sur ce même fold 0, soit **+1,08e-3** au-dessus
du modèle unique à 560 000 lignes. En unités de la courbe, c'est **2,2 doublements**, donc un
ensemble équivalent à un modèle unique entraîné sur ~**2,6 millions de lignes**. L'ensemble a
déjà acheté tout le « plus de données » disponible en moyennant des estimateurs décorrélés.

Conséquence : l'erreur restante n'est plus de l'erreur d'estimation *séparable* — c'est le
résidu **partagé** par tous les estimateurs entraînés sur les mêmes 700 000 lignes, plus le
bruit de Bernoulli des étiquettes. C'est exactement pourquoi un membre à `0,9612` et `ρ 0,98`
(valeur isolée honorable, +5e-4 de marge d'estimation) apporte **zéro** en blend : son erreur
recouvre celle qu'on a déjà. Aucun membre à `ρ < 0,98` avec X n'a jamais dépassé `0,9614`
d'AUC, et tous les membres à `0,9622` sont à `ρ ≥ 0,999` de X : il n'existe pas, dans les 368
vecteurs mesurés, de membre à la fois **fort** et **décorrélé**.


## 4. Sélection gloutonne (Caruana) sur 368 vecteurs — 2,3e-5 sous X

Le stack logistique est un combineur paramétrique ; la sélection gloutonne avec remise est le
combineur non paramétrique que les blends publics utilisent réellement, et elle ne peut pas
produire de poids négatif. Si la frontière 0,9622 était un artefact de géométrie, c'est ce
combineur qui l'aurait montré.

Pool : 288 vecteurs publics + 80 vecteurs maison = **368 candidats**, 40 tours, 150 candidats
retenus par fold (triés sur 200 000 lignes d'entraînement), poids appris sur les 4 folds
d'entraînement et appliqués au fold tenu à l'écart.

| vecteur | AUC moyenne des folds | vs X | folds |
|---|---|---|---|
| **X** (0,85 nj05 + 0,15 v7) | **0,962249** | — | — |
| glouton 40 tours | 0,962226 | **−2,31e-5** | **0/5** |
| v7 | 0,961861 | −3,88e-4 | 0/5 |

Le glouton est **positif sur les 5 folds contre v7** (`+3,65e-4`, t 13,8) et **négatif sur les
5 folds contre X** (`−2,31e-5`, t 6,4). Il converge donc vers nj05 + stack245 (0,35–0,65 de
poids selon le fold) et échoue exactement là où le stack logistique échouait : il n'y a rien
au-dessus de nj05 dans le pool, et diluer nj05 avec ses quasi-copies coûte 2e-5.

**Note de méthode** : les poids étant propres à chaque fold, les prédictions des folds ne sont
pas sur la même échelle et l'AUC *pooled* (0,961793) mélange des offsets — elle sous-estime le
combineur de 4,3e-4. Le chiffre honnête est la moyenne des AUC par fold, et elle est utilisée
ci-dessus.

## 5. Deuxième slot : ce qu'il rapporte réellement

Seule la meilleure des deux soumissions sélectionnées est jugée, donc le second slot vaut
`E[max(AUC_A, AUC_B)] − AUC_A`, pas `AUC_B − AUC_A`. Bootstrap apparié de 400 rééchantillons à
la taille exacte du privé (239 875 lignes), sur les OOF des folds immuables :

| second slot | OOF | ρ Spearman avec X | P(dépasse X sur le privé) | E[gain du max] |
|---|---|---|---|---|
| **v7 (interne)** | 0,961853 | 0,9920 | **0,00** | **0,000e-4** |
| `najiama_05` | 0,962234 | 0,9998 | 0,06 | +0,002e-4 |
| `goodpjw200_stack245` | 0,962224 | 0,9995 | 0,04 | +0,002e-4 |
| `najiama_06` | 0,962211 | 0,9994 | 0,00 | 0,000e-4 |

Tous les candidats sont à `ρ ≥ 0,9994` de X : le second slot vaut au mieux **2e-6 d'AUC
espérée**, soit 1/150 de la barre de bruit public et 1/200 de l'écart-type du privé. Le choix
est donc **indifférent en espérance**, et le seul critère qui reste est la queue de risque :
v7 est le seul vecteur produit par un pipeline entièrement interne, donc la seule assurance
contre un défaut systémique du côté externe. On garde **X + v7**.

Argument supplémentaire, mesuré : le score public de X (0,961840) a déjà dé-risqué le vecteur
externe sur 59 969 lignes jamais vues. Un défaut qui ne se verrait ni dans l'OOF ni sur le
public devrait être spécifique aux 239 875 lignes privées — c'est-à-dire ne pas exister.


## 6. TabPFN natif raw8 — lancé, mesuré, et fermé par la structure

Demandé le 2026-10-24. Le protocole phase 23 (`configs/phase23/raw8_kaggle_prereg.json`) demande
8 unités (`n_estimators=8`) sur 559 708 lignes de contexte, soit ~19 h de GPU. Sur une carte de
15,9 Gio les 8 caches ne peuvent pas coexister : la recette est réalisée en **8 unités
séquentielles**, un processus chacune, agrégées en logit moyen — exactement la géométrie du
kernel Kaggle (son répertoire `units/`). J'ai lancé la première unité de mesure (fold 0, raw,
graine 1202).

### 6.1 Le fait déjà mesuré : la moyenne d'unités vaut +1,54e-4

Le membre TabPFN de v7 est lui-même un **2 unités**. Les artefacts certifiés
`artifacts/phase17_B1/f{0..4}/member_{0,1}_raw_logits.npy` permettent de mesurer directement ce
que la moyenne apporte, sans un seul GPU :

| fold | unité 0 | unité 1 | moyenne des 2 | gain vs meilleure unité |
|---|---|---|---|---|
| 0 | 0,961132 | 0,961330 | 0,961458 | +1,27e-4 |
| 1 | 0,961379 | 0,961432 | 0,961616 | +1,84e-4 |
| 2 | 0,961200 | 0,961417 | 0,961530 | +1,13e-4 |
| 3 | 0,960608 | 0,960687 | 0,960886 | +1,99e-4 |
| 4 | 0,961819 | 0,961988 | 0,962135 | +1,47e-4 |

**+1,54e-4 en moyenne, 5/5 folds**, écart entre unités `1,43e-4`. La loi de réduction de
variance `gain(n) = V·(1 − 1/n)` donne `V = 3,08e-4`, donc :

| étape | gain sur le membre |
|---|---|
| 1 → 2 unités (déjà fait dans v7) | +1,54e-4 |
| 2 → 8 unités | **+1,16e-4** |
| 1 → 8 unités | +2,70e-4 |

### 6.2 Pourquoi cela ne peut pas déplacer X

Vérifié numériquement sur les 5 folds : **`v7` est exactement le blend équipondéré de 60
membres** (`max|v7 − portfolio| = 0,0` sur les 5 folds), et `X = 0,85·nj05 + 0,15·v7`. Le
membre TabPFN y porte donc le poids `1/60`, et v7 le poids `0,15` dans X :

```
gain sur X = gain du membre × (1/60) × 0,15 = gain du membre / 400
```

| scénario | gain sur le membre | gain sur X |
|---|---|---|
| 2 → 8 unités (le protocole complet) | +1,16e-4 | **+2,9e-7** |
| 1 → 8 unités (borne haute absolue) | +2,70e-4 | **+6,8e-7** |
| membre neuf ajouté à v7 à w=0,10 (référence externe `tabpfn35`) | +2,99e-5 sur v7 | **+4,5e-6** |

La barre de soumission est `+3e-4` : le protocole complet en vaut **1/1000**, sa borne haute
absolue **1/440**, et le meilleur scénario jamais mesuré pour un TabPFN raw8 **1/67**. Ce n'est
pas un problème de bruit de mesure : c'est un facteur de dilution structurel. Les 40 unités
(~17 h) sont donc arrêtées après la mesure de la loi, pas après une intuition.

### 6.3 Note d'exécution

Deux échecs avant la bonne configuration, tous deux dus à la mémoire, aucun aux données :

1. `fit` OOM à `11,99 Gio alloués + 1,07 Gio demandés` avec `--gpu-fraction 0.85` (plafond
   13,53 Gio) : le plafond, pas la carte, était la contrainte.
2. Relance avec `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` et `--gpu-fraction 0.95`
   (15,1 Gio autorisés ; 14,7 Gio réellement libres, ~1,2 Gio pris par le bureau).

<!-- SECTION_RAW_PROBE -->

### 6.4 La mesure directe (fold 0, raw, 818 s de fit, 13,80 Gio de pic)

Une unité raw de plus (graine 1202) : AUC `0,960990` contre `0,960972` pour l'unité 1201 — les
deux unités sont à `1,7e-5` l'une de l'autre, donc *a priori* indiscernables. Pourtant :

| agrégat | AUC fold 0 | gain |
|---|---|---|
| unité 1201 seule | 0,960972 | — |
| unité 1202 seule | 0,960990 | — |
| **moyenne logit des 2** | **0,961213** | **+2,41e-4** (bootstrap : sd 0,60e-4, `P>0 = 1,00`) |

Deux unités de qualité quasi identique se moyennent en `+2,41e-4` : les erreurs des unités sont
fortement décorrélées même quand leurs AUC ne diffèrent pas. La loi de variance donne alors
`V = 4,82e-4`, donc **2 → 8 unités = +1,81e-4 sur le membre**.

Et voici le chiffre qui tranche, mesuré et non extrapolé — le membre 2-unités ajouté à v7 :

| poids dans v7 | Δ AUC de v7 (fold 0) |
|---|---|
| 0,05 | **+3,78e-6** |
| 0,10 | +1,46e-6 |
| 0,15 | +0,44e-6 |
| 0,25 | −0,82e-6 |
| 0,50 | −3,44e-6 |

Le membre est déjà dans v7 sous forme de sa version route (`ρ logit 0,9947`) : il n'apporte que
**+3,8e-6 à v7**, soit `0,15 × 3,8e-6 =` **+5,7e-7 sur X**. Même en supposant que les 8 unités
ajoutent leurs `+1,81e-4` au membre (elles ne les ajouteront pas toutes, la dérivée diminue),
X gagne **moins de 1e-6**.

Conclusion chiffrée : le protocole complet (40 unités, ~17 h) vaut **≤ 1e-6 sur X**, contre une
barre de soumission de `3e-4` — un facteur **1/300**. La dilution structurelle (`1/400` par
construction) et la mesure directe (`+5,7e-7`) donnent le même ordre. L'axe est fermé.


## 7. Ce qui est fermé, définitivement

| axe | mesure | verdict |
|---|---|---|
| pool OOF public (288 vecteurs) | stack imbriqué `0,962201` vs nj05, 0/5 folds | fermé |
| géométries de blend | poids plat 0,80–0,90, `0,962247` sur primary/shadow/block10 | fermé |
| blocs de features | `full_tec` `+1,24e-4` (t 3,76), `full_enrich` `+6,6e-5` (t 2,68) → **0** en admission | fermé |
| lignes originales ajoutées à l'entraînement | `0,961078` (w=0) → `0,960493` (w=0,3) → `0,959975` (w=1) | fermé, monotone négatif |
| pseudo-étiquetage | effet négatif le plus fort de la campagne (~30× son contrôle) | fermé |
| décalage de distribution / fuite d'id | voir §1 et §2 | fermé |
| stacking de nos 60–80 vecteurs | `0,961786`, `−6,7e-5` vs v7 | fermé |
