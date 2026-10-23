# Phase 25 — reprise : envoi de X, pool externe épuisé, écran de features

**Ouverture** : 2026-10-23T20:00Z · **Échéance** : 2026-10-31T23:59Z
**Objectif utilisateur** : gagner le privé, pas seulement le public.

---

## 1. Envoi du candidat X — fait, accepté

| élément | valeur |
|---|---|
| fichier | `artifacts/phase24_candidateX/candidate_x.csv` |
| réf. Kaggle | **57570936** |
| score public observé | **0,961840** |
| score public prédit (règle gelée) | 0,961878 |
| écart | `−0,38e-4` (`0,3 σ` du résidu mesuré `1,21e-4`) |
| règle gelée | `≥ 0,96170` → **accepté** |
| décision | **finaliste A = X (57570936)**, finaliste B = v7 (57090508) |

X bat v7 de **+3,8e-4 sur le public** (0,96184 vs 0,96146), soit exactement le delta OOF
`+3,886e-4`. Le transfert OOF→public est donc confirmé sur ce vecteur : l'avantage de `nj05`
n'est pas un artefact de protocole OOF (arrêt précoce sur le fold de validation), il se
reproduit sur 59 969 lignes jamais vues.

Budget consommé : **1/10** soumissions le 2026-10-23.

---

## 2. Pool OOF externe — 288 vecteurs, aucun admissible

Neuf bibliothèques publiques ont été téléchargées (hors dépôt, dans
`%TEMP%\opencode\s6e10_oof_pool`) : `cdeotte` (124 membres ×2), `sachith7`, `koumeimaki`,
`kagankoral`, `kratosyan`, `goodpjw2008`, `megayak`, `najiama`. `dariushafshar/golem` est
introuvable (404). Après alignement prouvé par AUC sur nos folds (une erreur d'ordre
donnerait 0,5 ; une colonne de label donnerait 1,0 — les deux sont rejetées) :

- **288 vecteurs OOF** alignés, dont **252** avec leur prédiction test appariée ;
- meilleur membre seul : `najiama_05_blend` = **0,962235** (identique à notre mesure de nj05) ;
- `goodpjw200_stack245` = 0,962224 ; `najiama_06_blend` = 0,962212 ;
- les membres de base (hors stacks pré-faits) plafonnent à **0,9614** ; moyenne = 0,96105.

Trois tests, tous négatifs :

| test | résultat | verdict |
|---|---|---|
| stack logistique imbriqué, top-120 (rang et coefficients appris hors fold) | **0,962201** | `−3,3e-5` vs nj05, **5/5 folds négatifs** |
| idem + 25 vecteurs de notre store (145 membres) | 0,962206 | `−2,8e-5` vs nj05, 4/5 négatifs |
| écran pairwise : X + chaque membre, poids logistique appris sur 4 folds | meilleur `+4,5e-6`, 3/5, poids **négatif** | **0 admissible** (`n_pos5 = 3`, `n_gate = 0`) |

Moyennes équipondérées : top-5 `0,962204`, top-20 `0,962049`, top-288 `0,961765` — plus on
ajoute de membres, plus on perd. La frontière du pool est **0,9622**, exactement là où nous
sommes. Le stack externe est donc **fait et fermé**, pas « à faire » : il ne reste aucune
information à extraire de cette direction.

Conséquence : `najiama_05` n'est pas améliorable par une meilleure géométrie de blend. X
(`0,85 nj05 + 0,15 v7`) reste le meilleur vecteur mesuré.

---

## 3. Écran de features — deux blocs franchissent la porte, aucun n'est actionnable

### 3.1 Écran 5 folds apparié (lancé le 2026-10-23)

Baseline `abl_lgbm_full_primary` : `full`, 285 colonnes, `extra_trees`, 127 feuilles,
lr 0,02, seed 1, arrêt précoce sur split interne aux lignes d'entraînement.

| vue | colonnes | OOF | Δ moyen apparié | SE | t | folds | porte |
|---|---|---|---|---|---|---|---|
| `full_tec` (`full` + `te_cond`) | 357 | **0,961199** | **+1,244e-4** | 0,331e-4 | **+3,76** | **5/5** | **franchie** |
| `full_all21te` (`te_all21` remplace `te`) | 258 | 0,961058 | −0,136e-4 | 0,607e-4 | −0,22 | 2/5 | échouée |

Deltas par fold de `full_tec` : `+12,5`, `+8,6`, `+24,7`, `+11,3`, `+5,2` (×1e-5). C'est le plus
gros gain de bloc de features jamais mesuré dans le dépôt (hors `external`/`ogte`), et le
premier qui franchit la porte avec `t > 3`.

### 3.2 Mais aucun des deux ne complète l'ensemble

Test d'admission direct, poids logistique appris sur 4 folds et appliqué au 5e :

| base | candidat | OOF du blend | Δ moyen | t | folds | poids appris |
|---|---|---|---|---|---|---|
| X | membre `full_tec` (ρ Spearman `0,9802`) | 0,962243 | −0,03e-4 | −1,57 | 1/5 | **−0,052** |
| v7 | membre `full_tec` | 0,961846 | −0,04e-4 | −2,88 | **0/5** | −0,042 |
| X | membre `full_enrich` | 0,962245 | +0,01e-4 | +0,24 | 3/5 | −0,052 |

Le poids optimal est **négatif** dans les trois cas : le membre sert de terme correctif, pas de
signal, et le gain net est nul. C'est la signature d'un modèle qui estime *la même fonction*
un peu moins bien : l'ensemble l'a déjà. `te_cond` améliore donc un GBDT isolé de
`+1,24e-4` sans ajouter une seule unité d'information nouvelle.

### 3.3 `enrich` : réel, mais seulement sous `extra_trees`

Le seul bloc de features jamais mesuré positif est `enrich` (colonnes label-free
supplémentaires, construites pour l'interaction `extra_trees` × nombre de colonnes). Mesure
appariée sur les folds immuables, configuration identique (`extra_trees`, 127 feuilles,
lr 0,02, seed 1) :

| vue | Δ vs `full` | SE | t | folds |
|---|---|---|---|---|
| `full_enrich` (`extra_trees`) | **+6,61e-5** | 2,47e-5 | **+2,68** | **5/5** |
| `full_enrich` (déterministe) | −1,10e-4 | 4,67e-5 | −2,35 | 1/5 |
| `full_ogsurf` | +5,85e-5 | 6,78e-5 | +0,86 | 2/5 |
| `full_ogm_ogs` | +3,98e-5 | 6,20e-5 | +0,64 | 3/5 |
| `full_ogm` | −1,36e-4 | 5,32e-5 | −2,55 | 1/5 |

`full_enrich` franchit la porte d'admission (`+6,6e-5`, `t 2,68`, 5/5) **et** le mécanisme est
confirmé par le signe : le gain disparaît et s'inverse sans `extra_trees`. C'est la
définition même de l'interaction documentée dans `AGENTS.md`. `full_tec` (`+1,24e-4`,
`t 3,76`) est plus fort encore, mais aucun des deux ne survit au test d'admission contre
l'ensemble (§3.2).

**Pourquoi cela ne change pas le finaliste** : le gain porte sur un GBDT isolé à 0,96115. Le
finaliste X est à 85 % un blend externe ; même si `v7` gagnait 6,6e-5, X ne gagnerait que
`0,15 × 6,6e-5 ≈ 1e-5`, soit un quart de la barre de bruit. `enrich` reste donc un résultat
scientifique propre, pas un levier de classement.

---

## 4. Où en est la probabilité de victoire

Tout converge vers une seule lecture : la frontière honnête est **0,9622**, nous sommes à
**0,962247** (X), et aucun des 288 vecteurs publics, aucune géométrie de blend et aucun bloc
de features ne produit de gain admissible. Le privé (239 875 lignes, `SE ≈ 4,2e-4`) décidera
entre des modèles qui mesurent la même fonction à ~2e-5 près.

Ce n'est pas un échec de méthode : c'est le régime où le générateur a lissé la fonction
d'étiquetage (source 0,9888 → synthétique 0,9253 sur profil 16 clés). La victoire est un
tirage parmi une dizaine d'équipes à égalité, avec un avantage réel mais faible pour nous
(+3,8e-4 public mesuré sur v7).

---

## 5. État des finalistes

| slot | réf. | vecteur | public | OOF |
|---|---|---|---|---|
| A | **57570936** | X = 0,85 nj05 + 0,15 v7 | **0,96184** | 0,962247 |
| B | 57090508 | v7 (pipeline interne, aucun vecteur externe) | 0,96146 | 0,961853 |

B est conservé pour l'indépendance méthodologique : c'est la seule assurance contre un défaut
systématique du côté externe. Les deux slots restent à certifier dans l'UI Kaggle avant
2026-10-31T23:59Z.
