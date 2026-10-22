# Phase 24 — décision de soumission, candidat X

```text
SUBMISSION_DECISION = SUBMIT_CANDIDATE_X_ONCE_AS_MEASUREMENT
                      STATUS: PENDING_USER_AUTHORISATION (Phase22 policy conflict)
frozen_utc          = 2026-10-23
prereg              = configs/phase24/candidate_x_prereg.json
```

## Arbitrage requis avant l'envoi

La Phase 22 avait classé les vecteurs externes « diagnostic seulement » et n'en a promu
aucun. Le candidat X contredit cette règle : il utilise comme composante un blend public
(`najiama/s6e10-oof`, fichier `05_blend`). Les raisons de le faire sont mesurées :

- le delta est mesuré **par nous**, sur nos folds immuables : `+3,886e-4`, t `13,38`, 5/5 ;
- il est répliqué sur `shadow` : `+3,944e-4`, t `9,48`, 5/5 ;
- le côté test est aligné au bit près (ids, ordre, distribution identique à l'OOF) ;
- le risque est borné : v7 reste le finaliste B, et le score privé est le **max** des deux ;
- le seul contrôle non encore dépensé est le score public, avec une règle d'acceptation
  gelée ci-dessous.

Deux options, à trancher par l'utilisateur :

1. **Autoriser un envoi de mesure** (1/10) — recommandé. Si le public tombe dans la bande
   d'acceptation, X devient A et v7 reste B ; sinon on ne sélectionne pas X et rien n'est perdu.
2. **Maintenir la règle Phase 22** — aucun envoi, finalistes inchangés A=v7 / B=v6, et le
   candidat X reste un résultat mesuré non déployé.

## Ce qui est soumis

```
X = sigmoid( 0.15 * logit(v7) + 0.85 * logit(nj05) )
```

| élément | valeur |
|---|---|
| OOF pooled | **0,9622466203** |
| delta vs champion v7 | **+3,886e-4** |
| primary (5 folds, seed 20261010) | t **13,38**, **5/5** folds positifs |
| shadow (5 folds, seed 777001) | delta **+3,944e-4**, t **9,48**, **5/5** |
| portes PATH A alternative | `≥1,5e-5`, `t≥2,5`, 5/5 → **franchies 26×** |
| barre routinière AGENTS.md | `≥3e-4` → **franchie** |
| lignes test | 299 844, ids exactement ceux de `test`, ordre identique |
| probabilités | min `0,005019`, max `0,997794`, moyenne `0,443673`, 0 NaN |
| corrélation logit test vs v7 | `0,9992566` |
| SHA tableau test | `7032b4d928911cdbe23eb76433e1e9b1c4d32d73fe10405cd6ed8df40f1427f1` |
| SHA CSV | `aca3e115568acebf8a07ccdcfb6c72bb201dc527b79fc47d61a6b051e8a579ea` |
| inputs | v7 OOF `bd4d288b…`, v7 test `e5b3b819…`, nj05 OOF `4aa93222…`, nj05 sub `a6d49839…` |

## Règle de validation, gelée avant le score public

Score public attendu : `0,9622466 − 3,689e-4 = 0,9618776` ; écart-type résiduel mesuré sur
17 soumissions `1,213e-4`.

| score public observé | action |
|---|---|
| `≥ 0,96170` | **accepté** → finaliste A = X, finaliste B = v7 (remplace v6) |
| `[0,96155 ; 0,96170[` | **limite** → finaliste B = X devant v6, v7 reste A |
| `< 0,96155` | **rejeté** → v7 A / v6 B, consigné comme négatif mesuré |
| `> 0,96250` | **suspect** (ajustement public spécifique) → pas de promotion sans argument privé |

Cette règle est prospective : elle est commitée **avant** que le score public n'existe. Le score
public ne servira jamais à régler un poids, un seuil ou un choix de modèle ; il sert uniquement
à accepter ou rejeter un vecteur déjà figé.

## Pourquoi ce n'est pas du public-LB fitting

1. Les poids sont **fixes** (`0,85`) et se situent sur le plateau optimal mesuré sur les folds
   d'entraînement (`0,825–0,850` dans les 5 folds). Aucun paramètre n'a été ajusté sur le public.
2. Le gain est mesuré sur nos folds immuables, **avant** toute soumission : `+3,886e-4` en
   primary et `+3,944e-4` en shadow, 5/5 dans les deux.
3. Le score public est une **mesure de cohérence** (le vecteur test est-il bien le même modèle
   que l'OOF ?), pas un objectif.
4. La porte d'acceptation est fixée à `0,96170`, soit `1,5 σ` sous la prédiction ; elle n'est pas
   déplacée après observation.

## Provenance

`nj05` provient du dataset public `najiama/s6e10-oof` (datasetId 12333973). Le champ licence est
`unknown`, mais la description du dataset accorde explicitement : « You are completely free to
use, reproduce, and include these OOF and submission CSV files in your own pipelines and
ensembles », avec demande de crédit. Crédit à najiama et aux notebooks cités (yekenot, ektarr,
goodpjw2008, evgendvorkin, kospintr, cdeotte) dans le rapport final.

## Budget

| poste | avant | après |
|---|---|---|
| soumissions du jour | 0/10 | 1/10 |
| GPU | 0 h | 0 h |
| slots finalistes | A=v7, B=v6 | à revoir selon le score public |

## Repli

- Si rejet : `A = v7 (57090508)`, `B = v6 (56951906)` — inchangé, aucune perte.
- Si acceptation : `A = X`, `B = v7` — deux méthodologies distinctes, la meilleure des deux
  compte pour le privé.
