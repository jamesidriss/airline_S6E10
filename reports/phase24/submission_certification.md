# Phase 24 — certification de soumission

## Décision prospective

```
SUBMISSION_DECISION = NO_UPLOAD_IN_PHASE_24
```

Aucun candidat de la Phase 24 n'a franchi la porte PATH A (alternative `+1,5e-5`, 5/5 folds,
`t >= 2,5`) ni PATH B (`rho < 0,99` contre v7 avec un score autonome au niveau du champion).
Aucun envoi n'a donc été effectué, et le budget de soumissions du jour est resté intact
(`0/10` utilisées ; `0` soumission sur toute la Phase 24).

## Vérification du champion, refaite aujourd'hui (2026-10-20T07:57Z)

Commande : `scripts/phase24/certify_champion_now.py` (lecture seule).
Sortie complète : `reports/phase24/champion_reverification.json`.

| artefact | chemin | SHA-256 | contrôle |
|---|---|---|---|
| OOF champion | `artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy` | fichier `bd4d288b3e980b41f74c95a319eaa2be5dd5ea0e18fedd49718e0d0868b0fe5c` ; tableau `1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552` | AUC recalculée `0,9618529228157235`, 699 635 lignes |
| OOF portefeuille Phase 17 | `artifacts/phase17_B1/primary/portfolio_oof.npy` | identique au vecteur stocké (bit-égal) | confirme l'identité de composition |
| prédiction test certifiée | `artifacts/v7_sol_tabpfn_internal2_aux10_test/candidate.npy` | fichier `e5b3b819a4986cf3ba8239a67cddc03c6f187c196d7ddddb6367a0dced67d6b9` ; tableau `2b744d0df36a0b1143a92e1db18661ccd94fa881490784308624aaa0a0fc95af` | 299 844 lignes |
| recomposition depuis les deux parents test | `route.npy`, `aux10.npy` | — | écart max `5,163e-8` contre la prédiction certifiée |
| CSV soumis | `submissions/v7_sol_tabpfn_internal2_aux10.csv` | `fa4947fe9fc381fe85b105c1c489375327f7bb2321e4ff7f0d9ffc7cd7c9c0bd` | 299 844 lignes, colonnes `id,satisfaction`, ids identiques à `sample_submission.csv`, probabilités dans `[0,008389 ; 0,996948]`, écart max `1,11e-16` contre `candidate.npy` |

Référence Kaggle : **57090508**, score public `0,96146`, statut `complete`.

## Contrôles pré-vol exigés par l'AGENTS.md

| contrôle | état |
|---|---|
| OOF complet 5 folds | ✅ recalculé (`0,9618529228157235`) |
| AUC recalculée depuis le vecteur sur disque | ✅ identique au certificat |
| confirmation indépendante | ⛔ non ouverte, aucun candidat n'ayant passé la découverte (voir `independent_confirmation.md`) |
| provenance / licence des données | ✅ compétition Kaggle + dataset d'origine public ; aucun modèle à licence non commerciale n'entre dans v7 |
| prédictions test certifiées | ✅ SHA-256 ci-dessus |
| ids et ordre exacts | ✅ identiques à `sample_submission.csv` |
| enregistrements SHA-256 | ✅ dans ce document et dans `champion_reverification.json` |
| tests du dépôt | ✅ exécutés avant commit (voir `FINAL_REPORT.md`) |
| scan de secrets | ✅ aucun secret dans les fichiers ajoutés |
| quota disponible | ✅ `0/10` utilisées aujourd'hui |
| `SUBMISSION_DECISION` commité avant envoi | ✅ décision `NO_UPLOAD`, aucun envoi |

## Ce qui n'a **pas** été envoyé, et pourquoi

Les 23 candidats historiques en inventaire (`endgame_01..10`, `cand_family_balanced`, etc.)
restent exclus par l'amendement PATH A/B : aucun n'a de porte franchie, et plusieurs ont un OOF
inférieur de `1e-4` à `7e-4` au champion. Les 7 soumissions de la Phase 16 et antérieures sont
toutes utilisées (`10/10` le 2026-10-19) et ne peuvent plus être rejouées avant le
2026-10-20 (fenêtre déjà rouverte, `0/10`).

## Conclusion

L'artefact certifié qui représente le mieux le programme reste **v7, réf. 57090508**. La
Phase 24 ne propose aucun remplacement. La certification est donc une **re-certification**
du champion existant, pas une nouvelle soumission.
