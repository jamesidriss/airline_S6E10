# Phase22 — exécution et décision du 19 octobre 2026

**V7 reste le meilleur candidat certifié : OOF 0,961852923, ref Kaggle 57090508. Aucun challenger qualifié ; aucune nouvelle soumission.**

Deux recettes nouvelles ont été entraînées sur les cinq folds primary immuables : LightGBM et CatBoost sur 84 variables compactes. Chaque fold utilise un split interne de 10 % pour choisir les itérations, puis un réentraînement sur tout le FIT externe. Les encodages de cible sont reconstruits dans chaque contexte. Les mêmes modèles produisent les prédictions OOF et les 299 844 lignes test. Les modèles, prédictions et identifiants sont conservés et hashés. Ce sont des composants de recherche, pas de nouveaux finalistes certifiés.

| Comparaison préenregistrée | Delta OOF vs v7 | Folds positifs | P05 private simulé | Décision |
|---|---:|---:|---:|---|
| lean84_w0.10 | -0.000009528 | 2/5 | -0.000026545 | Rejet |
| lean84_w0.25 | -0.000081192 | 0/5 | -0.000122500 | Rejet |
| lean84_w1.00 | -0.001317134 | 0/5 | -0.001455534 | Rejet |
| own_raw_w0.10 | +0.000005840 | 3/5 | -0.000006500 | Rejet |
| own_raw_w0.25 | -0.000026618 | 2/5 | -0.000058434 | Rejet |
| lean84_cat_w0.10 | -0.000013002 | 1/5 | -0.000023683 | Rejet |
| lean84_cat_w0.25 | -0.000071107 | 0/5 | -0.000097010 | Rejet |
| lean84_cat_w1.00 | -0.001044036 | 0/5 | -0.001148968 | Rejet |

Les 24 diagnostics sur les vecteurs publics donnent un meilleur mélange apparent à **0.961895157** (+0.000042234). Il reste **inéligible** : absence de manifestes de lignes et de contextes d'entraînement, folds seed42 différents, code de production complet non publié et confirmation shadow correspondante absente. Reproduire les AUC annoncées ne prouve pas la provenance des lignes. Les folds différents ne prouvent pas, à eux seuls, une fuite dans chaque prédiction.

La source associée au score annoncé de 0,96238 ne permet pas de le reproduire : la version 4 téléchargée n'a aucun résultat sauvegardé et ne contient pas la fusion complète des 11 modèles annoncés. Son TE global contamine le FIT externe et son score de méta-modèle est calculé sur ses données d'ajustement. Un contrôle réel par inversion des labels du fold évalué modifie l’encodage TE de 652 992 lignes du FIT ; notre version imbriquée en modifie zéro. Cela établit un défaut du protocole fourni, sans chiffrer l'inflation du score annoncé.

Les licences ont été vérifiées par source officielle et distinguées de la licence CC0 des vecteurs. Plusieurs poids ont des restrictions non commerciales ; aucun poids externe nouveau n’a été adopté. Les révisions exactes utilisées par le producteur restent inconnues. Voir public_tfm_provenance_audit.json et license_source_manifest.json.

Les huit comparaisons locales utilisent un seuil Bonferroni de 0,05/8 = 0,00625, en plus du gain moyen ≥15e-6, du rapport gain/SE ≥2,5 et de 5/5 folds positifs. Les 24 comparaisons externes sont des diagnostics avec correction Holm. Ces corrections locales ne corrigent pas rétroactivement les 96 pistes de la campagne ni le choix adaptatif des familles. Une confirmation sur les splits shadow préexistants reste obligatoire pour un candidat positif ; aucune n’a été déclenchée ici. Les labels primary ont déjà servi à la sélection historique.

Chaque comparaison comprend 100 partitions disjointes simulant 59 969 lignes publiques et 239 875 privées. Le classement final dépend du private, pas d’une moyenne pondérée des deux scores. Ces simulations mesurent une sensibilité conditionnelle des prédictions OOF ; elles ne donnent ni la probabilité de finir premier, ni une preuve sur les vrais labels privés, ni l’incertitude liée à un réentraînement.

JINT04 : incident nvlddmkm153 à 01:04:58 UTC, immédiatement après la dernière progression connue. L'heure exacte de fin du processus demeure inconnue. Minimum comptabilisé **15.678301 h / 22 h** ; borne supérieure conservatrice **24.043071 h**, qui n'est pas une consommation mesurée. Arrêt GPU maintenu indépendamment du solde inconnu. **0 nouvelle heure GPU, aucun achat ni extension.** La reproduction propre du TabPFN brut à huit estimateurs, principale piste positive, reste bloquée. Le replay ancien à un seul estimateur ne la réfute pas.

Le quota quotidien vérifié pendant cette phase était épuisé par les dix soumissions d’inventaire antérieures. Aucune de ces dix variantes n’atteint la fenêtre PATH B. La revue des huit candidats locaux au titre de PATH B ne retient aucun second finaliste : les modèles purs échouent à la fenêtre OOF, et les mélanges conservent 75–90 % de v7 avec une corrélation supérieure à 0,99 sans différence structurelle démontrée du portefeuille. Détails : path_B_and_quota_review.json. V6 reste un secours préservé, sans complémentarité suffisante pour être recommandé comme second finaliste. La sélection finale Kaggle n’a pas été modifiée.

Contrôles : second passage : **245 contrôles exécutés sans échec et 2 contrôles CUDA ignorés** (247 retours sans exception dans le compteur historique). La suite est exécutée par module sur CPU avec CUDA masqué ; elle ne valide pas le matériel GPU. Le premier passage (246 sans exception, 1 échec) est conservé : un test de distance exigeait une précision absolue irréaliste en float32 ; la tolérance est désormais proportionnelle à l'échelle et les indices exacts restent vérifiés. Aucun modèle de production n'a été modifié par cette correction.

Les artefacts v6/v7 et les folds restent inchangés. 37 entrées ont été ajoutées au registre sans réécrire ses 584 entrées précédentes. Les preuves de reproduction, les sources publiques et les résultats négatifs sont conservés. La cible 0,962740 et la première place ne sont pas démontrées.

Sources : [discussion du 0,96238](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750479), [notebook associé](https://www.kaggle.com/code/jiatianzhang2/s6e10-15-fold-gpu-stacking-pipeline), [vecteurs publics](https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions), [licence TabFM](https://github.com/google-research/tabfm), [règles](https://www.kaggle.com/competitions/playground-series-s6e10/rules).
