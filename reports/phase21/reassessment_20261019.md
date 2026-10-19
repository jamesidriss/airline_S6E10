# Réévaluation — priorité aux 80 % privés

Date UTC : 2026-10-19T09:37:14.865056+00:00. Cette réévaluation corrige les conclusions générales trop fortes des rapports de fin de phase 20 ; leurs données historiques restent conservées.

## Décision

Conserver **v7**, référence Kaggle **57090508**, OOF **0.9618529228157235**. Aucun nouveau candidat admis, aucun nouvel entraînement, aucune nouvelle soumission. Fichiers v6/v7 et plis primaires inchangés.

Les **20 % publics /80 % privés désignent des ensembles de lignes distincts**. Le classement final dépend uniquement du privé, pas d'une moyenne pondérée des deux scores. Les règles maintiennent deux sélections finales et 10 soumissions par jour ; échéance 31 octobre 2026 à 23:59 UTC. [Règles Kaggle](https://www.kaggle.com/competitions/playground-series-s6e10/rules).

## Historique repris

Les **558 entrées antérieures** ont été analysées comme des expériences, contrôles, corrections, essais de ressources et soumissions : elles ne représentent pas 558 modèles indépendants. Le catalogue actuel comporte 96 lignes : F0=23, F1=9, F2=60, F3=2, F5=2. Les 23 F0 ne sont pas des résultats prédictifs négatifs. Les comptes du catalogue actuel remplacent les chiffres divergents de l'ancien rapport final.

Les phases 17–18 établissent l'apport du contexte TabPFN et de son ensemble interne 2, puis des rendements limités des changements de contexte et du passage 2→4 sur les folds mesurés. Les phases 19–20 explorent de nombreuses familles, avec beaucoup de sélection sur le fold 0 déjà réutilisé. Cela reste de la découverte, pas une confirmation indépendante.

## Corrections scientifiques et techniques

1. **Incertitude appariée.** L'erreur standard de l'AUC seule de v7 vaut **0.000242411** ; celle du gain v7−v6 vaut **0.000013383**. Le gain **+0.000087309** a un intervalle conditionnel à 95 % **[+0.000061079 ; +0.000113539]**. L'ancienne affirmation qu'aucun gain inférieur à 0.00025 n'est distinguable du bruit est incorrecte. La comparaison utilise les placements positifs/négatifs et demi-crédits pour les égalités, selon [DeLong et al.](https://pubmed.ncbi.nlm.nih.gov/3203132/). Elle exclut l'incertitude de réentraînement et de sélection adaptative.
2. **Poids des mélanges.** Neuf mélanges H94 à 50/50 ne prouvaient pas qu'une contribution de 5 % ou 10 % échouerait. Les mesures ci-dessous remplacent cette extrapolation.
3. **Dérive.** Le contrôle utilisait des codes catégoriels construits séparément. Il compare désormais les catégories réelles, y compris valeurs nouvelles et absentes. Sur les données présentes : SMD maximal **0.00334236**, variation totale maximale **0.00291365**. Cela indique peu de dérive marginale observée, sans prouver une composition privée identique ni une absence de dérive conditionnelle.
4. **Internal4 déjà mesuré.** Le passage 2→4 existait dans l'audit de phase 18 : gain de portefeuille fold 0 **+0.000003818**. JINT04 était interrompu sans résultat complet. Ses trois membres terminés sont **bit à bit identiques** à l'ancien B2. RUNNING est corrigé en FAILED ; fichiers conservés. Un seul fold ne prouve pas un plateau général.
5. **Budget.** Au moins **4101.485 secondes /1.139301h** manquaient à la comptabilité. Le minimum documenté atteint **15.678301h** sur 22 h. Le solde de **6.321699h maximum** n'est pas dépensable tant que la durée finale manquante n'est pas réconciliée. Le garde refuse ce cas et un fichier de budget existant illisible ; les analyses CPU restent possibles.

## Étude nouvelle, protocole figé avant résultats

24 combinaisons : 10 auxiliaires certifiés ajoutés séparément à 5 % et 10 %, deux poids de route fixés à 40 %/60 %, puis deux mélanges TabM historiques à 5 %/10 % (diagnostic seulement, provenance insuffisante pour admission directe).

Chaque combinaison a été évaluée sur les cinq plis existants et 100 partitions communes : **59 969 lignes publiques /239 875 privées**. Partitions disjointes et indépendantes des candidats ; correction de Holm sur 24 comparaisons ; aucune pondération ou sélection fondée sur le public Kaggle.

**Résultat : 0/24 admissions.** Le meilleur essai exploratoire, **5 % TabM**, gagne **+0.000003726** en OOF :4/5 plis positifs, t=1.981, IC à 95 % apparié **[−0.000002606 ; +0.000010057]**, p corrigée=1, P05 privé **−0.000003166**. Il ne justifie ni remplacement ni soumission. Le meilleur ajout provenant exclusivement de membres certifiés gagne **+0.000000707** et échoue aussi.

Ces simulations conditionnelles réutilisent les prédictions OOF : elles ne révèlent pas les étiquettes privées et ne simulent pas le réentraînement. Les taux de renversement ne sont pas des probabilités de victoire. Holm couvre ces 24 comparaisons, pas toute l'exploration historique.

## Recherche publique actualisée

Une [discussion du 18 octobre](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750479) rapporte 0.96238 pour un **ensemble OOF** utilisant des prédictions TFM partagées, et 0.96114 public pour les arbres seuls. Ce 0.96238 n'est pas un résultat privé démontré. Protocole CV et provenance non certifiés ici ; aucun fichier externe incorporé.

## Prochaine décision utile

Conserver v7 comme référence. Une nouvelle piste doit apporter un mécanisme différent ou un signal reproductible, passer une comparaison contre v7 à contrôle identique puis une confirmation indépendante. Les évaluations déjà fermées et le quota disponible ne justifient pas seuls de nouveaux essais. Avant tout calcul GPU : terminer la comptabilité de JINT04. Avant toute promotion : conserver les gates de fuite, CV, multiplicité, inférence et certification.

La sélection finale Kaggle n'a pas été modifiée pendant cette réévaluation. Deux places existent ; aucun second modèle suffisamment solide et complémentaire n'est établi.

## Vérification

Huit nouveaux tests de statistiques/dérive et deux protections de budget sont intégrés à la suite commune. Suite complète sur la machine hôte : **244 tests réussis, 0 échec**. Le rapport JSON de clôture et le journal conservent le résultat et son empreinte SHA-256.

Fichiers détaillés : private_reassessment_20261019.json ; history_inventory_20261019.json ; interrupted_internal4_recovery_20261019.json ; official_and_community_refresh_20261019.json ; reassessment_completion_20261019.json.
