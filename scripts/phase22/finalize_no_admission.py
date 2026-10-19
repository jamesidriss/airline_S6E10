"""Close the negative-result branch only after all frozen comparisons finish.

This script refuses a passing challenger: that branch requires shadow and certification.
"""
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.common import file_sha256,arr_sha256,load_cached_parquet,save_json
from src.validation.folds import get_scheme
from src.submission.store import load_oof,load_test

BASE='a919a8949f9c0f4c7df21c1b31462d2b18e551bc'
OUT=Path('reports/phase22')
def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8-sig'))

def main():
    destination=OUT/'completion_20261019.json';assert not destination.exists()
    now=datetime.now(timezone.utc).isoformat()
    protocol=json.loads(Path('configs/phase22/private_breakthrough_20261019.json').read_text())
    cat_spec=json.loads(Path('configs/phase22/lean84_cat_20261019.json').read_text())
    groups={n:read(n) for n in ('external_comparison','fresh_comparison','own_raw_replay','cat_comparison')}
    eligible=[r for n,d in groups.items() if n!='external_comparison' for r in d['candidates']]
    assert len(eligible)==8 and len(groups['external_comparison']['candidates'])==24
    passed=lambda r: (r['mean_fold_delta']>=1.5e-5 and r['fold_t']>=2.5 and r['positive_folds']==5 and r['pooled']['one_sided_p']<=cat_spec['alpha_per_comparison'] and r['private80']['private_p05']>=0)
    assert not any(passed(r) for r in eligible),'A challenger passed: run shadow; do not close as negative.'
    qa=read('full_tests_isolated_v2');initial=read('full_tests_isolated')
    assert qa['failed']==0 and qa['passed']==247
    for p,h in protocol['protected_files'].items():assert file_sha256(p)==h,p
    assert arr_sha256(load_oof(protocol['champion']))=='1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552'
    assert arr_sha256(load_test(protocol['champion']))=='2b744d0df36a0b1143a92e1db18661ccd94fa881490784308624aaa0a0fc95af'
    tr,te=load_cached_parquet();y=tr.satisfaction.to_numpy(dtype='int8');ids=tr.id.to_numpy();folds=get_scheme('primary',y,ids).folds
    training=[]
    for name,report_name,source,model_file in [('lean84','lean84_primary','lean84.py','model.txt'),('lean84_cat','lean84_cat_primary','lean84_cat.py','model.cbm')]:
        r=read(report_name);root=Path('artifacts/phase22')/name
        assert r['status']=='COMPLETE_FIVE_FOLD_PRIMARY' and len(r['folds'])==5
        assert file_sha256(Path('scripts/phase22')/source)==r['source_sha256']
        oof=np.load(root/'oof.npy',allow_pickle=False);test=np.load(root/'test.npy',allow_pickle=False)
        assert oof.shape==(len(tr),) and test.shape==(len(te),)
        assert arr_sha256(oof)==r['oof_sha256'] and arr_sha256(test)==r['test_sha256']
        assert np.isfinite(oof).all() and np.isfinite(test).all()
        assert ((oof>=0)&(oof<=1)).all() and ((test>=0)&(test<=1)).all()
        assert arr_sha256(ids)==r['train_ids_sha256'] and arr_sha256(te.id.to_numpy())==r['test_ids_sha256']
        recombined=np.zeros(len(te),dtype='float64')
        for f in r['folds']:
            k=f['fold'];folder=root/f'f{k}'
            a=np.load(folder/'oof.npy',allow_pickle=False);b=np.load(folder/'test.npy',allow_pickle=False)
            assert np.array_equal(a,oof[folds==k])
            assert np.array_equal(np.load(folder/'eval_ids.npy',allow_pickle=False),ids[folds==k])
            assert arr_sha256(ids[folds!=k])==f['fit_ids_sha256'] and arr_sha256(ids[folds==k])==f['eval_ids_sha256']
            assert arr_sha256(np.concatenate([a,b]))==f['prediction_sha256']
            assert file_sha256(folder/model_file)==f['model_sha256']
            recombined+=b/5 if name=='lean84' else b.astype('float64')/5
        assert np.array_equal(test,recombined.astype('float32'))
        training.append({'name':name,'report':str(OUT/(report_name+'.json')),'OOF_AUC':r['oof_auc'],'CPU_wall_seconds':r['CPU_wall_seconds'],'GPU_hours':r['GPU_hours'],'complete_primary_contexts':5,'full_test_rows':len(te),'integrity':'PASS','submission_certified':False})
    budget=read('JINT04_reconciliation')
    prefix=subprocess.check_output(['git','show',BASE+':experiments/ledger.jsonl'])
    ledger=Path('experiments/ledger.jsonl');current=ledger.read_bytes();assert current==prefix
    rows=[]
    for group,d in groups.items():
        for r in d['candidates']:
            external=group=='external_comparison'
            rows.append({'kind':'EXTERNAL_DIAGNOSTIC' if external else 'EXPERIMENT','exp_id':'phase22_20261019_'+group+'_'+r['name'],'ts':now,'report':str(OUT/(group+'.json')),'candidate':r,'eligible':not external,'final_family_size':24 if external else 8,'final_alpha_per_comparison':None if external else .00625,'final_verdict':'INELIGIBLE_UNVERIFIED_PRODUCER' if external else 'NO_ADMISSION','public_LB_used_for_selection':False})
    for t in training:rows.append({'kind':'TRAINING','exp_id':'phase22_20261019_'+t['name']+'_five_fold_cpu','ts':now,**t})
    rows.append({'kind':'AUDIT','exp_id':'phase22_20261019_public_notebook_label_flip','ts':now,'report':'reports/phase22/public_notebook_leakage_audit.json','verdict':'REJECT_PUBLIC_GLOBAL_TE_AND_INSAMPLE_META_PROTOCOL'})
    rows.append({'kind':'CORRECTION','exp_id':'phase22_20261019_JINT04_driver_fault','ts':now,'report':'reports/phase22/JINT04_reconciliation.json','GPU_hours_new':0,'exact_exit_time_unknown':True,'hardware_stop':True})
    rows.append({'kind':'DECISION','exp_id':'phase22_20261019_keep_v7','ts':now,'verdict':'NO_QUALIFIED_IMPROVEMENT_KEEP_V7','eligible_comparisons':8,'external_diagnostics':24,'new_submissions':0,'private_score_unknown':True,'campaign_wide_selection_correction_claimed':False})
    assert len({r['exp_id'] for r in rows})==len(rows)==37
    with ledger.open('ab') as f:
        for r in rows:f.write((json.dumps(r,ensure_ascii=False)+'\n').encode('utf-8'))
    finalists=Path('experiments/private_finalists.json');fs=json.loads(finalists.read_text(encoding='utf-8'))
    assert fs['BEST_A']['ref']==57090508 and fs['BEST_B_HEDGE'] is None
    fs['phase22_reassessment']={'utc':now,'decision':'KEEP_CERTIFIED_V7_NO_NEW_ADMISSION','report':'reports/phase22/final_report_20261019.md','completion':str(destination),'fresh_five_fold_models':2,'fixed_comparisons':32,'new_submissions':0,'GPU_hours_new':0,'private_B':None,'Kaggle_final_selection_modified':False,'hardware_stop':True}
    save_json(fs,finalists)
    best_external=max(groups['external_comparison']['candidates'],key=lambda r:r['pooled']['delta'])
    body=['# Phase22 — exécution et décision du 19 octobre 2026','',
      '**V7 reste le meilleur candidat certifié : OOF 0,961852923, ref Kaggle 57090508. Aucun challenger qualifié ; aucune nouvelle soumission.**','',
      'Deux recettes nouvelles ont été entraînées sur les cinq folds primary immuables : LightGBM et CatBoost sur 84 variables compactes. Chaque fold utilise un split interne de 10 % pour choisir les itérations, puis un réentraînement sur tout le FIT externe. Les encodages de cible sont reconstruits dans chaque contexte. Les mêmes modèles produisent les prédictions OOF et les 299 844 lignes test. Les modèles, prédictions et identifiants sont conservés et hashés. Ce sont des composants de recherche, pas de nouveaux finalistes certifiés.','',
      '| Comparaison préenregistrée | Delta OOF vs v7 | Folds positifs | P05 private simulé | Décision |','|---|---:|---:|---:|---|']
    for group in ('fresh_comparison','own_raw_replay','cat_comparison'):
        for r in groups[group]['candidates']:body.append(f"| {r['name']} | {r['pooled']['delta']:+.9f} | {r['positive_folds']}/5 | {r['private80']['private_p05']:+.9f} | Rejet |")
    body+=['',f"Les 24 diagnostics sur les vecteurs publics donnent un meilleur mélange apparent à **{best_external['pooled']['candidate_auc']:.9f}** ({best_external['pooled']['delta']:+.9f}). Il reste **inéligible** : absence de manifestes de lignes et de contextes d'entraînement, folds seed42 différents, code de production complet non publié et confirmation shadow correspondante absente. Reproduire les AUC annoncées ne prouve pas la provenance des lignes. Les folds différents ne prouvent pas, à eux seuls, une fuite dans chaque prédiction.",'',
      "La source associée au score annoncé de 0,96238 ne permet pas de le reproduire : la version 4 téléchargée n'a aucun résultat sauvegardé et ne contient pas la fusion complète des 11 modèles annoncés. Son TE global contamine le FIT externe et son score de méta-modèle est calculé sur ses données d'ajustement. Un contrôle réel de permutation des labels du fold évalué modifie les 652 992 variables TE du FIT ; notre version imbriquée en modifie zéro. Cela établit un défaut du protocole fourni, sans chiffrer l'inflation du score annoncé.",'',
      'Les licences ont été vérifiées par source officielle et distinguées de la licence CC0 des vecteurs. Plusieurs poids ont des restrictions non commerciales ; aucun poids externe nouveau n’a été adopté. Les révisions exactes utilisées par le producteur restent inconnues. Voir public_tfm_provenance_audit.json et license_source_manifest.json.','',
      'Les huit comparaisons locales utilisent un seuil Bonferroni de 0,05/8 = 0,00625, en plus du gain moyen ≥15e-6, du rapport gain/SE ≥2,5 et de 5/5 folds positifs. Les 24 comparaisons externes sont des diagnostics avec correction Holm. Ces corrections locales ne corrigent pas rétroactivement les 96 pistes de la campagne ni le choix adaptatif des familles. Une confirmation sur les splits shadow préexistants reste obligatoire pour un candidat positif ; aucune n’a été déclenchée ici. Les labels primary ont déjà servi à la sélection historique.','',
      'Chaque comparaison comprend 100 partitions disjointes simulant 59 969 lignes publiques et 239 875 privées. Le classement final dépend du private, pas d’une moyenne pondérée des deux scores. Ces simulations mesurent une sensibilité conditionnelle des prédictions OOF ; elles ne donnent ni la probabilité de finir premier, ni une preuve sur les vrais labels privés, ni l’incertitude liée à un réentraînement.','',
      f"JINT04 : incident nvlddmkm153 à 01:04:58 UTC, immédiatement après la dernière progression connue. L'heure exacte de fin du processus demeure inconnue. Minimum comptabilisé **{budget['budget_total_interval_hours'][0]:.6f} h / 22 h** ; borne supérieure conservatrice **{budget['budget_total_interval_hours'][1]:.6f} h**, qui n'est pas une consommation mesurée. Arrêt GPU maintenu indépendamment du solde inconnu. **0 nouvelle heure GPU, aucun achat ni extension.** La reproduction propre du TabPFN brut à huit estimateurs, principale piste positive, reste bloquée. Le replay ancien à un seul estimateur ne la réfute pas.",'',
      'Le quota quotidien vérifié pendant cette phase était épuisé par les dix soumissions d’inventaire antérieures. Aucune de ces dix variantes n’atteint la fenêtre PATH B. V6 reste un secours préservé, sans complémentarité suffisante pour être recommandé comme second finaliste. La sélection finale Kaggle n’a pas été modifiée.','',
      f"Contrôles : second passage **{qa['passed']} contrôles enregistrés sans échec**, exécutés par module sur CPU avec CUDA masqué. Les tests exclusivement CUDA sont donc exclus de la validation matérielle actuelle. Le premier passage ({initial['passed']} sans exception, {initial['failed']} échec) est conservé : un test de distance exigeait une précision absolue irréaliste en float32 ; la tolérance est désormais proportionnelle à l'échelle et les indices exacts restent vérifiés. Aucun modèle de production n'a été modifié par cette correction.",'',
      'Les artefacts v6/v7 et les folds restent inchangés. 37 entrées ont été ajoutées au registre sans réécrire ses 584 entrées précédentes. Les preuves de reproduction, les sources publiques et les résultats négatifs sont conservés. La cible 0,962740 et la première place ne sont pas démontrées.','',
      'Sources : [discussion du 0,96238](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750479), [notebook associé](https://www.kaggle.com/code/jiatianzhang2/s6e10-15-fold-gpu-stacking-pipeline), [vecteurs publics](https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions), [licence TabFM](https://github.com/google-research/tabfm), [règles](https://www.kaggle.com/competitions/playground-series-s6e10/rules).','']
    (OUT/'final_report_20261019.md').write_text('\n'.join(body),encoding='utf-8')
    status=Path('STATUS_sol.md');old=status.read_bytes();status.write_bytes(('Phase22 — 2026-10-19 : deux recettes CPU entraînées sur cinq folds, 32 comparaisons fixes, aucune admission. V7 reste A ; v6 préservé ; aucun B qualifié. Piste publique raw8 positive mais non certifiable sans reproduction. GPU arrêté pour incidents pilote répétés et durée JINT04 inconnue ; 0 GPU nouveau. Rapport : reports/phase22/final_report_20261019.md. Les statuts ci-dessous sont historiques.\n\n').encode('utf-8')+old)
    with Path('research/current_meta.md').open('a',encoding='utf-8') as f:f.write('\n\nPhase22 closure 2026-10-19: two full five-fold CPU recipes, 24 external diagnostics and eight eligible fixed comparisons completed; none eligible passes the frozen family gate. Preserve v7, no qualified B, no submission. External raw8 positive signal remains unverified and hardware-blocked; raw1 is not an equivalent reproduction. See reports/phase22/final_report_20261019.md.\n')
    completion={'utc':now,'status':'NO_QUALIFIED_CHALLENGER_GPU_HARD_CONSTRAINT','champion_ref':57090508,'champion_OOF':.9618529228157235,'target':.962740,'training':training,'comparisons':32,'eligible_comparisons':8,'external_diagnostics':24,'new_GPU_hours':0,'new_submissions':0,'paid_compute':False,'budget_extension':False,'shadow_triggered':False,'private_B':None,'protected_files':protocol['protected_files'],'ledger_baseline_rows':len(prefix.splitlines()),'ledger_baseline_sha256':hashlib.sha256(prefix).hexdigest(),'ledger_appended_rows':len(rows),'ledger_current_sha256':file_sha256(ledger),'QA_registered_checks':qa['passed'],'QA_failures':qa['failed'],'QA_CUDA_hidden':True,'initial_QA_failure_preserved':True,'historical_campaign_selection_corrected':False,'private_score_unknown':True,'prereg_commits':['80d9eea','1d7a1eb','479b642'],'JINT04':budget,'report':'reports/phase22/final_report_20261019.md'}
    save_json(completion,destination);print(json.dumps({'status':completion['status'],'ledger_added':len(rows),'training_contexts':10,'new_submissions':0}))

if __name__=='__main__':main()
