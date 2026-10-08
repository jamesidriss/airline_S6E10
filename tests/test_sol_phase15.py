"""Real certificate rejection checks: overlapping labels, row order and recipe drift."""
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import numpy as np
from src.common import arr_sha256, file_sha256
import scripts.certify_sol_phase15_raw as certificate


@contextmanager
def fixture():
    with TemporaryDirectory() as directory:
        root=Path(directory);reports=root/'reports';artifacts=root/'artifacts'
        source_dir=reports/'sol_tabpfn35_predict_guard';source_dir.mkdir(parents=True)
        context_dir=reports/'context';context_dir.mkdir();pred_dir=artifacts/'context';pred_dir.mkdir(parents=True)
        weights=root/'weights';weights.write_bytes(b'fixture checkpoint')
        source={'params':{'model_path':str(weights),'random_state':1201},
                'checkpoint':{'checkpoint_sha256':file_sha256(weights)},'feature_names':['x']}
        source_path=source_dir/'raw_f0.json';source_path.write_text(json.dumps(source))
        ids=np.arange(10);folds=ids%5;y=(ids%2).astype('int8');apply_ids=np.array([100,101,102]);k=2
        fit_ids=ids[folds!=k];p=np.array([.1,.7,.8],dtype='float32')
        cv_dir=reports/'cv';cv_dir.mkdir();cv_path=cv_dir/'raw_f2.json'
        cv={'prediction_sha256':'fixture_cv','feature_fit_sha256':'fixture_features','fit_ids_sha256':arr_sha256(fit_ids)}
        cv_path.write_text(json.dumps(cv));dh={'train':'fixture_train','test':'fixture_test'}
        c={'mode':'test','arm':'raw','fold':k,'seed':1201,'train_rows':len(fit_ids),'apply_rows':len(apply_ids),
           'fit_ids_sha256':arr_sha256(fit_ids),'apply_ids_sha256':arr_sha256(apply_ids),'data_sha256':dh,
           'fold_sha256':arr_sha256(folds),'scheme':'primary','params':source['params'],'checkpoint':source['checkpoint'],
           'source_reference_sha256':file_sha256(source_path),'feature_names':source['feature_names'],
           'protocol_sha256':file_sha256('research/sol_phase15_raw_protocol.md'),
           'scope_sha256':file_sha256('research/sol_phase15_scope_20261008.json'),
           'source_sha256':{},'library_versions':{'numpy':version('numpy')},
           'feature_fit_sha256':'fixture_features',
           'cv_proof':{'tag':'cv','report_sha256':file_sha256(cv_path),'prediction_sha256':'fixture_cv'}}
        r={'contract':c,'prediction_sha256':arr_sha256(p)};path=context_dir/'test_f2.json';path.write_text(json.dumps(r))
        np.save(pred_dir/'test_f2.npy',p);np.save(pred_dir/'ids_test_f2.npy',apply_ids)
        with patch.object(certificate,'REPORTS',reports),patch.object(certificate,'ARTIFACTS',artifacts):
            yield path,ids,y,folds,apply_ids,dh,r,pred_dir


def reject(call):
    try:call()
    except AssertionError:return
    raise AssertionError('Invalid evidence was accepted')


def test_certificate_rejects_inference_recipe_drift():
    with fixture() as (path,ids,y,folds,apply_ids,dh,r,_):
        certificate.verify_context(path,ids,y,folds,apply_ids,dh,'test',2)
        r=deepcopy(r);r['contract']['params']['random_state']=999;path.write_text(json.dumps(r))
        reject(lambda:certificate.verify_context(path,ids,y,folds,apply_ids,dh,'test',2))


def test_certificate_rejects_apply_row_order_drift():
    with fixture() as (path,ids,y,folds,apply_ids,dh,_,pred_dir):
        certificate.verify_context(path,ids,y,folds,apply_ids,dh,'test',2)
        np.save(pred_dir/'ids_test_f2.npy',apply_ids[::-1])
        reject(lambda:certificate.verify_context(path,ids,y,folds,apply_ids,dh,'test',2))


def test_certificate_rejects_apply_labels_in_fit_context():
    with fixture() as (path,ids,y,folds,apply_ids,dh,r,pred_dir):
        # All lengths, ID hashes and sidecars agree, but one apply row is in FIT.
        poisoned=apply_ids.copy();poisoned[0]=ids[0]
        r['contract']['apply_ids_sha256']=arr_sha256(poisoned);path.write_text(json.dumps(r))
        np.save(pred_dir/'ids_test_f2.npy',poisoned)
        reject(lambda:certificate.verify_context(path,ids,y,folds,poisoned,dh,'test',2))
