"""Numerical gate rejects semantic drift even when rankings look plausible."""
import numpy as np
import torch
from unittest.mock import patch
from scripts.phase17_contracts import equivalence
from scripts.phase17_four_contracts import equivalence_four
from scripts.assemble_phase17_primary import scatter_predictions
from scripts.probe_phase17_tabpfn import tensor_description


def test_internal_equivalence_rejects_class_order_and_row_order_drift():
    p=np.array([[.8,.2],[.3,.7],[.6,.4]],dtype='float32'); ids=np.array([4,7,9]); cfg=['a','b']
    assert equivalence(p,p,ids,ids,cfg,cfg)['passes']
    assert not equivalence(p,p[:,::-1],ids,ids,cfg,cfg)['passes']
    for other_ids,other_cfg in ((ids[::-1],cfg),(ids,cfg[::-1])):
        try: equivalence(p,p,ids,other_ids,cfg,other_cfg)
        except ValueError: continue
        raise AssertionError('Semantic row/config drift accepted')


def test_internal_equivalence_rejects_nonfinite_badshape_and_denormalized_outputs():
    p=np.array([[.8,.2],[.3,.7]],dtype='float32'); ids=np.array([1,2]); cfg=['a','b']
    for bad in (p[:,1],p*np.nan,p*.99,np.array([[1.1,-.1],[.3,.7]])):
        try: equivalence(p,bad,ids,ids,cfg,cfg)
        except ValueError: continue
        raise AssertionError('Invalid output accepted')
    shifted=p.copy(); shifted[0]+=[-3e-6,3e-6]
    assert not equivalence(p,shifted,ids,ids,cfg,cfg)['passes']


def test_attention_diagnostic_preserves_actual_strides_dtype_and_shape():
    t=torch.zeros(2,3,8,5).permute(0,2,1,3)
    d=tensor_description(t)
    assert d['shape']==[2,8,3,5] and d['stride']==list(t.stride())
    assert not d['contiguous'] and d['dtype']=='torch.float32'


def test_supported_auto_memory_retains_fit_saving_without_splitting_small_query():
    from tabpfn.memory import should_save_peak_mem
    with patch('tabpfn.memory._get_free_cuda_memory_bytes',return_value=12e9):
        devices=[torch.device('cuda')]
        assert should_save_peak_mem('auto',(100000,22),(0,22),devices,2)
        assert not should_save_peak_mem('auto',(0,22),(1024,22),devices,2)
        assert should_save_peak_mem(True,(0,22),(1024,22),devices,2)


def test_official_zero_budget_produces_separate_member_caches_in_order():
    from tabpfn.inference import _cache_builds
    from tabpfn.architectures.interface import EstimatorBatchBudget
    assert _cache_builds([0,1],100000,22,EstimatorBatchBudget(rows=0,cells=4000000))==[[[0]],[[1]]]
    # A small positive budget still concatenates separately built caches; zero is essential.
    assert _cache_builds([0,1],100000,22,EstimatorBatchBudget(rows=100000,cells=4000000))==[[[0],[1]]]


def test_four_estimator_gate_checks_every_configuration_and_class_column():
    p=np.array([[.8,.2],[.3,.7]],dtype='float32'); ids=np.array([7,9]); cfg=['a','b','c','d']
    assert equivalence_four(p,p,ids,ids,cfg,cfg)['passes']
    assert not equivalence_four(p,p[:,::-1],ids,ids,cfg,cfg)['passes']
    for wrong in (cfg[:2],['a','b','d','c']):
        try: equivalence_four(p,p,ids,ids,cfg,wrong)
        except ValueError: continue
        raise AssertionError('Last two configurations escaped validation')


def test_full_primary_assembly_rejects_missing_fold_and_swapped_ids():
    ids=np.arange(10);folds=np.arange(10)%5;p=np.linspace(.1,.9,10,dtype='float32')
    parts={k:(p[folds==k],ids[folds==k]) for k in range(5)}
    assert np.array_equal(scatter_predictions(parts,ids,folds),p)
    for bad in ({k:v for k,v in parts.items() if k!=4},{**parts,4:(parts[4][0],parts[4][1][::-1])}):
        try:scatter_predictions(bad,ids,folds)
        except ValueError:continue
        raise AssertionError('Incomplete/misaligned OOF was accepted')


def test_saved_official_member_replay_rejects_tampering_and_wrong_aggregation():
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from src.common import arr_sha256
    from scripts.phase17_certification import replay_members
    params={'n_estimators':2,'device':'cpu','average_before_softmax':False,'balance_probabilities':False}
    values=[np.array([[[4.,0.],[0.,2.]]],dtype='float32'),
            np.array([[[0.,1.],[1.,0.]]],dtype='float32')]
    metadata={'n_estimators_resolved':2,'row_subsampling':None,'balance_probabilities':False,
        'average_before_softmax':False,'softmax_temperature':1.,
        'official_inference_config':{'USE_SKLEARN_16_DECIMAL_PRECISION':False},
        'members':[{'index':i,'raw_logits_sha256':arr_sha256(v)} for i,v in enumerate(values)]}
    with TemporaryDirectory() as directory:
        folder=Path(directory)
        for i,v in enumerate(values):
            np.save(folder/f'member_{i}_raw_logits.npy',v)
            (folder/f'member_{i}_stats.json').write_text(json.dumps(metadata['members'][i]))
        p=replay_members(folder,metadata,params,2)
        expected=torch.softmax(torch.from_numpy(np.concatenate(values,axis=0)),dim=-1).mean(0).numpy()[:,1]
        wrong=torch.softmax(torch.from_numpy(np.concatenate(values,axis=0)).mean(0),dim=-1).numpy()[:,1]
        assert np.allclose(p,expected,atol=1e-7,rtol=0) and np.max(np.abs(p-wrong))>.1
        damaged=values[1].copy();damaged[0,0,0]+=1
        np.save(folder/'member_1_raw_logits.npy',damaged)
        try:replay_members(folder,metadata,params,2)
        except AssertionError:pass
        else:raise AssertionError('Altered member logits escaped certification')


def test_secret_scan_checks_untruncated_contents_without_returning_secret_values():
    from scripts.audit_phase17_integrity import scan_bytes
    secret='sk-'+'a'*36
    findings=scan_bytes(('safe line\n'+'x'*300000+'\n'+secret).encode())
    assert findings==[{'kind':'openai_token','line':3}]
    assert secret not in str(findings)
    assert scan_bytes(b'{"token": "placeholder"}')==[]


def test_resume_rejects_missing_noncontiguous_and_tampered_member_contributions():
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from src.common import arr_sha256
    from scripts.phase17_resume import completed_prefix
    v=np.zeros((1,2,2),dtype='float32')
    with TemporaryDirectory() as directory:
        p=Path(directory)
        np.save(p/'member_1_raw_logits.npy',v)
        stats={'index':1,'raw_logits_sha256':arr_sha256(v),'gpu_allocated_after_release':0}
        (p/'member_1_stats.json').write_text(json.dumps(stats))
        try:completed_prefix(p,2,2)
        except ValueError:pass
        else:raise AssertionError('Noncontiguous completed member accepted')
        np.save(p/'member_0_raw_logits.npy',v)
        try:completed_prefix(p,2,2)
        except ValueError:pass
        else:raise AssertionError('Incomplete member accepted')
        (p/'member_0_stats.json').write_text(json.dumps({**stats,'index':0}))
        assert len(completed_prefix(p,2,2))==2
        changed=v.copy();changed[0,0,0]=1
        np.save(p/'member_1_raw_logits.npy',changed)
        try:completed_prefix(p,2,2)
        except ValueError:pass
        else:raise AssertionError('Tampered saved member accepted')


def test_resume_skips_completed_fit_and_preserves_official_member_aggregation():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from tabpfn import TabPFNClassifier
    from src.common import arr_sha256
    from scripts.phase16_tabpfn import probabilities
    from scripts.phase17_resume import resume_predict
    values=[np.array([[[4.,0.],[0.,2.]]],dtype='float32'),np.array([[[0.,1.],[1.,0.]]],dtype='float32')]
    clf=TabPFNClassifier(n_estimators=2,device='cpu',average_before_softmax=False,balance_probabilities=False)
    clf.executor_=None;clf.n_classes_=2;clf.softmax_temperature_=1.
    clf.inference_config_=SimpleNamespace(USE_SKLEARN_16_DECIMAL_PRECISION=False)
    clf.predict_raw_logits=lambda a:values[clf.executor_.index][:,a[:,0].astype(int)]
    fitted=[];completed=[]
    def factory(**kw):
        i=kw['ensemble_preprocessor'].member.index;fitted.append(i);return SimpleNamespace(index=i)
    def release_cpu(model):model.executor_=None
    stats={'index':0,'fit_seconds':1.,'total_seconds':2.,'peak_gpu_bytes':0,
           'raw_logits_sha256':arr_sha256(values[0]),'gpu_allocated_after_release':0}
    prepared=(factory,{'ensemble_preprocessor':None},[SimpleNamespace(index=0),SimpleNamespace(index=1)],
              {'members':[{'index':0},{'index':1}]})
    with patch('scripts.phase17_resume.release',release_cpu),patch('torch.cuda.reset_peak_memory_stats'):
        p,metadata=resume_predict(clf,prepared,np.array([[0],[1]]),[(values[0],stats)],
            lambda *_:nullcontext(),lambda *_:None,lambda i,v,s:completed.append((i,arr_sha256(v))))
    assert fitted==[1] and completed==[(0,arr_sha256(values[0])),(1,arr_sha256(values[1]))]
    assert np.array_equal(p,probabilities(clf,np.concatenate(values)).astype('float32'))
    assert [m['raw_logits_sha256'] for m in metadata['members']]==[arr_sha256(v) for v in values]


def test_submission_boundary_rejects_partial_certificate_even_with_valid_vectors():
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from src.common import arr_sha256,file_sha256
    import scripts.decide_phase17_submission as d
    ids=np.array([1,2]);test_ids=np.array([3,4]);p=np.array([.2,.8],dtype='float32')
    with TemporaryDirectory() as directory:
        root=Path(directory);reports=root/'reports';reports.mkdir();artifact=root/'prediction';artifact.mkdir()
        np.save(artifact/'candidate.npy',p);np.save(artifact/'test_ids.npy',test_ids)
        pp=reports/'toy_primary.json';pp.write_text('{}')
        sp=reports/'toy_shadow_confirmation.json';sp.write_text(json.dumps({'status':'MATCHED_SHADOW_CONFIRMATION_PASS',
            'primary_report_sha256':file_sha256(pp),'source_sha256':{},'mean_paired_delta':.0001,'paired_fold_deltas':[.0001,.0001]}))
        csv=root/'toy.csv';csv.write_text('id,satisfaction\n3,0.2\n4,0.8\n')
        cert={'status':'CERTIFIED_PHASE17_TEST_REQUIRES_ROBUSTNESS_DECISION','bank':{},'source_sha256':{},
            'primary_tag':'toy','primary_report_sha256':file_sha256(pp),'shadow_report_sha256':file_sha256(sp),
            'test_artifact_root':str(artifact),'test_sha256':{'candidate':arr_sha256(p)},'oof_sha256':{'candidate':arr_sha256(p)},
            'submission_path':str(csv),'submission_sha256':file_sha256(csv)}
        cp=root/'certificate.json'
        real_path=Path
        def test_path(value):return reports if str(value)=='reports' else real_path(value)
        with patch.object(d,'Path',test_path),patch.object(d,'load_primary',return_value=({}, {'candidate':p})):
            cp.write_text(json.dumps(cert));d.checked_certificate(cp,{},ids,test_ids)
            cp.write_text(json.dumps({**cert,'status':'PRIMARY_FOLD0_ONLY'}))
            try:d.checked_certificate(cp,{},ids,test_ids)
            except AssertionError:pass
            else:raise AssertionError('Partial-fold candidate crossed the submission boundary')


def test_resume_worker_guard_skips_own_launcher_but_blocks_separate_worker():
    import os
    from pathlib import Path
    from types import SimpleNamespace
    import psutil
    from scripts.resume_phase17_context import assert_no_other_workers
    make = lambda pid, cwd: SimpleNamespace(info={'pid':pid,
        'cmdline':['python.exe','scripts/resume_phase17_context.py']}, cwd=lambda:cwd)
    launcher, sibling = make(99,Path.cwd()), make(200,Path.cwd())
    with patch.object(os,'getpid',return_value=100), patch.object(psutil,'Process',
            return_value=SimpleNamespace(parents=lambda:[SimpleNamespace(pid=99)])):
        with patch.object(psutil,'process_iter',return_value=[launcher]):
            assert_no_other_workers()
        with patch.object(psutil,'process_iter',return_value=[launcher,sibling]):
            try:assert_no_other_workers()
            except AssertionError:pass
            else:raise AssertionError('A separate live workspace worker escaped detection')
        with patch.object(psutil,'process_iter',return_value=[make(201,Path.cwd().parent)]):
            assert_no_other_workers()
