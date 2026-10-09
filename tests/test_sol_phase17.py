"""Numerical gate rejects semantic drift even when rankings look plausible."""
import numpy as np
import torch
from scripts.phase17_contracts import equivalence
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
