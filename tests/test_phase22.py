"""Leak boundaries for the independent lean84 reconstruction."""
import numpy as np
import pandas as pd
from scripts.phase22.lean84 import encoded, static_features
from src.features.s6e10 import SURVEY13, META4, NUMS

def test_p22_encoding_excludes_apply_labels_and_own_label():
    keys={'te_0':pd.Series(['a','b','a','b']*10)}
    y=np.array([0,1,1,0]*10); fit=np.arange(30); apply=np.arange(30,40)
    a,b=encoded(keys,y,fit,apply,19)
    changed=y.copy(); changed[apply]=1-changed[apply]
    c,d=encoded(keys,changed,fit,apply,19)
    np.testing.assert_array_equal(a,c); np.testing.assert_array_equal(b,d)
    changed=y.copy(); changed[0]=1-y[0]
    c,_=encoded(keys,changed,fit,apply,19)
    np.testing.assert_array_equal(a[0],c[0])

def test_p22_static_ignores_target_and_id():
    f=pd.DataFrame({c:[0,1,3,5,2,4] for c in SURVEY13+NUMS})
    for c in META4: f[c]=['a','b','a','b','a','b']
    f['id']=np.arange(6); f['satisfaction']=[0,1,0,1,0,1]
    a,n,cat,k=static_features(f,np.full(6,.5))
    f['satisfaction']=1-f.satisfaction; f['id']+=1000
    b,n2,cat2,k2=static_features(f,np.full(6,.5))
    np.testing.assert_array_equal(a,b)
    assert n==n2 and cat==cat2 and len(n)==76 and len(cat)==44
    for name in k: pd.testing.assert_series_equal(k[name],k2[name])
