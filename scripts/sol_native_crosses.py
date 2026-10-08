"""Thirty-nine exact rating/context categorical identities; no fitted target."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scripts.native_cat import to_cat_series
from scripts.run_phase13 import RATINGS

CONTEXT3=('Class','Type of Travel','Customer Type')


def cross_frame(src,rows):
    # Read only the declared covariates. A target/ID column in src is ignored.
    selected=src.iloc[rows]
    context={c:to_cat_series(selected[c]).to_numpy() for c in CONTEXT3}
    data={}
    for rating in RATINGS:
        answer=to_cat_series(selected[rating]).to_numpy()
        for c in CONTEXT3:
            # Factorization merely shares storage of equal strings. The output
            # retains literal tuple values, so different split vocabularies cannot
            # alias categories. JSON escaping avoids delimiter collisions.
            codes,levels=pd.factorize(pd.MultiIndex.from_arrays([answer,context[c]]),sort=True)
            labels=np.array([json.dumps([a,b],ensure_ascii=False,separators=(',',':')) for a,b in levels],dtype=object)
            assert (codes>=0).all()
            data[f'ix_pair_{rating}_{c}__cat']=pd.Series(labels[codes],dtype=object)
    return pd.DataFrame(data)
