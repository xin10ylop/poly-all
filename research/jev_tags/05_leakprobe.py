"""Leakage diagnostic (NOT a trading signal): ask Jev directly whether each market resolved YES, on a
stratified sample of uncertain markets, and compare its AUC by end-month with the executable market price.
If Jev's direct forecast beats the market mostly on older markets, the model has outcome knowledge and any
semantic tag (esp. 'drama') may be contaminated."""
import os, sys, pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from jevq import *

OUT = os.path.join(ROOT, 'data', 'jev_tags', 'leakprobe.pkl')
U = pd.read_pickle(os.path.join(ROOT, 'data', 'jev_tags', 'universe.pkl')).set_index('mid')
P = pd.read_pickle(os.path.join(ROOT, 'data', 'jev_tags', 'panel.pkl'))
# YES reference price at hd=3 (fallback 1/7) from executable YES VWAP or 1 - NO VWAP
yp = P.assign(py=np.where(P.side == 'YES', P.px, 1 - P.px)).groupby(['mid', 'hd']).py.mean().unstack()
yp['p'] = yp[3].fillna(yp[1]).fillna(yp[7])
yp = yp[(yp.p > 0.1) & (yp.p < 0.9)].join(U[['eid', 'end', 'y']])
yp['m'] = pd.to_datetime(yp.end, unit='s').dt.to_period('M').astype(str)
yp = yp[(yp.m >= '2025-10') & (yp.m <= '2026-08')]
samp = (yp.reset_index().sample(frac=1, random_state=0).groupby('eid').head(2)
        .groupby('m').head(130).set_index('mid'))
print('sample', len(samp), samp.m.value_counts().sort_index().to_dict(), flush=True)

Q = {'fc': {'type': 'noul',
            'instructions': 'Forecast whether this prediction market resolved YES. Use any knowledge you have about what actually happened.',
            'criteria': {'true': 'the market resolved YES', 'false': 'the market resolved NO'}}}
prev = load_spend()


def work(mid):
    r = U.loc[mid]
    st = {'market_question': clean(r.q, 300), 'scheduled_end_date': d(r.end)}
    if r.n_ev_mkts > 1:
        st['group_title'] = clean(r.ev_title, 200)
        if r.mtitle and r.mtitle != r.q:
            st['this_option'] = clean(r.mtitle, 120)
    try:
        return mid, jev.decide(st, Q)['fc']['noul']
    except Exception:
        return mid, None


with ThreadPoolExecutor(8) as ex:
    res = dict(ex.map(work, list(samp.index)))
samp['jev'] = pd.Series(res)
samp.to_pickle(OUT)
print('spend', save_spend(prev))

from sklearn.metrics import roc_auc_score
def auc(g, c):
    g = g.dropna(subset=[c])
    return roc_auc_score(g.y, g[c]) if g.y.nunique() == 2 else np.nan
out = samp.groupby('m').apply(lambda g: pd.Series({'n': len(g), 'yes': g.y.mean(), 'auc_mkt': auc(g, 'p'), 'auc_jev': auc(g, 'jev'),
                                                    'brier_mkt': ((g.p - g.y) ** 2).mean(), 'brier_jev': ((g.jev - g.y) ** 2).mean()}))
print(out.round(3).to_string())
