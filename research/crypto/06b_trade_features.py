"""Add trade-derived features to the panel: last trade before t (liquidity/staleness check),
next taker fills after t (realistic taker execution), and extreme fill prices within windows (maker fills).
YES-equivalent convention: a taker BUY YES @p or SELL NO @(1-p) is an ask-side print at p;
a taker SELL YES @p or BUY NO @(1-p) is a bid-side print at p.
Output: data/crypto/panel_tr.pkl"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa

P = pd.read_pickle(os.path.join(DATA, 'panel.pkl'))
P = P[P.kind != 'above_hourly'].reset_index(drop=True)
TR = pickle.load(open(os.path.join(DATA, 'trades.pkl'), 'rb'))
W_EXEC = 30 * 60
W_MAKER = [15 * 60, 60 * 60]

cols = {c: np.full(len(P), np.nan) for c in
        ['last_p', 'last_age', 'n_2h', 'med_2h', 'ask_next', 'ask_next_dt', 'ask_next_sz', 'bid_next', 'bid_next_dt',
         'bid_next_sz'] + [f'minbid_{w // 60}' for w in W_MAKER] + [f'maxask_{w // 60}' for w in W_MAKER]}
for mid, g in P.groupby('mid_id'):
    a = TR.get(mid)
    if a is None or len(a) == 0:
        continue
    t = a[:, 0].astype(float)
    o = np.argsort(t, kind='stable')
    t = t[o]
    side = a[o, 1]
    oi = a[o, 2].astype(int)
    px = a[o, 3].astype(float)
    sz = a[o, 4].astype(float)
    yeq = np.where(oi == 0, px, 1 - px)
    askside = ((side == 'BUY') & (oi == 0)) | ((side == 'SELL') & (oi == 1))
    ta, pa, sa = t[askside], yeq[askside], sz[askside]
    tb, pb, sb = t[~askside], yeq[~askside], sz[~askside]
    for i, tt in zip(g.index, g.t.values):
        j = np.searchsorted(t, tt, side='right') - 1
        if j >= 0:
            cols['last_p'][i] = yeq[j]
            cols['last_age'][i] = tt - t[j]
        lo = np.searchsorted(t, tt - 7200, side='left')
        cols['n_2h'][i] = j + 1 - lo
        if j + 1 - lo > 0:
            cols['med_2h'][i] = np.median(yeq[lo:j + 1])
        k = np.searchsorted(ta, tt, side='right')
        if k < len(ta) and ta[k] - tt <= W_EXEC:
            cols['ask_next'][i], cols['ask_next_dt'][i], cols['ask_next_sz'][i] = pa[k], ta[k] - tt, sa[k]
        k = np.searchsorted(tb, tt, side='right')
        if k < len(tb) and tb[k] - tt <= W_EXEC:
            cols['bid_next'][i], cols['bid_next_dt'][i], cols['bid_next_sz'][i] = pb[k], tb[k] - tt, sb[k]
        for w in W_MAKER:
            k0, k1 = np.searchsorted(tb, tt, side='right'), np.searchsorted(tb, tt + w, side='right')
            if k1 > k0:
                cols[f'minbid_{w // 60}'][i] = pb[k0:k1].min()
            k0, k1 = np.searchsorted(ta, tt, side='right'), np.searchsorted(ta, tt + w, side='right')
            if k1 > k0:
                cols[f'maxask_{w // 60}'][i] = pa[k0:k1].max()
for c, v in cols.items():
    P[c] = v
P.to_pickle(os.path.join(DATA, 'panel_tr.pkl'))
print(len(P), P[['last_age', 'n_2h', 'ask_next', 'bid_next']].describe().round(3))
