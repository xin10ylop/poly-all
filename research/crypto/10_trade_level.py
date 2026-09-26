"""Trade-level analysis of resolved non-hourly crypto threshold markets:
who wins - takers or makers? and does a model-anchored maker (quote at model +/- thr, re-quoted with spot)
earn money on the flow that trades through its quotes?
Output: data/crypto/trade_level.pkl + printed tables."""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa
from bt_models import *  # noqa

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 30)

evs = [json.loads(l) for l in open(os.path.join(DATA, 'closed_events.jsonl'))]
TR = pickle.load(open(os.path.join(DATA, 'trades.pkl'), 'rb'))
emp = build_empirical()
K1 = {u: load_k(u, '1m') for u in UNDS}
K1a = {u: (K1[u].index.values.astype('datetime64[s]').astype(np.int64), K1[u]['c'].values, K1[u]['h'].values,
           K1[u]['l'].values) for u in UNDS}
RV = {u: rv_frame(load_k(u, '5m')) for u in UNDS}
RVa = {u: (RV[u].index.values.astype('datetime64[s]').astype(np.int64), RV[u]['rv'].values) for u in UNDS}

rows = []
for e in evs:
    for m in e['markets']:
        r = parse_market(e, m)
        if r is None or r['kind'] == 'above_hourly':
            continue
        a = TR.get(m['id'])
        if a is None or len(a) == 0:
            continue
        try:
            y = float(r['outcomePrices'][0])
        except Exception:
            continue
        if y not in (0.0, 1.0):
            continue
        u = r['und']
        T = int(r['end'].timestamp())
        t = a[:, 0].astype(np.int64)
        oi = a[:, 2].astype(int)
        px = a[:, 3].astype(float)
        sz = a[:, 4].astype(float)
        side = a[:, 1]
        yeq = np.where(oi == 0, px, 1 - px)
        ask = ((side == 'BUY') & (oi == 0)) | ((side == 'SELL') & (oi == 1))
        keep = t < T - 60
        tk, ck, hk, lk = K1a[u]
        ci = np.searchsorted(tk, t - 60, side='right') - 1
        keep &= ci >= 0
        hi_sf = np.full(len(t), np.nan)
        lo_sf = np.full(len(t), np.nan)
        if r['payoff'].startswith('touch'):
            ws = int(r['win_start'].timestamp())
            a0 = np.searchsorted(tk, ws, side='left')
            b1 = np.searchsorted(tk, T, side='right')
            cmx = np.maximum.accumulate(hk[a0:b1])
            cmn = np.minimum.accumulate(lk[a0:b1])
            j = ci - a0
            okj = (j >= 0) & (j < len(cmx))
            hi_sf[okj] = cmx[j[okj]]
            lo_sf[okj] = cmn[j[okj]]
            touched = (hi_sf >= r['K']) if r['payoff'] == 'touch_up' else (lo_sf <= r['K'])
            keep &= ~touched & (t >= ws)
        if not keep.any():
            continue
        kind = 'hit_weekly' if r['kind'] == 'hit_weekly2' else r['kind']
        for i in np.where(keep)[0]:
            rows.append((m['id'], e['slug'], kind, u, r['payoff'], r.get('K'), r.get('lo'), r.get('hi'), T, int(t[i]),
                         yeq[i], bool(ask[i]), sz[i], y, ck[ci[i]], r['vol']))
D = pd.DataFrame(rows, columns=['mid_id', 'event', 'kind', 'und', 'payoff', 'K', 'lo', 'hi', 'T', 't', 'p', 'ask',
                                'sz', 'y', 'S', 'mvol'])
print('trades', len(D), 'markets', D.mid_id.nunique())
D['tau_s'] = D['T'] - D['t']
D['rv'] = np.nan
for u in UNDS:
    msk = D.und == u
    ii = np.searchsorted(RVa[u][0], D.loc[msk, 't'].values, side='right') - 1
    D.loc[msk, 'rv'] = RVa[u][1][ii]
D['q'] = [emp_prob(emp, pf, S, ta, sg, K=K, lo=lo, hi=hi) for pf, S, ta, sg, K, lo, hi in
          zip(D.payoff, D.S, D.tau_s, D.rv, D.K, D['lo'].where(D['lo'].notna(), None),
              D['hi'].where(D['hi'].notna(), None))]
D['q_rv'] = [gbm_prob(pf, S, ta, sg, K=K, lo=lo, hi=hi) for pf, S, ta, sg, K, lo, hi in
             zip(D.payoff, D.S, D.tau_s, D.rv, D.K, D['lo'].where(D['lo'].notna(), None),
                 D['hi'].where(D['hi'].notna(), None))]
D.to_pickle(os.path.join(DATA, 'trade_level.pkl'))
