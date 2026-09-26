"""Build backtest panel: resolved markets x decision horizons -> market mid, spot, vol, model probs, outcome.
Output: data/crypto/panel.pkl"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa
from bt_models import *  # noqa

HZ = {'default': [7 * 1440, 5 * 1440, 3 * 1440, 2 * 1440, 1440, 1080, 720, 360, 180, 120, 60, 30],
      'above_hourly': [60, 45, 30, 20, 10, 5]}
MAX_STALE = {'default': 30 * 60, 'above_hourly': 3 * 60}

evs = [json.loads(l) for l in open(os.path.join(DATA, 'closed_events.jsonl'))]
prices = {}
for k in ['above_daily', 'above_hourly', 'range_daily', 'hit_daily', 'hit_weekly', 'hit_monthly']:
    p = os.path.join(DATA, f'prices_{k}.pkl')
    if os.path.exists(p):
        prices.update(pickle.load(open(p, 'rb')))
print('price series', len(prices), flush=True)

emp = build_empirical()
K1 = {u: load_k(u, '1m') for u in UNDS}
K1a = {u: (K1[u].index.values.astype('datetime64[s]').astype(np.int64), K1[u]['c'].values, K1[u]['h'].values,
           K1[u]['l'].values) for u in UNDS}
RV = {u: rv_frame(load_k(u, '5m')) for u in UNDS}
RVa = {u: (RV[u].index.values.astype('datetime64[s]').astype(np.int64), RV[u]['rv'].values, RV[u]['rv1d'].values)
       for u in UNDS}
DV = {}
for u, c in DVOL.items():
    d = pd.read_pickle(os.path.join(DATA, f'dvol_{c}.pkl'))
    DV[u] = (d.index.values.astype('datetime64[s]').astype(np.int64) + 3600, d.values)  # hourly close avail at +1h


def last_before(tarr, varr, t):
    i = np.searchsorted(tarr, t, side='right') - 1
    return (varr[i], tarr[i]) if i >= 0 else (np.nan, None)


rows = []
for e in evs:
    for m in e['markets']:
        r = parse_market(e, m)
        if r is None:
            continue
        op = r['outcomePrices']
        try:
            y = float(op[0])
        except Exception:
            continue
        if y not in (0.0, 1.0):
            continue
        arr = prices.get(m['id'])
        if arr is None or len(arr) == 0:
            continue
        u = r['und']
        kind = 'hit_weekly' if r['kind'] == 'hit_weekly2' else r['kind']
        T = int(r['end'].timestamp())
        created = int(r['created'].timestamp())
        tk, ck, hk, lk = K1a[u]
        for hmin in HZ.get(kind, HZ['default']):
            t = T - hmin * 60
            if t < created + 600 or t < tk[0] + 86400:
                continue
            pi = np.searchsorted(arr[:, 0], t, side='right') - 1
            if pi < 0 or t - arr[pi, 0] > MAX_STALE.get(kind, MAX_STALE['default']):
                continue
            mid = arr[pi, 1]
            # spot: close of the last complete 1m candle before t
            ci = np.searchsorted(tk, t - 60, side='right') - 1
            if ci < 0:
                continue
            S = ck[ci]
            hi_sf = lo_sf = np.nan
            if r['payoff'].startswith('touch'):
                ws = int(r['win_start'].timestamp())
                if t < ws:
                    continue
                a, b = np.searchsorted(tk, ws, side='left'), ci + 1
                if b > a:
                    hi_sf, lo_sf = hk[a:b].max(), lk[a:b].min()
                    if (r['payoff'] == 'touch_up' and hi_sf >= r['K']) or (r['payoff'] == 'touch_down' and lo_sf <= r['K']):
                        continue      # already touched -> market already resolved
            rv, _ = last_before(RVa[u][0], RVa[u][1], t)
            rv1d, _ = last_before(RVa[u][0], RVa[u][2], t)
            dv = last_before(DV[u][0], DV[u][1], t)[0] if u in DV else np.nan
            tau = T - t
            kw = dict(K=r.get('K'), lo=r.get('lo'), hi=r.get('hi'))
            rows.append(dict(mid_id=m['id'], event=e['slug'], kind=kind, und=u, payoff=r['payoff'], K=r.get('K'),
                             lo=r.get('lo'), hi=r.get('hi'), title=r['title'], T=T, t=t, h_min=hmin, tau_s=tau,
                             p_mkt=mid, S=S, hi_sf=hi_sf, lo_sf=lo_sf, rv=rv, rv1d=rv1d, dvol=dv, y=y,
                             vol=r['vol'], tick=r['tick'],
                             p_rv=gbm_prob(r['payoff'], S, tau, rv, **kw),
                             p_dvol=gbm_prob(r['payoff'], S, tau, dv, **kw) if np.isfinite(dv) else np.nan,
                             p_emp=emp_prob(emp, r['payoff'], S, tau, rv, **kw)))
panel = pd.DataFrame(rows)
panel.to_pickle(os.path.join(DATA, 'panel.pkl'))
print(panel.groupby(['kind', 'h_min']).size().unstack(0), flush=True)
