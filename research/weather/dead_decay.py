"""After a bucket becomes 'dead' per METAR (running extreme passes it), how much taker volume trades
at non-trivial YES prices, as a function of delay from the METAR observation time?"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
from wlib import *

evs = load_events()
T = pickle.load(open('data/weather_trades.pkl', 'rb'))
metars = {}
rows = []
for e in evs:
    if e['icao'] not in metars:
        metars[e['icao']] = load_metar(e['icao'], e['unit'])
    mt = metars[e['icao']]
    if mt is None:
        continue
    w = mt[(mt.ts >= e['day_start']) & (mt.ts < e['day_end'])]
    if len(w) < 8:
        continue
    vals = w.t.values if e['kind'] == 'high' else -w.t.values
    run = np.maximum.accumulate(vals)
    for b in e['buckets']:
        lo, hi = b['rng']
        if b['cid'] not in T:
            continue
        # time bucket becomes dead: first obs where running extreme passes the bucket
        thr = hi if e['kind'] == 'high' else -lo
        idx = np.nonzero(run > thr)[0]
        if len(idx) == 0:
            continue   # never dead intraday (e.g., bucket above the final max) -> skip
        tdead = w.ts.values[idx[0]]
        for x in T[b['cid']]:
            dtm = (x['timestamp'] - tdead) / 60
            if dtm < -30 or dtm > 600:
                continue
            yes = x['outcomeIndex'] == 0
            buy_yes = (x['side'] == 'BUY') == yes
            p = float(x['price']); pyes = p if yes else 1 - p
            sz = float(x['size'])
            rows.append((e['slug'], b['title'], dtm, buy_yes, pyes, sz, x['proxyWallet'], bool(b['won']), e['city'], e['kind']))
df = pd.DataFrame(rows, columns=['slug', 'bucket', 'dtm', 'buy_yes', 'pyes', 'sz', 'taker', 'won', 'city', 'kind'])
df.to_pickle('data/dead_trades.pkl')
# opportunity = taker BUY_NO (buy_yes False) at pyes >= 0.01 after dead time: profit per share = pyes - fee
d = df[(~df.buy_yes) & (df.pyes >= 0.005) & (df.dtm >= 0)].copy()
d['profit'] = d.sz * (d.pyes - d.won.astype(float) - 0.05 * d.pyes * (1 - d.pyes))
d['win'] = pd.cut(d.dtm, [0, 1, 2, 3, 5, 10, 20, 30, 60, 120, 600])
g = d.groupby('win', observed=True).agg(n=('sz', 'size'), shares=('sz', 'sum'), profit=('profit', 'sum'),
                                          pyes=('pyes', 'median'), events=('slug', 'nunique'), won=('won', 'mean'))
print(g.round(2))
print('days covered', df.slug.str.extract(r'-on-(.*)')[0].nunique())
# also: how many dead-bucket events had ANY bid (maker selling NO... ) i.e. taker BUY_NO after 5 min
