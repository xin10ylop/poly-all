"""Hourly 'above' ladders (BTC/ETH, 1h Binance candle close): fetch taker prints for resolved markets with volume,
compare print prices with the model at print time, and evaluate a model-anchored maker.
Output: data/crypto/hourly_trades.pkl + printed tables."""
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa
from bt_models import *  # noqa

evs = [json.loads(l) for l in open(os.path.join(DATA, 'closed_events.jsonl'))]
mk = []
ne = 0
for e in sorted(evs, key=lambda x: x['endDate']):
    k, _ = classify(e['slug'])
    if k != 'above_hourly':
        continue
    ne += 1
    if ne % 3:          # sample every 3rd hourly event
        continue
    for m in e['markets']:
        if float(m.get('volume') or 0) > 0:
            r = parse_market(e, m)
            if r:
                mk.append(r)
print('hourly markets with volume', len(mk), 'total volume $', round(sum(r['vol'] for r in mk)))
path = os.path.join(DATA, 'trades_hourly.pkl')
have = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
todo = [r for r in mk if r['mid_id'] not in have]


def fetch(r):
    for attempt in range(4):
        try:
            t = trades(market=r['cid'], limit=500, offset=0)
            return r['mid_id'], np.array([(x['timestamp'], x['side'], x.get('outcomeIndex'), float(x['price']),
                                           float(x['size'])) for x in t], dtype=object)
        except Exception:
            time.sleep(2 + 2 * attempt)
    return r['mid_id'], None


print('to fetch', len(todo), flush=True)
with ThreadPoolExecutor(16) as ex:
    for i, (mid, a) in enumerate(ex.map(fetch, todo)):
        if a is not None:
            have[mid] = a
        if i % 2000 == 1999:
            pickle.dump(have, open(path, 'wb'))
            print('fetched', i + 1, flush=True)
pickle.dump(have, open(path, 'wb'))

emp = build_empirical()
K1 = {u: load_k(u, '1m') for u in ['BTCUSDT', 'ETHUSDT']}
K1a = {u: (K1[u].index.values.astype('datetime64[s]').astype(np.int64), K1[u]['c'].values) for u in K1}
RVa = {}
for u in K1:
    f = rv_frame(load_k(u, '5m'))
    RVa[u] = (f.index.values.astype('datetime64[s]').astype(np.int64), f['rv1d'].values, f['rv'].values)
rows = []
for r in mk:
    a = have.get(r['mid_id'])
    if a is None or len(a) == 0:
        continue
    y = float(r['outcomePrices'][0])
    if y not in (0.0, 1.0):
        continue
    T = int(r['end'].timestamp())
    for ts_, side, oi, px, sz in a:
        ts_ = int(ts_)
        if ts_ >= T - 30:
            continue
        tk, ck = K1a[r['und']]
        ci = np.searchsorted(tk, ts_ - 60, side='right') - 1
        ri = np.searchsorted(RVa[r['und']][0], ts_, side='right') - 1
        S, rv = ck[ci], RVa[r['und']][2][ri]
        q = emp_prob(emp, 'digital', S, T - ts_, rv, K=r['K'])
        yeq = px if oi == 0 else 1 - px
        ask = (side == 'BUY' and oi == 0) or (side == 'SELL' and oi == 1)
        rows.append((r['mid_id'], r['event'], r['und'], r['K'], T, ts_, yeq, ask, sz, y, S, q))
D = pd.DataFrame(rows, columns=['mid_id', 'event', 'und', 'K', 'T', 't', 'p', 'ask', 'sz', 'y', 'S', 'q'])
D['date'] = pd.to_datetime(D['T'], unit='s', utc=True).dt.floor('D')
D.to_pickle(os.path.join(DATA, 'hourly_trades.pkl'))
D['bp'] = np.where(D.ask, D.p, 1 - D.p)
D['bwin'] = np.where(D.ask, D.y, 1 - D.y)
D['usd'] = D.bp * D.sz
print('prints', len(D), 'markets', D.mid_id.nunique(), 'events', D.event.nunique(), 'taker $', round(D.usd.sum()))
print('taker gross return', round(((D.bwin - D.bp) * D.sz).sum() / D.usd.sum(), 4), ' fee',
      round((taker_fee(D.bp) * D.sz).sum() / D.usd.sum(), 4))
print('prints per event: median', D.groupby('event').size().median(), ' $ per event median',
      round(D.groupby('event').usd.sum().median(), 1))
D['dev'] = D.p - D.q
print('print price - model (YES-eq), abs mean', round(D.dev.abs().mean(), 4), ' corr(p,q)', round(D[['p', 'q']].corr().iloc[0, 1], 3))
print('Brier: prints', round(np.mean((D.p - D.y) ** 2), 4), ' model at print time', round(np.mean((D.q - D.y) ** 2), 4))
for thr in [0.0, 0.02, 0.05, 0.1]:
    fa = D.ask & (D.p > D.q + thr) & (D.q + thr < 0.99)
    fb = ~D.ask & (D.p < D.q - thr) & (D.q - thr > 0.01)
    s = np.minimum(D.sz, 50)
    pnl = np.where(fa, D.q + thr - D.y, np.where(fb, D.y - (D.q - thr), np.nan))
    ok = np.isfinite(pnl)
    byday = pd.Series(pnl[ok] * s[ok]).groupby(D.date[ok].values).sum()
    print(f'maker at model+/-{thr:.2f}: fills={ok.sum()} pnl/share={100 * np.average(pnl[ok], weights=s[ok]):+.2f}c '
          f'total=${(pnl[ok] * s[ok]).sum():.0f} over {D.date.nunique()} days, t_day={byday.mean() / (byday.std() / np.sqrt(len(byday))):+.2f}')
