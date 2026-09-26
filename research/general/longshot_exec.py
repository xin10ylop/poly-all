"""Re-price the longshot-NO signal using actual taker trade prices after the signal time."""
import json, pickle, sys, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

df = pd.read_pickle('data/longshot_panel.pkl')
evs = [json.loads(l) for l in open('data/closed_events.jsonl')]
cid = {m['id']: (m['cid'], m['end']) for e in evs for m in e['markets']}
cand = df[(df.p >= 0.05) & (df.p <= 0.6) & (df.hd.isin([1, 2, 3, 5, 7]))].copy()
mids = cand.mid.unique()
print('markets', len(mids), flush=True)
out = 'data/longshot_trades.pkl'
T = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}


def work(m):
    c = cid[m][0]
    allt, off = [], 0
    try:
        while off < 5000:
            t = trades(market=c, limit=500, offset=off)
            allt += [(x['timestamp'], x['side'], x['outcomeIndex'], float(x['price']), float(x['size'])) for x in t]
            if len(t) < 500:
                break
            off += 500
        return m, allt
    except Exception:
        return m, None


with ThreadPoolExecutor(12) as ex:
    for i, (m, t) in enumerate(ex.map(work, [m for m in mids if m not in T])):
        if t is not None:
            T[m] = t
        if i % 1000 == 0:
            print(i, flush=True)
pickle.dump(T, open(out, 'wb'))
print('done', len(T))
