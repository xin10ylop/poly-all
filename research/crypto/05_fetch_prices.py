"""Fetch CLOB price history (YES token) for closed crypto threshold markets.
Usage: python 05_fetch_prices.py [hourly_sample_every=6]
Output: data/crypto/prices_<kind>.pkl  {market_id: np.array([[t, p], ...])}"""
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa

every = int(sys.argv[1]) if len(sys.argv) > 1 else 6
evs = [json.loads(l) for l in open(os.path.join(DATA, 'closed_events.jsonl'))]
jobs = {}
hourly_i = 0
for e in sorted(evs, key=lambda x: x['endDate']):
    k, _ = classify(e['slug'])
    k = 'hit_weekly' if k == 'hit_weekly2' else k
    if k == 'above_hourly':
        hourly_i += 1
        if hourly_i % every:
            continue
    for m in e['markets']:
        toks = jl(m['clobTokenIds'])
        if not toks:
            continue
        end = pd.Timestamp(m['endDate'])
        st = pd.Timestamp(m['createdAt']) - pd.Timedelta(minutes=5)
        st = max(st, end - pd.Timedelta(days=40))
        fid = 1 if k == 'above_hourly' else 5
        jobs.setdefault(k, []).append((m['id'], toks[0], st.timestamp(), end.timestamp() + 3600, fid))


def fetch(job):
    mid, tok, s, e, fid = job
    out = []
    cs = s
    while cs < e:                      # API rejects windows longer than ~15 days
        ce = min(cs + 14 * 86400, e)
        for attempt in range(3):
            try:
                out.extend(prices_history(tok, fidelity=fid, start_ts=cs, end_ts=ce))
                break
            except Exception:
                time.sleep(2)
        else:
            return mid, None
        cs = ce
    return mid, np.array(sorted({x['t']: x['p'] for x in out}.items()), dtype=float)


for k, js in jobs.items():
    path = os.path.join(DATA, f'prices_{k}.pkl')
    have = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    js = [j for j in js if j[0] not in have]
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        for i, (mid, arr) in enumerate(ex.map(fetch, js)):
            if arr is not None:
                have[mid] = arr
            if i % 1000 == 999:
                print(k, i + 1, len(js), round(time.time() - t0), flush=True)
                pickle.dump(have, open(path, 'wb'))
    pickle.dump(have, open(path, 'wb'))
    print('done', k, len(have), round(time.time() - t0), flush=True)
