"""Fetch taker fills (data-api /trades) for closed non-hourly crypto threshold markets.
Output: data/crypto/trades.pkl  DataFrame[mid, t, side, outcomeIndex, price, size]"""
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa

evs = [json.loads(l) for l in open(os.path.join(DATA, 'closed_events.jsonl'))]
jobs = []
for e in evs:
    k, _ = classify(e['slug'])
    if k == 'above_hourly':
        continue
    for m in e['markets']:
        if float(m.get('volume') or 0) > 0 and m.get('conditionId'):
            jobs.append((m['id'], m['conditionId']))
path = os.path.join(DATA, 'trades.pkl')
have = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
jobs = [j for j in jobs if j[0] not in have]
print('jobs', len(jobs), flush=True)


def fetch(job):
    mid, cid = job
    out, off = [], 0
    while True:
        for attempt in range(4):
            try:
                t = trades(market=cid, limit=500, offset=off)
                break
            except Exception:
                time.sleep(2 + attempt * 3)
        else:
            return mid, None
        out.extend((x['timestamp'], x['side'], x.get('outcomeIndex'), float(x['price']), float(x['size'])) for x in t)
        if len(t) < 500 or off >= 20000:
            break
        off += 500
    return mid, np.array(out, dtype=object)


t0 = time.time()
with ThreadPoolExecutor(12) as ex:
    for i, (mid, arr) in enumerate(ex.map(fetch, jobs)):
        if arr is not None:
            have[mid] = arr
        if i % 1000 == 999:
            print(i + 1, len(jobs), round(time.time() - t0), flush=True)
            pickle.dump(have, open(path, 'wb'))
pickle.dump(have, open(path, 'wb'))
print('done', len(have), round(time.time() - t0), flush=True)
