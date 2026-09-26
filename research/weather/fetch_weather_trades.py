"""Fetch all taker trades for weather markets (data-api /trades by conditionId)."""
import json, sys, os, pickle, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

min_end = sys.argv[1] if len(sys.argv) > 1 else '2026-06-20'
evs = [json.loads(l) for l in open('data/weather_events.jsonl')]
evs = [e for e in evs if e['endDate'] >= min_end]
out = 'data/weather_trades.pkl'
store = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
jobs = [m['cid'] for e in evs for m in e['markets'] if m['cid'] not in store and m['vol'] > 0]
print('jobs', len(jobs), flush=True)
KEEP = ('side', 'asset', 'size', 'price', 'timestamp', 'outcomeIndex', 'proxyWallet')


def work(cid):
    allt, off = [], 0
    try:
        while True:
            t = trades(market=cid, limit=500, offset=off)
            allt.extend({k: x.get(k) for k in KEEP} for x in t)
            if len(t) < 500 or off >= 9500:
                break
            off += 500
        return cid, allt
    except Exception:
        return cid, None


with ThreadPoolExecutor(16) as ex:
    for i, (cid, t) in enumerate(ex.map(work, jobs)):
        if t is not None:
            store[cid] = t
        if i % 3000 == 0:
            print(i, flush=True)
            pickle.dump(store, open(out + '.tmp', 'wb')); os.replace(out + '.tmp', out)
pickle.dump(store, open(out + '.tmp', 'wb')); os.replace(out + '.tmp', out)
print('done', len(store))
