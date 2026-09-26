"""Fetch price history (YES token, 10-min native granularity) for weather markets."""
import json, sys, os, pickle
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

min_end = sys.argv[1] if len(sys.argv) > 1 else '2026-08-27'
evs = [json.loads(l) for l in open('data/weather_events.jsonl')]
evs = [e for e in evs if e['endDate'] >= min_end]
out = 'data/weather_prices.pkl'
store = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
jobs = [m['toks'][0] for e in evs for m in e['markets'] if m['toks'] and m['toks'][0] not in store]
print('events', len(evs), 'jobs', len(jobs), flush=True)

def work(tok):
    try:
        return tok, prices_history(tok, interval='max', fidelity=1)
    except Exception as ex:
        return tok, None

with ThreadPoolExecutor(24) as ex:
    for i, (tok, h) in enumerate(ex.map(work, jobs)):
        if h is not None:
            store[tok] = h
        if i % 2000 == 0:
            print(i, flush=True)
            pickle.dump(store, open(out, 'wb'))
pickle.dump(store, open(out, 'wb'))
print('done', len(store))
