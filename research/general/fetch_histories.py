"""Fetch 12h-fidelity YES price histories for closed non-sports binary markets."""
import json, sys, os, pickle
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

EXCL_TAGS = {'sports', 'games', 'esports', 'crypto-prices', 'recurring'}
evs = [json.loads(l) for l in open('data/closed_events.jsonl')]
out = 'data/hist12h.pkl'
store = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
jobs = []
for e in evs:
    if EXCL_TAGS & set(e['tags']):
        continue
    for m in e['markets']:
        if m['toks'] and m['vol'] >= 1000 and m['toks'][0] not in store:
            jobs.append(m['toks'][0])
print('jobs', len(jobs), flush=True)


def work(tok):
    try:
        return tok, prices_history(tok, interval='max', fidelity=720)
    except Exception:
        return tok, None


with ThreadPoolExecutor(16) as ex:
    for i, (tok, h) in enumerate(ex.map(work, jobs)):
        if h is not None:
            store[tok] = h
        if i % 3000 == 0:
            print(i, flush=True)
            pickle.dump(store, open(out + '.tmp', 'wb')); os.replace(out + '.tmp', out)
pickle.dump(store, open(out + '.tmp', 'wb')); os.replace(out + '.tmp', out)
print('done', len(store))
