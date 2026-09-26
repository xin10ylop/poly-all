"""Fetch price histories (12h and 10-min) and taker prints for Elon tweet-count bracket markets."""
import json, sys, os, pickle
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

ev = json.load(open('data/elon_events.json'))
mk = [(e['slug'], e['end'], m) for e in ev for m in e['markets'] if m['toks']]
print('markets', len(mk), flush=True)


def hist(x):
    slug, end, m = x
    try:
        h12 = prices_history(m['toks'][0], interval='max', fidelity=720)
        h10 = prices_history(m['toks'][0], interval='max', fidelity=10) if end >= '2026-08-20' else []
        return m['cid'], (h12, h10)
    except Exception:
        return m['cid'], None


H = {}
with ThreadPoolExecutor(16) as ex:
    for cid, h in ex.map(hist, mk):
        if h: H[cid] = h
pickle.dump(H, open('data/elon_hist.pkl', 'wb'))
print('hist', len(H), flush=True)


def tr(x):
    slug, end, m = x
    out, off = [], 0
    try:
        while off <= 9500:
            t = trades(market=m['cid'], limit=500, offset=off)
            out += [(x['timestamp'], x['side'] == 'BUY', x['outcomeIndex'], float(x['price']), float(x['size'])) for x in t]
            if len(t) < 500: break
            off += 500
        return m['cid'], out
    except Exception:
        return m['cid'], None


T = {}
recent = [x for x in mk if x[1] >= '2026-06-15']
print('trade markets', len(recent), flush=True)
with ThreadPoolExecutor(16) as ex:
    for i, (cid, t) in enumerate(ex.map(tr, recent)):
        if t is not None: T[cid] = t
        if i % 200 == 0: print(i, flush=True)
pickle.dump(T, open('data/elon_trades.pkl', 'wb'))
print('done', len(T))
