"""Fetch closed crypto threshold events (above ladders, price-range, hit-price) by end date.
Usage: python 03_fetch_closed.py 2026-07-25 2026-09-26"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa

start, end = sys.argv[1], sys.argv[2]
out_path = os.path.join(DATA, 'closed_events.jsonl')
seen = set()
if os.path.exists(out_path):
    for l in open(out_path):
        seen.add(json.loads(l)['slug'])
f = open(out_path, 'a')
d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
n = 0
while d0 < d1:
    dn = d0 + dt.timedelta(days=1)
    kept = 0
    for tag in ('multi-strikes', 'hit-price', 'neg-risk'):
        try:
            evs = gamma_keyset('events/keyset', {'closed': 'true', 'tag_slug': tag, 'end_date_min': f'{d0}T00:00:00Z',
                                                 'end_date_max': f'{dn}T00:00:00Z'}, key='events')
        except Exception as ex:
            print('ERR', d0, tag, ex, flush=True)
            continue
        for e in evs:
            k, _ = classify(e.get('slug') or '')
            if k is None or e['slug'] in seen:
                continue
            ms = []
            for m in e.get('markets') or []:
                ms.append({kk: m.get(kk) for kk in ('id', 'conditionId', 'question', 'groupItemTitle', 'clobTokenIds',
                                                   'outcomes', 'outcomePrices', 'volume', 'endDate', 'createdAt',
                                                   'closedTime', 'closed', 'description', 'orderPriceMinTickSize',
                                                   'umaResolutionStatus', 'acceptingOrders', 'negRisk')})
            rec = {kk: e.get(kk) for kk in ('id', 'slug', 'title', 'endDate', 'startDate', 'createdAt', 'volume')}
            rec['markets'] = ms
            f.write(json.dumps(rec) + '\n')
            seen.add(e['slug'])
            n += 1
            kept += 1
    f.flush()
    print(d0, kept, n, flush=True)
    d0 = dn
