"""Fetch closed daily temperature events (high/low) with market metadata + resolution."""
import json, sys, os, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

out_path = 'data/weather_events.jsonl'
start = sys.argv[1] if len(sys.argv) > 1 else '2026-07-20'
end = sys.argv[2] if len(sys.argv) > 2 else '2026-09-26'
seen = set()
if os.path.exists(out_path):
    for l in open(out_path):
        seen.add(json.loads(l)['slug'])
f = open(out_path, 'a')
d0 = dt.date.fromisoformat(start)
d1 = dt.date.fromisoformat(end)
n = 0
while d0 < d1:
    dn = d0 + dt.timedelta(days=1)
    evs = gamma_paginate('events', {'tag_slug': 'weather', 'closed': 'true',
                                    'end_date_min': f'{d0}T00:00:00Z', 'end_date_max': f'{dn}T00:00:00Z'}, page=100)
    for e in evs:
        if 'temperature' not in e['slug'] or e['slug'] in seen:
            continue
        mk = []
        for m in e['markets']:
            mk.append(dict(id=m['id'], cid=m['conditionId'], title=m.get('groupItemTitle'), q=m['question'],
                           toks=jl(m.get('clobTokenIds')), outcomePrices=jl(m.get('outcomePrices')),
                           vol=float(m.get('volume') or 0), tick=m.get('orderPriceMinTickSize'),
                           fee=m.get('feeSchedule'), rewMin=m.get('rewardsMinSize'), rewSpread=m.get('rewardsMaxSpread'),
                           created=m.get('createdAt'), closedTime=m.get('closedTime'), uma=m.get('umaResolutionStatus')))
        rec = dict(slug=e['slug'], title=e['title'], endDate=e['endDate'], startDate=e.get('startDate'),
                   created=e.get('createdAt'), vol=float(e.get('volume') or 0), negRisk=e.get('negRisk'),
                   desc=e.get('description'), resolutionSource=e.get('resolutionSource'), markets=mk)
        f.write(json.dumps(rec) + '\n'); seen.add(e['slug']); n += 1
    f.flush()
    print(d0, len(evs), n, flush=True)
    d0 = dn
