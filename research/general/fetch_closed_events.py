"""Fetch closed events (with markets + tags) by end-date range, compact JSONL."""
import json, sys, os, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

start, end = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else 'data/closed_events.jsonl'
seen = set()
if os.path.exists(out_path):
    for l in open(out_path):
        seen.add(json.loads(l)['id'])
f = open(out_path, 'a')
d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
SKIP = ('updown', 'up-or-down', 'highest-temperature', 'lowest-temperature')
n = 0
while d0 < d1:
    dn = d0 + dt.timedelta(days=1)
    try:
        evs = gamma_keyset('events/keyset', {'closed': 'true', 'end_date_min': f'{d0}T00:00:00Z',
                                             'end_date_max': f'{dn}T00:00:00Z'}, key='events')
    except Exception as ex:
        print('ERR', d0, ex, flush=True); d0 = dn; continue
    kept = 0
    for e in evs:
        if e['id'] in seen or any(s in (e.get('slug') or '') for s in SKIP):
            continue
        vol = float(e.get('volume') or 0)
        if vol < 1000:
            continue
        mk = []
        for m in e.get('markets') or []:
            mk.append(dict(id=m['id'], cid=m.get('conditionId'), q=m.get('question'), title=m.get('groupItemTitle'),
                           slug=m.get('slug'), toks=jl(m.get('clobTokenIds')), outcomes=jl(m.get('outcomes')),
                           outcomePrices=jl(m.get('outcomePrices')), vol=float(m.get('volume') or 0),
                           start=m.get('startDate'), end=m.get('endDate'), closedTime=m.get('closedTime'),
                           created=m.get('createdAt'), fee=m.get('feeSchedule'), negRisk=m.get('negRisk'),
                           sportsType=m.get('sportsMarketType'), desc=(m.get('description') or '')[:1500],
                           uma=m.get('umaResolutionStatus'), gameStart=m.get('gameStartTime')))
        rec = dict(id=e['id'], slug=e.get('slug'), title=e.get('title'), end=e.get('endDate'), start=e.get('startDate'),
                   created=e.get('createdAt'), vol=vol, negRisk=e.get('negRisk'),
                   tags=[t.get('slug') for t in (e.get('tags') or [])],
                   series=[s.get('slug') for s in (e.get('series') or [])], markets=mk)
        f.write(json.dumps(rec) + '\n'); seen.add(e['id']); n += 1; kept += 1
    f.flush()
    print(d0, len(evs), kept, n, flush=True)
    d0 = dn
