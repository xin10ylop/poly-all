"""Calibration of non-sports binary markets vs time to scheduled end (12h price buckets).
Question: does buying NO on low-priced YES (or YES on high-priced) earn money, by horizon/category?"""
import json, pickle, sys, os
import numpy as np, pandas as pd, datetime as dt

EXCL = {'sports', 'games', 'esports', 'crypto-prices', 'recurring'}
evs = [json.loads(l) for l in open('data/closed_events.jsonl')]
H = pickle.load(open('data/hist12h.pkl', 'rb'))


def ts(s):
    try:
        return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
    except Exception:
        return None


rows = []
for e in evs:
    tags = set(e['tags'])
    if tags & EXCL:
        continue
    cat = next((t for t in ['politics', 'geopolitics', 'crypto', 'finance', 'economy', 'pop-culture', 'tech',
                            'mention-markets', 'world', 'elections', 'business', 'science', 'weather'] if t in tags), 'other')
    for m in e['markets']:
        op = m.get('outcomePrices')
        if not op or len(op) != 2 or not m['toks']:
            continue
        try:
            y = float(op[0])
        except Exception:
            continue
        if y not in (0.0, 1.0):
            continue
        h = H.get(m['toks'][0])
        end = ts(m.get('end') or e.get('end') or '')
        if not h or not end:
            continue
        t = np.array([x['t'] for x in h]); p = np.array([x['p'] for x in h])
        closed = ts(m.get('closedTime') or '') or t[-1]
        for hd in [0.5, 1, 2, 3, 5, 7, 14, 30]:
            te = end - hd * 86400
            if te > closed - 3600 or te < t[0]:
                continue
            i = np.searchsorted(t, te, side='right') - 1
            if i < 0:
                continue
            rows.append((m['id'], e['id'], cat, m['q'], hd, p[i], y, m['vol'], (closed - te) / 86400, bool(m.get('negRisk'))))
df = pd.DataFrame(rows, columns=['mid', 'eid', 'cat', 'q', 'hd', 'p', 'y', 'vol', 'hold_days', 'negRisk'])
df.to_pickle('data/longshot_panel.pkl')
print(len(df), df.mid.nunique())
df['pb'] = pd.cut(df.p, [0, .01, .03, .05, .1, .2, .35, .5, .65, .8, .9, .95, .97, .99, 1])
g = df.groupby(['hd', 'pb'], observed=True).agg(n=('y', 'size'), p=('p', 'mean'), yes=('y', 'mean'))
g['edge_no'] = g.p - g.yes      # expected profit per $1 face buying NO at (1-p)
print(g.round(3).to_string())
