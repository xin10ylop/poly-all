"""Build the compact universe of non-sports / non-price / non-weather binary resolved markets (vol >= $5k).
Output: data/jev_tags/universe.pkl (one row per market, with the text fields Jev is allowed to see)."""
import json, os, re, datetime as dt
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT = os.path.join(ROOT, 'data', 'jev_tags', 'universe.pkl')
EXCL_TAGS = {'sports', 'games', 'esports', 'crypto-prices', 'recurring', 'weather',
             # asset-price markets (option-pricing problem, not semantic): stock/index/commodity/crypto strikes
             'hit-price', 'finance-updown', 'pyth-finance', 'daily-close', 'up-or-down', 'multi-strikes',
             'stock-prices', 'monthly-hit', 'commodities'}
EXCL_EV_RE = re.compile(r'close-above|closes-above|what-price-will|settle-(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
                        r'|what-will-.*-settle-at|over-under-(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|above-end-of'
                        r'|close-at-in|-hit-(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|hit-by-end-of')
EXCL_SLUG = ('updown', 'up-or-down', 'temperature', 'highest-temp', 'lowest-temp')


def ts(s):
    if not s:
        return None
    x = s.replace('Z', '+00:00')
    if x.endswith('+00'):
        x += ':00'
    try:
        return dt.datetime.fromisoformat(x.replace(' ', 'T')).timestamp()
    except Exception:
        return None


rows = []
with open(os.path.join(ROOT, 'data', 'closed_events.jsonl')) as f:
    for line in f:
        try:
            e = json.loads(line)
        except Exception:
            continue  # partially written last line
        tags = set(e.get('tags') or [])
        if tags & EXCL_TAGS or any(s in e['slug'] for s in EXCL_SLUG) or EXCL_EV_RE.search(e['slug']):
            continue
        for m in e['markets']:
            op = m.get('outcomePrices')
            if not op or len(op) != 2 or not m.get('toks') or not m.get('cid'):
                continue
            try:
                y = float(op[0])
            except Exception:
                continue
            if y not in (0.0, 1.0) or (m.get('vol') or 0) < 5000:
                continue
            if any(s in (m.get('slug') or '') for s in EXCL_SLUG):
                continue
            end = ts(m.get('end') or e.get('end'))
            closed = ts(m.get('closedTime'))
            created = ts(m.get('created') or m.get('start'))
            if not end or not closed or not created:
                continue
            fee = m.get('fee') or {}
            rows.append(dict(mid=m['id'], eid=e['id'], cid=m['cid'], ev_title=e['title'], ev_slug=e['slug'],
                             q=m['q'], mtitle=m.get('title'), desc=m.get('desc') or '', tags=sorted(tags),
                             y=int(y), vol=m['vol'], end=end, closed=closed, created=created,
                             fee_rate=fee.get('rate'), fee_exp=fee.get('exponent', 1), negRisk=bool(m.get('negRisk')),
                             n_ev_mkts=len(e['markets'])))
df = pd.DataFrame(rows).drop_duplicates('mid')
df.to_pickle(OUT)
print(len(df), df.eid.nunique())
print(pd.to_datetime(df.end, unit='s').dt.to_period('M').value_counts().sort_index().to_string())
