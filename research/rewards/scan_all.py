"""Snapshot books of every rewarded market and estimate reward capture for a small two-sided quote."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from polylib import *
from reward_share import estimate
import pandas as pd

rw = json.load(open('data/rewards_current.json'))
rw = {x['condition_id']: x for x in rw if x['total_daily_rate'] > 0}
# map condition -> YES token + meta via gamma markets (batched by condition_ids)
cids = list(rw)
meta = {}
for i in range(0, len(cids), 50):
    try:
        ms = get(f"{GAMMA}/markets", {'condition_ids': cids[i:i + 50], 'limit': 50})
    except Exception as ex:
        print('err', ex); continue
    for m in ms:
        toks = jl(m.get('clobTokenIds'))
        if toks:
            meta[m['conditionId']] = dict(tok=toks[0], q=m.get('question'), slug=m.get('slug'),
                                          tick=float(m.get('orderPriceMinTickSize') or 0.01), end=m.get('endDate'),
                                          fee=(m.get('feeSchedule') or {}).get('rate'))
print('meta', len(meta), flush=True)
toks = [v['tok'] for v in meta.values()]
bk = {}
for i in range(0, len(toks), 400):
    for b in books(toks[i:i + 400]):
        bids = sorted([(float(x['price']), float(x['size'])) for x in b.get('bids', [])], reverse=True)
        asks = sorted([(float(x['price']), float(x['size'])) for x in b.get('asks', [])])
        bk[b['asset_id']] = (bids, asks)
rows = []
for cid, m in meta.items():
    if m['tok'] not in bk:
        continue
    r = rw[cid]
    est = estimate(*bk[m['tok']], r['total_daily_rate'], float(r['rewards_max_spread']), float(r['rewards_min_size']),
                   m['tick'])
    if est:
        est.update(cid=cid, q=m['q'], slug=m['slug'], rate=r['total_daily_rate'], minsize=r['rewards_min_size'],
                   maxspread=r['rewards_max_spread'], end=m['end'], fee=m['fee'])
        rows.append(est)
df = pd.DataFrame(rows)
df['roi_day'] = df.usd_day / df.capital
df.to_pickle(f'data/reward_scan_{time.strftime("%Y%m%d%H%M")}.pkl')
print(len(df), 'markets; total est $/day', round(df.usd_day.sum()), 'capital', round(df.capital.sum()))
df = df.sort_values('roi_day', ascending=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(df.head(40)[['q', 'rate', 'minsize', 'mid', 'spread', 'share', 'usd_day', 'capital', 'roi_day', 'end']].round(3).to_string())
df['cat'] = df.slug.str.extract(r'^([a-z]+(?:-[a-z]+)?)')[0]
print(df.groupby('cat').agg(n=('usd_day', 'size'), usd=('usd_day', 'sum'), cap=('capital', 'sum'), rate=('rate', 'sum')).sort_values('usd', ascending=False).head(30))
