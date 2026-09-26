"""Backtest-vs-live consistency: for each live paper fill (bought at the displayed ask), check the real taker prints
in the backtest's fill window (t+5min .. t+35min) on the same market and side:
  - confirmed: at least one print bought the same side at a price <= our live fill price
  - vwap of those prints vs our live price (does the print model assume better/worse prices than the book?)"""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

F = pd.DataFrame([json.loads(l) for l in open('data/live/weather/fills.jsonl')])
rows = []
for r in F.itertuples():
    try:
        tr = get(f'{DATA}/trades', {'market': r.cid, 'limit': 500})
    except Exception:
        continue
    same, allw = [], []
    for x in tr:
        if not (r.ts + 300 < x['timestamp'] <= r.ts + 2100):
            continue
        p = float(x['price']); yes = x['outcomeIndex'] == 0
        yes_buy = (x['side'] == 'BUY') == yes
        pyes = p if yes else 1 - p
        px = pyes if r.side == 'YES' else 1 - pyes
        buys_our_side = yes_buy if r.side == 'YES' else (not yes_buy)
        if buys_our_side:
            allw.append((px, float(x['size'])))
            if px <= r.px + 1e-9:
                same.append((px, float(x['size'])))
    vw = sum(p * s for p, s in same) / sum(s for _, s in same) if same else np.nan
    rows.append(dict(slug=r.slug[:45], bucket=r.bucket, side=r.side, live_px=r.px, live_usd=r.shares * r.px,
                     prints_confirming=len(same), print_vwap=vw, prints_same_side=len(allw),
                     min_print_px=min((p for p, _ in allw), default=np.nan)))
V = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(V.round(3).to_string())
print('\nlive fills', len(V), '| confirmed by later prints at <= live price: %d (%.0f%%)' % ((V.prints_confirming > 0).sum(), 100 * (V.prints_confirming > 0).mean()),
      '| any same-side print in window: %d' % (V.prints_same_side > 0).sum())
print('avg (print_vwap - live_px) where confirmed: %.4f' % (V.print_vwap - V.live_px).mean())
V.to_csv('data/live/weather/fill_validation.csv', index=False)
