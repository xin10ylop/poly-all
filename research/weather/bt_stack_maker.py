"""Maker variant: at grid time t post a resting BUY at b = floor((fair - EDGE_M)/tick)*tick for 30 min.
Filled only if a real taker SELL print (someone hit bids) occurs at price <= b inside (t+DELAY, t+DELAY+WIN];
fill price = b (our limit). Optimistic on queue priority (assumes our bid was best), pessimistic on price."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_wtrades

EDGE = float(os.environ.get('EDGE', 0.10)); DELAY = 60; WIN = 1800; SHARE = float(os.environ.get('SHARE', 0.5))
MAXUSD = 50; REB = 0.25 * 0.05     # maker rebate share of taker fee (weather) -> per share rebate ~ REB*p(1-p)
te = pd.read_parquet(os.environ.get('TEST', 'data/stack_test_2026-08-01.parquet'))
W = load_wtrades(); W = W[W.cid.isin(set(te.cid))]
byc = {c: (g.ts.values, g.pyes.values, g.yes_buy.values, g.sz.values) for c, g in W.groupby('cid', sort=False)}
del W
out = []
for (slug, j), g in te.groupby(['slug', 'j'], sort=False):
    cid = g.cid.iloc[0]
    if cid not in byc:
        continue
    ts, py, yb, sz = byc[cid]; used = np.zeros(len(ts)); spent = {'YES': 0.0, 'NO': 0.0}; Y = g.y.iloc[0]
    for r in g.itertuples():
        lo_i = np.searchsorted(ts, r.t + DELAY, side='right'); hi_i = np.searchsorted(ts, r.t + DELAY + WIN, side='right')
        for side in ('YES', 'NO'):
            fair = r.p if side == 'YES' else 1 - r.p
            b = np.floor((fair - EDGE) * 100) / 100
            if b < 0.02 or b > 0.95 or spent[side] >= MAXUSD:
                continue
            for i in range(lo_i, hi_i):
                # YES bid is hit by a taker selling YES (yb False) at pyes <= b ; NO bid hit by taker selling NO (yb True) at 1-pyes <= b
                hit = ((side == 'YES' and (not yb[i]) and py[i] <= b) or (side == 'NO' and yb[i] and (1 - py[i]) <= b))
                if not hit:
                    continue
                avail = sz[i] * SHARE - used[i]
                if avail <= 0:
                    continue
                take = min(avail, (MAXUSD - spent[side]) / b); used[i] += take; spent[side] += take * b
                win = Y if side == 'YES' else 1 - Y
                out.append((slug, r.kind, r.date, side, r.hr, fair, b, take, take * (win - b) + REB * b * (1 - b) * take, win))
                if spent[side] >= MAXUSD:
                    break
df = pd.DataFrame(out, columns=['slug', 'kind', 'date', 'side', 'hr', 'fair', 'px', 'shares', 'pnl', 'win'])
df['cost'] = df.shares * df.px
d = df.groupby('date').pnl.sum()
print(f'MAKER EDGE={EDGE} SHARE={SHARE}: fills {len(df)} cost ${df.cost.sum():.0f} pnl ${df.pnl.sum():.0f} ret {df.pnl.sum()/df.cost.sum():.3f} $/day {d.mean():.0f} posdays {(d>0).mean():.2f} sharpe {d.mean()/d.std()*np.sqrt(365):.1f}')
print(df.groupby(['kind', 'side']).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum'), win=('win', 'mean'), fair=('fair', 'mean'), px=('px', 'mean')).round(3))
df.to_pickle(f'data/bt_maker_e{EDGE}.pkl')
