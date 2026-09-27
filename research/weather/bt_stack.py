"""Trade the stacked model on the out-of-sample period, filling only on real prints after the signal."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_wtrades

EDGE = float(os.environ.get('EDGE', 0.05)); DELAY = float(os.environ.get('DELAY', 300)); WIN = float(os.environ.get('WIN', 1800))
SHARE = float(os.environ.get('SHARE', 0.5)); MAXUSD = float(os.environ.get('MAXUSD', 50)); FEE = 0.05
PMIN = float(os.environ.get('PMIN', 0.02)); PMAX = float(os.environ.get('PMAX', 0.98))
# FILLMODE: 'follow' (default) = fill only alongside a same-direction taker print at <= fair-EDGE;
# 'any' = also fill on opposite-direction prints (a seller hitting the bid), paying that price + PEN (spread proxy);
# 'against' = ONLY opposite-direction prints (+PEN): the moments a displayed-ask taker gets that the default skips
FILLMODE = os.environ.get('FILLMODE', 'follow'); PEN = float(os.environ.get('PEN', 0.01))
te = pd.read_parquet(os.environ.get('TEST', 'data/stack_test.parquet'))
W = load_wtrades()
W = W[W.cid.isin(set(te.cid))]
byc = {c: (g.ts.values, g.pyes.values, g.yes_buy.values, g.sz.values) for c, g in W.groupby('cid', sort=False)}
del W
out = []
for (slug, j), g in te.groupby(['slug', 'j'], sort=False):
    cid = g.cid.iloc[0]
    if cid not in byc:
        continue
    ts, py, yb, sz = byc[cid]
    used = np.zeros(len(ts)); spent = {'YES': 0.0, 'NO': 0.0}
    Y = g.y.iloc[0]
    for r in g.itertuples():
        lo_i = np.searchsorted(ts, r.t + DELAY, side='right'); hi_i = np.searchsorted(ts, r.t + DELAY + WIN, side='right')
        for i in range(lo_i, hi_i):
            for side in ('YES', 'NO'):
                if spent[side] >= MAXUSD:
                    continue
                same = bool(yb[i]) if side == 'YES' else not yb[i]
                if (FILLMODE == 'follow' and not same) or (FILLMODE == 'against' and same):
                    continue
                px = (py[i] if side == 'YES' else 1 - py[i]) + (0 if same else PEN)
                if px > (r.p if side == 'YES' else 1 - r.p) - EDGE:
                    continue
                if px < PMIN or px > PMAX:
                    continue
                avail = sz[i] * SHARE - used[i]
                if avail <= 0:
                    continue
                take = min(avail, (MAXUSD - spent[side]) / px); used[i] += take; spent[side] += take * px
                win = Y if side == 'YES' else 1 - Y
                out.append((slug, r.city, r.kind, r.date, j, side, r.t, ts[i], r.hr, r.p if side == 'YES' else 1 - r.p,
                            px, take, take * (win - px) - FEE * px * (1 - px) * take, win, r.src))
df = pd.DataFrame(out, columns=['slug', 'city', 'kind', 'date', 'j', 'side', 'tsig', 'tfill', 'hr', 'p', 'px', 'shares', 'pnl', 'win', 'src'])
df['cost'] = df.shares * df.px
tag = os.environ.get('TAG', 'base'); df.to_pickle(f'data/bt_stack_{tag}.pkl')
print(f'EDGE={EDGE} DELAY={DELAY} WIN={WIN} SHARE={SHARE} MAXUSD={MAXUSD} PMIN={PMIN} PMAX={PMAX}')
print('fills', len(df), 'events', df.slug.nunique(), 'cost $%.0f  pnl $%.0f  ret %.3f' % (df.cost.sum(), df.pnl.sum(), df.pnl.sum() / max(df.cost.sum(), 1)))
for col in ['side', 'kind', 'src']:
    print(df.groupby(col).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum'), win=('win', 'mean'), p=('p', 'mean'), px=('px', 'mean')).round(3))
df['hb'] = pd.cut(df.hr, [-1, 6, 10, 13, 16, 19, 24])
print(df.groupby('hb', observed=True).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum'), win=('win', 'mean'), p=('p', 'mean'), px=('px', 'mean')).round(3))
df['pxb'] = pd.cut(df.px, [0, .05, .1, .3, .5, .7, .9, .95, 1])
print(df.groupby('pxb', observed=True).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum'), win=('win', 'mean'), p=('p', 'mean')).round(3))
d = df.groupby('date').agg(pnl=('pnl', 'sum'), cost=('cost', 'sum'))
print('days %d  mean pnl/day $%.1f  std %.1f  positive days %.2f  mean cost/day $%.0f' % (len(d), d.pnl.mean(), d.pnl.std(), (d.pnl > 0).mean(), d.cost.mean()))
