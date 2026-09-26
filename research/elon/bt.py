"""Backtest the Elon post-count nowcast vs market; fills only on real prints after the signal."""
import json, pickle, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from model import load_posts, Nowcast
from calib import windows

EDGE = float(os.environ.get('EDGE', 0.05)); DELAY = 300; WIN = 1800; SHARE = 0.5
MAXUSD = float(os.environ.get('MAXUSD', 100)); STEP = 1800; SPLIT = os.environ.get('SPLIT', '2026-08-01')
R = float(os.environ.get('R', 8))
P = load_posts(); nc = Nowcast(P, 0.1, 0.3, 0.6, r=R)
T = pickle.load(open('data/elon_trades.pkl', 'rb'))
ev = {e['slug']: e for e in json.load(open('data/elon_events.json'))}
W = [w for w in windows() if w['end'] >= '2026-06-15']
sig, fills = [], []
for w in W:
    e = ev[w['slug']]
    rate = (e['markets'][0].get('fee') or {}).get('rate', 0.05) or 0.05
    bk = w['bk']
    tape = {}
    for b in bk:
        tr = T.get(b['cid'])
        if not tr:
            continue
        a = np.array(sorted(tr))
        ts = a[:, 0]; buy = a[:, 1].astype(bool); oi = a[:, 2]; px = a[:, 3]; sz = a[:, 4]
        pyes = np.where(oi == 0, px, 1 - px); yb = buy == (oi == 0)
        tape[b['cid']] = (ts, pyes, yb, sz, np.zeros(len(ts)))
    spent = {}
    for t in np.arange(w['s'] + 6 * 3600, w['e'] - 1800, STEP):
        C, mu, q = nc.bucket_probs(w['s'], t, w['e'], [(b['lo'], b['hi']) for b in bk])
        for j, b in enumerate(bk):
            if b['cid'] not in tape:
                continue
            ts, pyes, yb, sz, used = tape[b['cid']]
            Y = 1.0 if b['won'] else 0.0
            i0 = np.searchsorted(ts, t, side='right') - 1
            ref = pyes[i0] if i0 >= 0 else np.nan
            sig.append((w['slug'], w['end'], j, t, (w['e'] - t) / 3600, q[j], ref, Y, C, mu))
            lo_i = np.searchsorted(ts, t + DELAY, side='right'); hi_i = np.searchsorted(ts, t + DELAY + WIN, side='right')
            for i in range(lo_i, hi_i):
                for side in ('YES', 'NO'):
                    key = (b['cid'], side)
                    if spent.get(key, 0) >= MAXUSD:
                        continue
                    if side == 'YES' and yb[i] and pyes[i] <= q[j] - EDGE:
                        px = pyes[i]
                    elif side == 'NO' and (not yb[i]) and (1 - pyes[i]) <= (1 - q[j]) - EDGE:
                        px = 1 - pyes[i]
                    else:
                        continue
                    if px < 0.01 or px > 0.99:
                        continue
                    avail = sz[i] * SHARE - used[i]
                    if avail <= 0:
                        continue
                    take = min(avail, (MAXUSD - spent.get(key, 0)) / px); used[i] += take
                    spent[key] = spent.get(key, 0) + take * px
                    win = Y if side == 'YES' else 1 - Y
                    fills.append((w['slug'], w['end'], j, side, t, (w['e'] - t) / 3600, q[j] if side == 'YES' else 1 - q[j],
                                  px, take, take * (win - px) - rate * px * (1 - px) * take, win))
S = pd.DataFrame(sig, columns=['slug', 'end', 'j', 't', 'hleft', 'q', 'ref', 'y', 'C', 'mu'])
F = pd.DataFrame(fills, columns=['slug', 'end', 'j', 'side', 't', 'hleft', 'q', 'px', 'shares', 'pnl', 'win'])
F['cost'] = F.shares * F.px
S.to_pickle('data/elon_sig.pkl'); F.to_pickle(f'data/elon_fills_e{EDGE}.pkl')
s = S.dropna(subset=['ref'])
s['hb'] = pd.cut(s.hleft, [0, 6, 24, 48, 96, 1000])
print('Brier by hours left: market(last print) vs model')
print(s.groupby('hb', observed=True).apply(lambda x: pd.Series(dict(n=len(x), market=np.mean((x.ref - x.y) ** 2), model=np.mean((x.q - x.y) ** 2), blend=np.mean(((x.q + x.ref) / 2 - x.y) ** 2)))).round(5))
for per, f in [('pre-' + SPLIT, F[F.end < SPLIT]), ('post-' + SPLIT, F[F.end >= SPLIT])]:
    print(per, 'fills', len(f), 'windows', f.slug.nunique(), 'cost $%.0f pnl $%.0f ret %.3f' % (f.cost.sum(), f.pnl.sum(), f.pnl.sum() / max(1, f.cost.sum())))
    if len(f):
        print(f.groupby('side').agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum'), win=('win', 'mean'), q=('q', 'mean'), px=('px', 'mean')).round(3))
        print(f.groupby(pd.cut(f.hleft, [0, 6, 24, 48, 96, 1000]), observed=True).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum')).round(1))
print(F.groupby('slug').pnl.sum().describe().round(1))
