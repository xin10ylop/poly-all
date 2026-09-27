"""Executable 'follow' policy on recorded books: when a same-direction taker print occurs at <= fair-EDGE (fair = latest
model signal at or before the print), buy the ask from the first book snapshot >= REACT seconds after the print, only
if that ask is still <= fair-EDGE. Compares with the print-based assumption (fill at the print price) on the same prints.
Needs: data/stack_fwd_test.parquet (signals), data/wtrades (prints), data/books/weather (60s book snapshots)."""
import os, sys, json, gzip, glob, bisect
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_wtrades
EDGE = float(os.environ.get('EDGE', 0.15)); REACT = float(os.environ.get('REACT', 30)); MAXUSD = 50; FEE = 0.05; TOL = 120
te = pd.read_parquet(os.environ.get('TEST', 'data/stack_fwd_test.parquet'))
tokmap = {}
for l in open('data/weather_events.jsonl'):
    e = json.loads(l)
    for m in e['markets']:
        if m.get('toks'):
            tokmap[m['cid']] = m['toks'][0]
te = te.assign(tok=te.cid.map(tokmap)).dropna(subset=['tok']).sort_values('t')
need = set(te.tok)
snaps = []
for f in sorted(glob.glob('data/books/weather/*.jsonl.gz')):
    with gzip.open(f, 'rt') as h:
        for l in h:
            try:
                d = json.loads(l)
            except Exception:
                continue
            snaps.append((d['ts'], {k: v for k, v in d['b'].items() if k in need}))
snaps.sort(key=lambda x: x[0]); sts = np.array([s[0] for s in snaps])
W = load_wtrades(); W = W[W.cid.isin(set(te.cid))].sort_values('ts')
out = []
for cid, g in te.groupby('cid', sort=False):
    w = W[W.cid == cid]
    if w.empty:
        continue
    tok = g.tok.iloc[0]; Y = g.y.iloc[0]; slug = g.slug.iloc[0]
    gt = g.t.values; gp = g.p.values
    spent = {('follow', 'YES'): 0, ('follow', 'NO'): 0, ('print', 'YES'): 0, ('print', 'NO'): 0}
    for ts, py, yb, sz in zip(w.ts.values, w.pyes.values, w.yes_buy.values, w.sz.values):
        k = np.searchsorted(gt, ts, side='right') - 1
        if k < 0 or ts - gt[k] > 1800:
            continue
        p = gp[k]; side = 'YES' if yb else 'NO'; fair = p if side == 'YES' else 1 - p
        pp = py if side == 'YES' else 1 - py
        if pp > fair - EDGE or pp < 0.02:
            continue
        s = np.searchsorted(sts, ts + REACT)
        if s >= len(sts) or sts[s] - (ts + REACT) > TOL or tok not in snaps[s][1]:
            continue   # no book coverage: skip both so the comparison is on identical prints
        win = Y if side == 'YES' else 1 - Y
        # print-based assumption: half the print size at the print price
        if spent[('print', side)] < MAXUSD:
            take = min(sz * 0.5, (MAXUSD - spent[('print', side)]) / pp); spent[('print', side)] += take * pp
            out.append(('print', slug, cid, side, ts, fair, pp, take, take * (win - pp) - FEE * pp * (1 - pp) * take, win))
        bids, asks = snaps[s][1][tok]
        lv = asks if side == 'YES' else [(1 - b, z) for b, z in sorted(bids, key=lambda x: -x[0])]
        for px, z in sorted(lv):
            if px > fair - EDGE or spent[('follow', side)] >= MAXUSD:
                break
            take = min(z, (MAXUSD - spent[('follow', side)]) / px)
            if take < 5:
                continue
            spent[('follow', side)] += take * px
            out.append(('follow', slug, cid, side, ts, fair, px, take, take * (win - px) - FEE * px * (1 - px) * take, win))
D = pd.DataFrame(out, columns=['mode', 'slug', 'cid', 'side', 'ts', 'fair', 'px', 'shares', 'pnl', 'win'])
D['cost'] = D.shares * D.px
for m, x in D.groupby('mode'):
    e = x.groupby('slug').pnl.sum().sort_values()
    print(f"{m:7s} fills {len(x):4d} events {x.slug.nunique():3d} cost ${x.cost.sum():6.0f} pnl ${x.pnl.sum():7.1f} ret {x.pnl.sum()/x.cost.sum():+.1%} "
          f"ex-top {(x.pnl.sum()-e.iloc[-1])/(x.cost.sum()-x[x.slug==e.index[-1]].cost.sum()):+.1%} avg px {x.px.mean():.3f}")
    for sd, y in x.groupby('side'):
        print(f"   {sd}: cost ${y.cost.sum():.0f} pnl ${y.pnl.sum():.1f} ret {y.pnl.sum()/y.cost.sum():+.1%}")
D.to_pickle('data/bt_follow.pkl')
